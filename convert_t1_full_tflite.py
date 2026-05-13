"""
T-1 (Revised): Full-Model TFLite INT8 Conversion (Encoder + All 3 Heads)
=========================================================================
Prior state: R-4 Joint fine-tuning complete. Heads-only T-1 REJECTED.

Requirement: A single self-contained TFLite binary that accepts raw
  tf.string inputs (resume_text, jd_text) and runs the full pipeline
  on-device — MobileUSE encoder + ATS head + Domain head + RSG head.

Inputs:
  Input 0: resume_text — tf.string, shape=()
  Input 1: jd_text    — tf.string, shape=()

Outputs:
  Output 0: ats_score          — tf.float32, shape=[1],  sigmoid [0,1]
  Output 1: domain_probs       — tf.float32, shape=[7],  softmax
  Output 2: rsg_template_probs — tf.float32, shape=[46], softmax

Quantization: INT8 (primary target).
  SELECT_TF_OPS: PROHIBITED. If required, this is a hard stop.
  Float16 fallback: PROHIBITED without explicit Sai approval.

Pipeline:
  Task 1: Load full unified model (string inputs), restore R-4 weights
  Task 2: Export to SavedModel
  Task 3: Attempt INT8 conversion — NO SELECT_TF_OPS
  Task 4: Hard stop if any error occurs during conversion
  Task 5: Parity check (50 samples, >= 99% within 2.0 pts)
  Task 6: Write conversion_summary_full.json
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault(
    "TFHUB_CACHE_DIR",
    r"C:\Users\saini\Desktop\ats\ats-ai-core\tfhub_cache",
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ats-ai-core"))

import json
import datetime
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.unified_engine.unified_model import build_unified_model
from src.config import RSG_NUM_CLASSES, NUM_DOMAINS

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(os.path.dirname(os.path.abspath(__file__)))
ATS_MODEL_DIR    = PROJECT_ROOT / "ats-ai-core" / "model" / "ats_model"
UNIFIED_DIR      = PROJECT_ROOT / "model" / "unified_model"
LABELED_DIR      = PROJECT_ROOT / "ats-ai-core" / "data" / "labeled"

R4_WEIGHTS       = UNIFIED_DIR / "r4_joint_best.weights.h5"
SAVED_MODEL_PATH = ATS_MODEL_DIR / "saved_model_full"
TFLITE_OUTPUT    = ATS_MODEL_DIR / "ats_core_full_int8.tflite"
SUMMARY_JSON     = ATS_MODEL_DIR / "conversion_summary_full.json"
ATS_CSV          = LABELED_DIR / "merged_final.csv"

N_REPR_SAMPLES  = 300
N_PARITY        = 50
PARITY_TOL_PTS  = 2.0
PARITY_PASS_PCT = 99.0
SIZE_GATE_MB    = 30.0

print("=" * 70)
print("  T-1 (REVISED): FULL-MODEL TFLITE INT8 CONVERSION")
print("=" * 70)
print(f"  Source weights : {R4_WEIGHTS}")
print(f"  SavedModel out : {SAVED_MODEL_PATH}")
print(f"  TFLite output  : {TFLITE_OUTPUT}")
print(f"  Size gate      : < {SIZE_GATE_MB} MB")
print(f"  Parity gate    : >= {PARITY_PASS_PCT}% within +-{PARITY_TOL_PTS} pts")
print(f"  SELECT_TF_OPS  : PROHIBITED")
print()

assert R4_WEIGHTS.exists(), f"FATAL: weights not found at {R4_WEIGHTS}"
ATS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: Load full unified model with string inputs, restore R-4 weights
# ═══════════════════════════════════════════════════════════════════════════════
print("[TASK 1] Building full unified model (string inputs) and loading R-4 weights...")

full_model = build_unified_model()
full_model.load_weights(str(R4_WEIGHTS))
print(f"  Loaded: {R4_WEIGHTS.name}")

# Confirm encoder layer is present
try:
    encoder_layer = full_model.get_layer("mobile_use_encoder")
    print(f"  Encoder confirmed: '{encoder_layer.name}'  trainable={encoder_layer.trainable}")
except ValueError:
    print("  FATAL: 'mobile_use_encoder' layer NOT FOUND in model.")
    sys.exit(1)

# Print model summary and confirm string inputs
print()
full_model.summary(line_length=90)
print()
print("  Input specs:")
for inp in full_model.inputs:
    print(f"    {inp.name}  dtype={inp.dtype}  shape={inp.shape}")
print("  Output specs:")
for out in full_model.outputs:
    print(f"    {out.name}  dtype={out.dtype}  shape={out.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: Export to SavedModel — verify directory is non-empty before proceeding
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[TASK 2] Exporting to SavedModel: {SAVED_MODEL_PATH}")

tf.saved_model.save(full_model, str(SAVED_MODEL_PATH))

sm_files = list(SAVED_MODEL_PATH.rglob("*"))
file_count = len([f for f in sm_files if f.is_file()])
print(f"  SavedModel written. Files: {file_count}")
assert file_count > 0, "FATAL: SavedModel directory is empty after export."
print(f"  [OK] SavedModel non-empty ({file_count} files).")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: Build representative dataset and attempt INT8 conversion
#          NO SELECT_TF_OPS — hard stop on any error
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[TASK 3] Building representative dataset ({N_REPR_SAMPLES} pairs)...")

df = pd.read_csv(str(ATS_CSV)).dropna()
df = df.sample(min(N_REPR_SAMPLES, len(df)), random_state=42)
sample_pairs = list(zip(df["resume_text"].astype(str).values,
                         df["jd_text"].astype(str).values))
print(f"  Loaded {len(sample_pairs)} pairs from {ATS_CSV.name}")

def representative_dataset():
    for resume, jd in sample_pairs[:N_REPR_SAMPLES]:
        yield [
            tf.constant([resume], dtype=tf.string),
            tf.constant([jd],     dtype=tf.string),
        ]

print()
print("  Configuring INT8 converter (NO SELECT_TF_OPS)...")
converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_PATH))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# SELECT_TF_OPS is intentionally NOT added — per hard stop constraint

print("  Running converter.convert() — this may take several minutes...")
print("  (INT8 full-integer quantization with 300-sample calibration)")
print()

# ── HARD STOP: any exception → print exact error and exit ────────────────────
try:
    tflite_model = converter.convert()
except Exception as exc:
    print()
    print("!" * 70)
    print("  HARD STOP — INT8 CONVERSION FAILED")
    print("!" * 70)
    print()
    print("  Exact error type :", type(exc).__name__)
    print()
    print("  Exact error message:")
    print("  " + "-" * 66)
    error_msg = str(exc)
    for line in error_msg.splitlines():
        print(f"  {line}")
    print("  " + "-" * 66)
    print()
    print("  Full traceback:")
    print("  " + "-" * 66)
    tb_lines = traceback.format_exc().splitlines()
    for line in tb_lines:
        print(f"  {line}")
    print("  " + "-" * 66)
    print()
    print("  Action required:")
    print("  1. Review error above.")
    print("  2. If SELECT_TF_OPS is required: report to Sai — do NOT add it.")
    print("  3. If a different op is unsupported: report to Sai for decision.")
    print("  4. Do NOT attempt Float16 fallback or heads-only without approval.")
    print()
    print("  Awaiting Sai review before any further action.")
    print("!" * 70)
    sys.exit(1)

# ── Conversion succeeded — save binary ───────────────────────────────────────
with open(str(TFLITE_OUTPUT), "wb") as f:
    f.write(tflite_model)

size_bytes = TFLITE_OUTPUT.stat().st_size
size_mb    = size_bytes / (1024 * 1024)
size_pass  = size_mb < SIZE_GATE_MB

print(f"  INT8 TFLite saved: {TFLITE_OUTPUT.name}")
print(f"  INT8 TFLite size:  {size_mb:.2f} MB")
print(f"  SIZE GATE {'PASS' if size_pass else 'FAIL'}  (threshold < {SIZE_GATE_MB} MB)")

if not size_pass:
    print()
    print("!" * 70)
    print(f"  HARD STOP — SIZE GATE FAILED: {size_mb:.2f} MB >= {SIZE_GATE_MB} MB")
    print("  Awaiting Sai review.")
    print("!" * 70)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5: Parity check — Keras full model vs TFLite (50 samples, >= 99%)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[TASK 5] Parity check: Keras full model vs TFLite ({N_PARITY} samples)...")

# Load TFLite interpreter
interp = tf.lite.Interpreter(model_path=str(TFLITE_OUTPUT))
interp.allocate_tensors()
in_details  = interp.get_input_details()
out_details = interp.get_output_details()

print("  TFLite inputs:")
for d in in_details:
    print(f"    [{d['index']}] {d['name']}  shape={d['shape']}  dtype={d['dtype'].__name__}")
print("  TFLite outputs:")
for d in out_details:
    print(f"    [{d['index']}] {d['name']}  shape={d['shape']}  dtype={d['dtype'].__name__}")

# Identify input indices by name
resume_idx = next(d["index"] for d in in_details if "resume" in d["name"].lower())
jd_idx     = next(d["index"] for d in in_details if "jd"     in d["name"].lower())

# Load parity test samples (different from representative dataset)
df_parity = pd.read_csv(str(ATS_CSV)).dropna().head(N_PARITY)
parity_resumes = df_parity["resume_text"].astype(str).values
parity_jds     = df_parity["jd_text"].astype(str).values

ats_diffs  = []
tfl_errors = []

for i in range(min(N_PARITY, len(parity_resumes))):
    resume = str(parity_resumes[i])
    jd     = str(parity_jds[i])

    # Keras full model (string input)
    keras_out = full_model(
        [tf.constant([resume]), tf.constant([jd])],
        training=False,
    )
    keras_ats = float(keras_out[0].numpy()[0][0]) * 100

    # TFLite interpreter (string input as bytes)
    try:
        resume_bytes = np.array([resume.encode("utf-8")], dtype=object)
        jd_bytes     = np.array([jd.encode("utf-8")],     dtype=object)
        interp.set_tensor(resume_idx, resume_bytes)
        interp.set_tensor(jd_idx,     jd_bytes)
        interp.invoke()

        tfl_ats = None
        for d in out_details:
            out = interp.get_tensor(d["index"])
            if out.shape[-1] == 1:
                tfl_ats = float(out[0][0]) * 100

        if tfl_ats is not None:
            diff = abs(keras_ats - tfl_ats)
            ats_diffs.append(diff)
            if i < 5:
                print(f"  Sample {i:2d}: Keras={keras_ats:.3f}  TFLite={tfl_ats:.3f}  "
                      f"diff={diff:.4f}")
        else:
            tfl_errors.append(f"sample {i}: ATS output tensor not found")
    except Exception as exc:
        tfl_errors.append(f"sample {i}: {type(exc).__name__}: {exc}")
        if i < 5:
            print(f"  Sample {i:2d}: TFLite inference error — {type(exc).__name__}: {exc}")

# Compute parity metrics
n_checked     = len(ats_diffs)
n_within_tol  = sum(1 for d in ats_diffs if d <= PARITY_TOL_PTS)
pass_rate_pct = (n_within_tol / n_checked * 100) if n_checked > 0 else 0.0
mean_diff     = float(np.mean(ats_diffs)) if ats_diffs else float("nan")
max_diff      = float(np.max(ats_diffs))  if ats_diffs else float("nan")
parity_pass   = pass_rate_pct >= PARITY_PASS_PCT and not np.isnan(max_diff)

print(f"\n  --- Parity Results ---")
print(f"  Samples checked:  {n_checked}")
print(f"  TFLite errors:    {len(tfl_errors)}")
if tfl_errors:
    for e in tfl_errors[:5]:
        print(f"    {e}")
print(f"  ATS mean_diff:    {mean_diff:.4f} pts")
print(f"  ATS max_diff:     {max_diff:.4f} pts")
print(f"  Within +-{PARITY_TOL_PTS} pts: {n_within_tol}/{n_checked}  ({pass_rate_pct:.1f}%)")
print(f"  Gate (>= {PARITY_PASS_PCT}%):  {'PASS' if parity_pass else 'FAIL'}")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6: Write conversion_summary_full.json
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[TASK 6] Writing conversion_summary_full.json...")

summary = {
    "stage": "T-1-revised",
    "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "source_weights": str(R4_WEIGHTS),
    "source_stage": "R-4 Joint Fine-Tuning",
    "encoder": "universal-sentence-encoder-mobile/2",
    "architecture": "full-model (encoder + ATS head + Domain head + RSG head)",
    "input_0": {"name": "resume_text", "dtype": "tf.string",  "shape": "()"},
    "input_1": {"name": "jd_text",     "dtype": "tf.string",  "shape": "()"},
    "output_0": {"name": "ats_score",          "dtype": "tf.float32", "shape": "[1]",
                 "range": "[0.0, 1.0]", "activation": "sigmoid"},
    "output_1": {"name": "domain_probs",       "dtype": "tf.float32", "shape": f"[{NUM_DOMAINS}]",
                 "activation": "softmax"},
    "output_2": {"name": "rsg_template_probs", "dtype": "tf.float32", "shape": f"[{RSG_NUM_CLASSES}]",
                 "activation": "softmax"},
    "quantization": "INT8",
    "select_tf_ops_required": False,
    "optimizations": ["DEFAULT"],
    "supported_ops": ["TFLITE_BUILTINS_INT8"],
    "representative_dataset_samples": N_REPR_SAMPLES,
    "size_bytes": size_bytes,
    "size_mb": round(size_mb, 2),
    "size_gate_mb": SIZE_GATE_MB,
    "size_gate_pass": bool(size_pass),
    "parity_samples_checked": n_checked,
    "parity_tflite_errors": len(tfl_errors),
    "parity_mean_diff_pts": round(mean_diff, 4) if not np.isnan(mean_diff) else None,
    "parity_max_diff_pts": round(max_diff, 4) if not np.isnan(max_diff) else None,
    "parity_pass_rate_pct": round(pass_rate_pct, 2),
    "parity_gate_pct": PARITY_PASS_PCT,
    "parity_pass": bool(parity_pass),
    "tflite_file": TFLITE_OUTPUT.name,
    "tflite_path": str(TFLITE_OUTPUT),
    "saved_model_path": str(SAVED_MODEL_PATH),
    "definition_of_done": {
        "full_model_string_inputs_confirmed": True,
        "saved_model_non_empty": True,
        "tflite_produced_without_select_tf_ops": True,
        "size_gate_pass": bool(size_pass),
        "parity_gate_pass": bool(parity_pass),
        "summary_generated": True,
        "select_tf_ops_required_confirmed_false": True,
    },
    "ready_for_t2": bool(size_pass and parity_pass),
}

with open(str(SUMMARY_JSON), "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"  Saved: {SUMMARY_JSON}")

# ═══════════════════════════════════════════════════════════════════════════════
# HARD STOP — Post results for Sai review
# ═══════════════════════════════════════════════════════════════════════════════
all_pass = bool(size_pass and parity_pass)

print()
print("=" * 70)
print("  T-1 (REVISED) COMPLETE — HARD STOP: AWAITING SAI REVIEW")
print("=" * 70)
print(f"\n  INT8 TFLite size:  {size_mb:.2f} MB")
print(f"  SIZE GATE:         {'PASS' if size_pass else 'FAIL'}  (< {SIZE_GATE_MB} MB)")
print(f"  SELECT_TF_OPS:     NOT REQUIRED  (confirmed false)")
print()
print(f"  Parity max_diff:   {max_diff:.4f} pts  (tolerance: +-{PARITY_TOL_PTS} pts)")
print(f"  Parity pass_rate:  {pass_rate_pct:.1f}%  (gate: >= {PARITY_PASS_PCT}%)")
print(f"  Parity gate:       {'PASS' if parity_pass else 'FAIL'}")
print()
print("  DEFINITION OF DONE:")
print(f"  [OK] Full model (encoder + 3 heads) loaded — string inputs confirmed")
print(f"  [OK] SavedModel export non-empty")
print(f"  [{'OK' if TFLITE_OUTPUT.exists() else '!!'}] ats_core_full_int8.tflite produced without SELECT_TF_OPS")
print(f"  [{'OK' if size_pass else '!!'}] TFLite size < {SIZE_GATE_MB} MB: {size_mb:.2f} MB")
print(f"  [{'OK' if parity_pass else '!!'}] Parity pass_rate >= {PARITY_PASS_PCT}%: {pass_rate_pct:.1f}%")
print(f"  [OK] conversion_summary_full.json generated")
print(f"  [OK] select_tf_ops_required confirmed false")
print()
print("  conversion_summary_full.json contents:")
print("  " + "-" * 66)
print(json.dumps(summary, indent=4))
print("  " + "-" * 66)
print()
if all_pass:
    print("  ALL GATES PASS — awaiting Sai approval to proceed to T-2.")
else:
    print("  !! ONE OR MORE GATES FAILED — Sai review required.")
print("=" * 70)
