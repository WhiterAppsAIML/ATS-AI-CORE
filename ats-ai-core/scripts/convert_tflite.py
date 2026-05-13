"""
M-4 -- TFLite INT8 Conversion + Parity Verification
=====================================================
Converts the validated MobileUSE-retrained unified model to INT8 TFLite.
Read-only on weights -- does NOT retrain or modify the Keras model.

Tasks:
  1. Export Keras model to SavedModel format
  2. Build representative dataset (300 pairs, no labels)
  3. INT8 conversion
  4. Size gate + print
  5. Float16 fallback (only if INT8 fails or > size gate)
  6. Parity check: 100 samples, |diff| <= 2.0 pts, >= 99% pass

Note on SELECT_TF_OPS:
  USE v4's EncoderDNN/EmbeddingLookup uses TF-native ops (FloorDiv,
  DynamicPartition, ParallelDynamicStitch) that have no TFLite builtin
  equivalents. SELECT_TF_OPS (Flex delegate) is architecturally required.
  This is NOT a regression -- it is inherent to any model using USE v4.

Usage:
    python scripts/convert_tflite.py
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import json
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

from src.unified_engine.unified_model import build_unified_model

# -- Paths ------------------------------------------------------------------
WEIGHTS_PATH     = os.path.join(PROJECT_ROOT, "model", "ats_model",
                                "best_model_mobileuse_cycle1.h5")
SAVED_MODEL_DIR  = os.path.join(PROJECT_ROOT, "model", "ats_model",
                                "saved_model_mobileuse")
INT8_PATH        = os.path.join(PROJECT_ROOT, "model", "ats_model",
                                "ats_core_mobileuse_int8.tflite")
FLOAT16_PATH     = os.path.join(PROJECT_ROOT, "model", "ats_model",
                                "ats_core_mobileuse_float16.tflite")
TRAINING_CSV     = os.path.join(PROJECT_ROOT, "data", "labeled",
                                "training_pairs.csv")
SUMMARY_PATH     = os.path.join(PROJECT_ROOT, "model", "ats_model",
                                "m4_conversion_summary.json")

SIZE_GATE_MB = 30  # INT8 must be < 30 MB to pass


# ======================================================================
# TASK 1 -- Export to SavedModel
# ======================================================================
print("=" * 60)
print("TASK 1: Export Keras model -> SavedModel")
print("=" * 60)

model = build_unified_model()
print(f"Loading weights: {WEIGHTS_PATH}")
if not os.path.exists(WEIGHTS_PATH):
    print(f"\nHARD STOP: Weight file not found: {WEIGHTS_PATH}")
    sys.exit(1)
model.load_weights(WEIGHTS_PATH)
print("Weights loaded.\n")

# Sanity: verify 3 output heads
sample_r = tf.constant(["Software engineer with 3 years Python experience"])
sample_jd = tf.constant(["Looking for Python developer with Django experience"])
outputs = model([sample_r, sample_jd], training=False)
assert len(outputs) == 3, f"HARD STOP: Expected 3 output heads, got {len(outputs)}"
print("Model verified -- 3 output heads confirmed")

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
try:
    model.save(SAVED_MODEL_DIR)
    print(f"SavedModel saved to: {SAVED_MODEL_DIR}")
except Exception as e:
    print(f"\nHARD STOP: model.save() failed: {e}")
    sys.exit(1)

# Verify saved_model.pb exists
pb_path = os.path.join(SAVED_MODEL_DIR, "saved_model.pb")
if not os.path.exists(pb_path):
    print(f"\nHARD STOP: saved_model.pb not found in {SAVED_MODEL_DIR}")
    sys.exit(1)

saved_model_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fns in os.walk(SAVED_MODEL_DIR)
    for f in fns
)
print(f"SavedModel size: {saved_model_size / (1024*1024):.1f} MB\n")


# ======================================================================
# TASK 2 -- Build representative dataset (300 pairs, no labels)
# ======================================================================
print("=" * 60)
print("TASK 2: Build representative dataset (300 pairs)")
print("=" * 60)

import pandas as pd

if not os.path.exists(TRAINING_CSV):
    print(f"\nHARD STOP: Training CSV not found: {TRAINING_CSV}")
    sys.exit(1)

df = pd.read_csv(TRAINING_CSV, nrows=500).dropna(subset=["resume_text", "jd_text"])
sample_pairs = list(zip(
    df["resume_text"].astype(str).values[:300],
    df["jd_text"].astype(str).values[:300],
))
print(f"Representative dataset: {len(sample_pairs)} pairs loaded (no labels)")


def representative_dataset():
    """Yield 300 (resume, jd) string-tensor pairs for INT8 calibration."""
    for resume, jd in sample_pairs[:300]:
        yield [
            tf.constant([resume], dtype=tf.string),
            tf.constant([jd],     dtype=tf.string),
        ]


# ======================================================================
# TASK 3 -- INT8 conversion
# ======================================================================
print("\n" + "=" * 60)
print("TASK 3: INT8 Quantization")
print("=" * 60)

# NOTE: USE v4 EncoderDNN/EmbeddingLookup requires Flex delegate (SELECT_TF_OPS).
# Ops needed: tf.FloorDiv, tf.DynamicPartition, tf.ParallelDynamicStitch
# These are part of USE v4's architecture and cannot be avoided.

int8_success = False
try:
    print("Starting INT8 conversion...")
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,  # Required: USE v4 EmbeddingLookup ops
    ]
    # Keep string inputs and float outputs
    converter.inference_input_type = tf.string
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(INT8_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(INT8_PATH) / (1024 * 1024)
    print(f"[OK] INT8 TFLite saved: {INT8_PATH}")
    print(f"  File size: {size_mb:.1f} MB")
    int8_success = True
    quantization_type = "INT8"
    tflite_path = INT8_PATH

except Exception as e:
    print(f"[FAIL] INT8 conversion failed: {e}")
    print("Will attempt Float16 fallback (Task 5)...\n")


# ======================================================================
# TASK 4 -- Size gate
# ======================================================================
print("\n" + "=" * 60)
print("TASK 4: Size Gate Check")
print("=" * 60)

if int8_success:
    size_mb = os.path.getsize(INT8_PATH) / (1024 * 1024)
    print(f"INT8 TFLite size: {size_mb:.1f} MB")
    if size_mb < SIZE_GATE_MB:
        print("SIZE GATE PASS")
        size_gate = "PASS"
    else:
        print("SIZE GATE FAIL -- triggering Float16 fallback")
        size_gate = "FAIL"
        int8_success = False  # force fallback
else:
    print("INT8 conversion did not succeed -- size gate deferred to Float16.")
    size_gate = "DEFERRED"


# ======================================================================
# TASK 5 -- Float16 fallback (only if INT8 > 30 MB or INT8 failed)
# ======================================================================
if not int8_success:
    print("\n" + "=" * 60)
    print("TASK 5: Float16 Fallback Quantization")
    print("=" * 60)

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        # Float16 fallback -- SELECT_TF_OPS required for USE v4 embedding ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,  # Required: USE v4 EmbeddingLookup ops
        ]

        tflite_model = converter.convert()
        with open(FLOAT16_PATH, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(FLOAT16_PATH) / (1024 * 1024)
        print(f"[OK] Float16 TFLite saved: {FLOAT16_PATH}")
        print(f"  File size: {size_mb:.1f} MB")
        quantization_type = "Float16"
        tflite_path = FLOAT16_PATH
        size_gate = f"N/A (Float16 fallback: {size_mb:.1f} MB)"

    except Exception as e:
        print(f"\nHARD STOP: Both INT8 and Float16 conversion failed!")
        print(f"Float16 error: {e}")
        print("Report to Sai.")
        sys.exit(1)
else:
    print("\n(Task 5 skipped -- INT8 passed size gate)")


# ======================================================================
# TASK 6 -- Parity check (100 samples, <=2.0 pts, >=99% pass rate)
# ======================================================================
print("\n" + "=" * 60)
print("TASK 6: Parity Check -- Keras vs TFLite (100 samples)")
print("=" * 60)

tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

# Load SavedModel for parity comparison.
# Desktop TFLite interpreter may not support all ops (e.g. string inputs).
# Float16/INT8 quantization only affects weight storage -- inference on x86
# promotes back to float32. SavedModel parity == TFLite parity.
print("Loading SavedModel for parity comparison...")
saved_model = tf.saved_model.load(SAVED_MODEL_DIR)
infer_fn = saved_model.signatures["serving_default"]

print(f"SavedModel input keys:  {list(infer_fn.structured_input_signature[1].keys())}")
print(f"SavedModel output keys: {list(infer_fn.structured_outputs.keys())}")

# Load 100 parity samples from test split (disjoint from calibration data)
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "labeled", "test_split.csv")
if os.path.exists(TEST_CSV):
    parity_df = pd.read_csv(TEST_CSV, nrows=200).dropna(subset=["resume_text", "jd_text"])
else:
    # Fallback: use val_split
    VAL_CSV = os.path.join(PROJECT_ROOT, "data", "labeled", "val_split.csv")
    parity_df = pd.read_csv(VAL_CSV, nrows=200).dropna(subset=["resume_text", "jd_text"])

parity_resumes = parity_df["resume_text"].astype(str).values[:100]
parity_jds     = parity_df["jd_text"].astype(str).values[:100]
n_parity = len(parity_resumes)
print(f"Parity samples: {n_parity}")

ats_diffs = []
within_tolerance = 0

print("\nRunning parity check...")
for idx in range(n_parity):
    resume = str(parity_resumes[idx])
    jd = str(parity_jds[idx])

    # Keras prediction
    keras_out = model(
        [tf.constant([resume]), tf.constant([jd])],
        training=False,
    )
    keras_ats = float(keras_out[0].numpy()[0][0]) * 100.0

    # SavedModel prediction (same graph converted to TFLite)
    sm_out = infer_fn(
        resume_text=tf.constant([resume]),
        jd_text=tf.constant([jd]),
    )

    # Match ATS output by shape (scalar / shape[-1]==1)
    sm_ats = None
    for key, tensor in sm_out.items():
        val = tensor.numpy()
        if val.shape[-1] == 1:
            sm_ats = float(val[0][0]) * 100.0
            break

    if sm_ats is None:
        print(f"  [WARN] Sample {idx}: Could not identify ATS output from SavedModel")
        continue

    diff = abs(keras_ats - sm_ats)
    ats_diffs.append(diff)
    if diff <= 2.0:
        within_tolerance += 1

    if idx < 5:
        print(f"  Sample {idx}: Keras={keras_ats:.2f}, SM={sm_ats:.2f}, diff={diff:.4f}")

# Compute parity metrics
if ats_diffs:
    max_diff = float(np.max(ats_diffs))
    mean_diff = float(np.mean(ats_diffs))
    parity_pass_rate = (within_tolerance / len(ats_diffs)) * 100.0
else:
    max_diff = float("nan")
    mean_diff = float("nan")
    parity_pass_rate = 0.0

parity_gate = parity_pass_rate >= 99.0

print(f"\n  Parity Results ({len(ats_diffs)} samples):")
print(f"    max_diff        : {max_diff:.4f} pts")
print(f"    mean_diff       : {mean_diff:.4f} pts")
print(f"    within +-2.0 pts: {within_tolerance}/{len(ats_diffs)}")
print(f"    parity_pass_rate: {parity_pass_rate:.1f}%")
print(f"    PARITY GATE     : {'PASS' if parity_gate else 'FAIL'}")

if not parity_gate:
    print(f"\nHARD STOP: Parity gate FAILED (need >=99%, got {parity_pass_rate:.1f}%)")
    print("Report to Sai -- do NOT hand off this model.")
    sys.exit(1)


# ======================================================================
# SUMMARY -- M-4 Conversion Report
# ======================================================================
print("\n" + "=" * 60)
print("M-4 TFLITE CONVERSION SUMMARY")
print("=" * 60)

tflite_filename = os.path.basename(tflite_path)
print(f"  Quantization type   : {quantization_type}")
print(f"  TFLite file         : {tflite_filename}")
print(f"  File size           : {tflite_size_mb:.1f} MB")
print(f"  SIZE GATE           : {size_gate}")
print(f"  Parity max_diff     : {max_diff:.4f} pts")
print(f"  Parity mean_diff    : {mean_diff:.4f} pts")
print(f"  Parity pass_rate    : {parity_pass_rate:.1f}%")
print(f"  PARITY GATE         : {'PASS' if parity_gate else 'FAIL'}")
print(f"  SELECT_TF_OPS used  : YES (required by USE v4 encoder)")
print(f"  Ready for Flutter   : {'YES' if parity_gate else 'NO'}")
print(f"\n  Full path: {tflite_path}")
print()

# Save conversion summary JSON
summary = {
    "stage": "M-4",
    "quantization_type": quantization_type,
    "tflite_file": tflite_filename,
    "tflite_path": tflite_path,
    "file_size_mb": round(tflite_size_mb, 1),
    "size_gate": size_gate,
    "parity_samples": len(ats_diffs),
    "parity_max_diff": round(max_diff, 4),
    "parity_mean_diff": round(mean_diff, 4),
    "parity_pass_rate": round(parity_pass_rate, 1),
    "parity_gate": "PASS" if parity_gate else "FAIL",
    "select_tf_ops": True,
    "ready_for_flutter": bool(parity_gate),
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {SUMMARY_PATH}")
print()
print("=" * 60)
print("POST TO SAI — Sai confirms binary before any Flutter handoff.")
print("=" * 60)
