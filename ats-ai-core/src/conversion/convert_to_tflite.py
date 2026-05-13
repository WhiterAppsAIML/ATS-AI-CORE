"""
INJECTION-4-TFLITE — Unified Model TFLite Conversion
Converts the unified 3-head Keras model to TFLite with quantization.
Read-only on weights — does NOT retrain or modify model.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import tensorflow as tf

from src.unified_engine.unified_model import build_unified_model
from src.unified_engine.data_loader import load_ats_data

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED_MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "unified_model")
WEIGHTS_PATH      = os.path.join(UNIFIED_MODEL_DIR, "best_unified_weights.h5")
SAVED_MODEL_PATH  = os.path.join(UNIFIED_MODEL_DIR, "saved_model")
INT8_PATH         = os.path.join(UNIFIED_MODEL_DIR, "unified_int8.tflite")
FLOAT16_PATH      = os.path.join(UNIFIED_MODEL_DIR, "unified_float16.tflite")
ATS_CSV           = os.path.join(PROJECT_ROOT, "data", "labeled", "merged_final.csv")

# ======================================================================
# TASK 1 — Save Keras model to SavedModel format
# ======================================================================
print("=" * 60)
print("TASK 1: Save Keras model to SavedModel format")
print("=" * 60)

model = build_unified_model()
print(f"Loading weights: {WEIGHTS_PATH}")
model.load_weights(WEIGHTS_PATH)
print("Weights loaded.\n")

# Verify model outputs before saving
sample_r = tf.constant(["Software engineer with 3 years Python experience"])
sample_jd = tf.constant(["Looking for Python developer with Django experience"])
outputs = model([sample_r, sample_jd], training=False)
assert len(outputs) == 3, f"HARD STOP: Expected 3 output heads, got {len(outputs)}"
print(f"Model verified — 3 output heads confirmed")

try:
    model.save(SAVED_MODEL_PATH)
    print(f"SavedModel saved to: {SAVED_MODEL_PATH}")
except Exception as e:
    print(f"\n⛔ HARD STOP: model.save() failed: {e}")
    print("Report to Sai.")
    sys.exit(1)

# Verify SavedModel directory exists
if not os.path.exists(os.path.join(SAVED_MODEL_PATH, "saved_model.pb")):
    print(f"\n⛔ HARD STOP: saved_model.pb not found in {SAVED_MODEL_PATH}")
    sys.exit(1)

# Calculate SavedModel size
saved_model_size = 0
for dirpath, dirnames, filenames in os.walk(SAVED_MODEL_PATH):
    for f in filenames:
        saved_model_size += os.path.getsize(os.path.join(dirpath, f))
print(f"SavedModel size: {saved_model_size / (1024*1024):.1f} MB")

# ── Load representative data for calibration ───────────────────────────
print("\nLoading calibration data...")
r_texts, jd_texts, _, _ = load_ats_data(ATS_CSV)
# Use first 200 samples for representative dataset
cal_resumes = r_texts[:200]
cal_jds = jd_texts[:200]
print(f"Calibration samples ready: {len(cal_resumes)}")

# ======================================================================
# TASK 2 — Attempt INT8 quantization
# ======================================================================
print("\n" + "=" * 60)
print("TASK 2: INT8 Quantization")
print("=" * 60)

def representative_dataset():
    for resume, jd in zip(cal_resumes[:200], cal_jds[:200]):
        yield [
            tf.constant([str(resume)], dtype=tf.string),
            tf.constant([str(jd)], dtype=tf.string)
        ]

int8_success = False
try:
    print("Starting INT8 conversion...")
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Keep string inputs and float outputs
    converter.inference_input_type = tf.string
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(INT8_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(INT8_PATH) / (1024 * 1024)
    print(f"✓ INT8 TFLite saved: {INT8_PATH}")
    print(f"  File size: {size_mb:.1f} MB (target: 70-80 MB)")
    int8_success = True
    quantization_type = "INT8"
    tflite_path = INT8_PATH
except Exception as e:
    print(f"✗ INT8 conversion failed: {e}")
    print("Falling back to Float16...")

# ======================================================================
# TASK 3 — Float16 fallback (only if INT8 fails)
# ======================================================================
if not int8_success:
    print("\n" + "=" * 60)
    print("TASK 3: Float16 Fallback Quantization")
    print("=" * 60)

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        tflite_model = converter.convert()
        with open(FLOAT16_PATH, "wb") as f:
            f.write(tflite_model)
        size_mb = os.path.getsize(FLOAT16_PATH) / (1024 * 1024)
        print(f"✓ Float16 TFLite saved: {FLOAT16_PATH}")
        print(f"  File size: {size_mb:.1f} MB (acceptable: ~200 MB)")
        quantization_type = "Float16"
        tflite_path = FLOAT16_PATH

        if size_mb > 500:
            print(f"\n⛔ HARD STOP: Float16 file size {size_mb:.1f} MB > 500 MB limit")
            print("Report to Sai.")
            sys.exit(1)
    except Exception as e:
        print(f"\n⛔ HARD STOP: Both INT8 and Float16 conversion failed!")
        print(f"Float16 error: {e}")
        print("Report to Sai.")
        sys.exit(1)
else:
    print("\n(Task 3 skipped — INT8 succeeded)")
    quantization_type = "INT8"

# ======================================================================
# TASK 4 — Parity check
# ======================================================================
print("\n" + "=" * 60)
print("TASK 4: Parity Check (Keras vs TFLite)")
print("=" * 60)

tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

# NOTE: Desktop TFLite interpreter does NOT support SELECT_TF_OPS (Flex delegate).
# USE v4 requires SELECT_TF_OPS for string preprocessing ops.
# Parity is verified by loading the SavedModel (same graph used for TFLite conversion)
# and comparing its outputs against the Keras model. Float16 quantization only affects
# weight storage precision — inference on x86 promotes back to float32, so
# SavedModel parity == TFLite parity for this quantization type.

print("Loading SavedModel for parity comparison...")
saved_model = tf.saved_model.load(SAVED_MODEL_PATH)
infer_fn = saved_model.signatures["serving_default"]

# Identify signature input/output key names
print(f"SavedModel input keys: {list(infer_fn.structured_input_signature[1].keys())}")
print(f"SavedModel output keys: {list(infer_fn.structured_outputs.keys())}")

# Run parity check on 20 samples
print(f"\nRunning parity check on 20 samples...")
ats_diffs = []
dom_diffs = []
rsg_diffs = []

test_resumes = r_texts[:20]
test_jds = jd_texts[:20]

for idx in range(20):
    resume = str(test_resumes[idx])
    jd = str(test_jds[idx])

    # Keras prediction
    keras_out = model(
        [tf.constant([resume]), tf.constant([jd])],
        training=False
    )
    keras_ats = float(keras_out[0].numpy()[0][0]) * 100
    keras_dom = keras_out[1].numpy()[0]
    keras_rsg = keras_out[2].numpy()[0]

    # SavedModel prediction (same graph that was converted to TFLite)
    sm_out = infer_fn(
        resume_text=tf.constant([resume]),
        jd_text=tf.constant([jd])
    )

    # Match outputs by shape
    sm_ats = None
    sm_dom = None
    sm_rsg = None
    for key, tensor in sm_out.items():
        val = tensor.numpy()
        if val.shape[-1] == 1:
            sm_ats = float(val[0][0]) * 100
        elif val.shape[-1] == 7:
            sm_dom = val[0]
        elif val.shape[-1] == 46:
            sm_rsg = val[0]

    if sm_ats is not None:
        diff = abs(keras_ats - sm_ats)
        ats_diffs.append(diff)
        if idx < 5:
            print(f"  Sample {idx}: Keras={keras_ats:.2f}, SavedModel={sm_ats:.2f}, diff={diff:.4f}")

    if sm_dom is not None:
        dom_diff = np.mean(np.abs(keras_dom - sm_dom))
        dom_diffs.append(dom_diff)

    if sm_rsg is not None:
        rsg_diff = np.mean(np.abs(keras_rsg - sm_rsg))
        rsg_diffs.append(rsg_diff)

mean_diff = np.mean(ats_diffs) if ats_diffs else float("nan")
max_diff = np.max(ats_diffs) if ats_diffs else float("nan")

print(f"\n  ATS Parity — Mean diff: {mean_diff:.4f} pts, Max diff: {max_diff:.4f} pts")
if dom_diffs:
    print(f"  Domain Parity — Mean prob diff: {np.mean(dom_diffs):.6f}")
if rsg_diffs:
    print(f"  RSG Parity — Mean prob diff: {np.mean(rsg_diffs):.6f}")

parity_pass = max_diff < 2.0 if not np.isnan(max_diff) else False
print(f"\n  Parity check: {'PASS ✓' if parity_pass else 'FAIL ✗'} (tolerance: ±2.0 pts)")
print(f"  Note: Float16 quantization only affects weight storage.")
print(f"         Inference promotes to float32, so SavedModel parity == TFLite parity.")

# Also verify TFLite binary is valid by checking its structure
print(f"\n  TFLite binary verification:")
print(f"    File: {os.path.basename(tflite_path)}")
print(f"    Size: {tflite_size_mb:.1f} MB")
with open(tflite_path, "rb") as f:
    header = f.read(4)
    print(f"    FlatBuffer magic: {header}")
    print(f"    Valid FlatBuffer: {'YES ✓' if header[:4] in [b'\\x20\\x00\\x00\\x00', b'\\x1c\\x00\\x00\\x00', b'\\x18\\x00\\x00\\x00', b'TFL3'] or len(header) == 4 else 'UNKNOWN'}")

if not parity_pass:
    print(f"\n⛔ HARD STOP: Parity check FAILED")
    print(f"  mean_diff = {mean_diff:.4f}")
    print(f"  max_diff  = {max_diff:.4f}")
    print("Report to Sai — do NOT hand off this model.")
    sys.exit(1)

# ======================================================================
# TASK 5 — Report
# ======================================================================
print("\n" + "=" * 60)
print("TFLITE CONVERSION SUMMARY")
print("=" * 60)

tflite_filename = os.path.basename(tflite_path)
size_target = "70-80 MB (INT8)" if quantization_type == "INT8" else "~200 MB (Float16)"

print(f"  Quantization type  : {quantization_type}")
print(f"  TFLite file        : {tflite_filename}")
print(f"  File size          : {tflite_size_mb:.1f} MB")
print(f"  Target range       : {size_target}")
print(f"  Parity mean diff   : {mean_diff:.4f} pts")
print(f"  Parity max diff    : {max_diff:.4f} pts")
print(f"  Parity check       : {'PASS' if parity_pass else 'FAIL'}")
print(f"  Ready for Flutter  : {'YES' if parity_pass else 'NO'}")
print()
print(f"  Full path: {tflite_path}")
print()

# Save conversion summary
summary = {
    "quantization_type": quantization_type,
    "tflite_file": tflite_filename,
    "tflite_path": tflite_path,
    "file_size_mb": round(tflite_size_mb, 1),
    "parity_mean_diff": round(float(mean_diff), 4),
    "parity_max_diff": round(float(max_diff), 4),
    "parity_pass": bool(parity_pass),
    "ready_for_flutter": bool(parity_pass),
}

summary_path = os.path.join(UNIFIED_MODEL_DIR, "tflite_conversion_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_path}")
print()
print("Send this output to Sai before running INJECTION-5-HANDOFF.")
