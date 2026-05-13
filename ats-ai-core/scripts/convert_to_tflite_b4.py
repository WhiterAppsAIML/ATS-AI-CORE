"""
B-4: TFLite Float16 Conversion & Parity Check (Heads-Only)
=============================================================
Prior state: B-3c complete. RSG=65.5%, Domain=75.5%, ATS MAE=7.50.

Strategy:
  The USELiteEncoder uses tf.py_function (EagerPyFunc) which is NOT
  compatible with TFLite conversion. Instead, we export a "heads-only"
  model that takes pre-computed 512-dim embeddings as input.
  
  At inference time, the Flutter/Python client will:
  1. Run SentencePiece + USE Lite v2 encoding separately (or via TFHub)
  2. Feed the 512-dim embeddings into this TFLite model
  3. Get ATS score, Domain probs, and RSG template probs as output

Pipeline:
  1. Build heads-only model (input: embeddings, output: 3 heads)
  2. Transfer B-3c trained weights to heads-only model
  3. Float16 quantization (INT8 not needed for such small model)
  4. Parity check: Keras(full) vs TFLite(heads-only)
  5. Size gate + report
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.unified_engine.unified_model import build_unified_model, USE_OUTPUT_DIM, RSG_NUM_CLASSES

# -- Paths ----------------------------------------------------------------
PROJECT_ROOT      = Path(os.path.dirname(os.path.abspath(__file__))).parent
UNIFIED_MODEL_DIR = PROJECT_ROOT / "model" / "unified_model"
ATS_MODEL_DIR     = PROJECT_ROOT / "model" / "ats_model"

B3C_WEIGHTS       = ATS_MODEL_DIR / "unified_model_B3c_augmented.h5"
SAVED_MODEL_PATH  = UNIFIED_MODEL_DIR / "saved_model_b4_heads"
FLOAT16_PATH      = UNIFIED_MODEL_DIR / "unified_model_lite_v2_float16.tflite"
ATS_CSV           = PROJECT_ROOT / "data" / "labeled" / "merged_final.csv"
SUMMARY_JSON      = UNIFIED_MODEL_DIR / "b4_conversion_summary.json"

print("=" * 65)
print("  B-4: TFLite CONVERSION & PARITY CHECK (Heads-Only)")
print("=" * 65)

# =====================================================================
# TASK 1: Build full model and extract head weights
# =====================================================================
print("\n[1/5] Building full model & loading B-3c weights...")
full_model = build_unified_model()
print(f"  Loading: {B3C_WEIGHTS}")
full_model.load_weights(str(B3C_WEIGHTS))
print("  B-3c weights loaded.")

# =====================================================================
# TASK 2: Build heads-only model (embedding inputs, no encoder)
# =====================================================================
print("\n[2/5] Building heads-only model (embedding inputs)...")

# Inputs: pre-computed 512-dim embeddings
resume_emb_input = tf.keras.Input(shape=(USE_OUTPUT_DIM,), dtype=tf.float32,
                                   name="resume_embedding")
jd_emb_input     = tf.keras.Input(shape=(USE_OUTPUT_DIM,), dtype=tf.float32,
                                   name="jd_embedding")

# --- Feature engineering (same as full model) ---
cosine_sim   = tf.keras.layers.Dot(axes=1, normalize=True,
                   name="cosine_sim")([resume_emb_input, jd_emb_input])
dot_prod     = tf.keras.layers.Dot(axes=1, normalize=False,
                   name="dot_prod")([resume_emb_input, jd_emb_input])
ats_features = tf.keras.layers.Concatenate(
                   name="ats_features")([resume_emb_input, jd_emb_input,
                                         cosine_sim, dot_prod])

# --- HEAD 1: ATS Score ---
x1 = tf.keras.layers.Dense(256, activation="relu",  name="ats_dense1")(ats_features)
x1 = tf.keras.layers.Dropout(0.3,                   name="ats_drop1")(x1)
x1 = tf.keras.layers.Dense(64,  activation="relu",  name="ats_dense2")(x1)
x1 = tf.keras.layers.Dropout(0.2,                   name="ats_drop2")(x1)
ats_output = tf.keras.layers.Dense(1, activation="sigmoid",
                 name="ats_score")(x1)

# --- HEAD 2: Domain Classifier ---
x2 = tf.keras.layers.Dense(256, activation="relu",  name="dom_dense1")(jd_emb_input)
x2 = tf.keras.layers.Dropout(0.3,                   name="dom_drop1")(x2)
x2 = tf.keras.layers.Dense(128, activation="relu",  name="dom_dense2")(x2)
x2 = tf.keras.layers.Dropout(0.2,                   name="dom_drop2")(x2)
domain_output = tf.keras.layers.Dense(7, activation="softmax",
                    name="domain_probs")(x2)

# --- HEAD 3: RSG Template Classifier ---
x3 = tf.keras.layers.Dense(512, activation="relu",  name="rsg_dense1")(resume_emb_input)
x3 = tf.keras.layers.BatchNormalization(             name="rsg_bn1")(x3)
x3 = tf.keras.layers.Dropout(0.4,                   name="rsg_drop1")(x3)
x3 = tf.keras.layers.Dense(256, activation="relu",  name="rsg_dense2")(x3)
x3 = tf.keras.layers.BatchNormalization(             name="rsg_bn2")(x3)
x3 = tf.keras.layers.Dropout(0.3,                   name="rsg_drop2")(x3)
x3 = tf.keras.layers.Dense(128, activation="relu",  name="rsg_dense3")(x3)
x3 = tf.keras.layers.BatchNormalization(             name="rsg_bn3")(x3)
x3 = tf.keras.layers.Dropout(0.3,                   name="rsg_drop3")(x3)
rsg_output = tf.keras.layers.Dense(RSG_NUM_CLASSES, activation="softmax",
                 name="rsg_template")(x3)

heads_model = tf.keras.Model(
    inputs=[resume_emb_input, jd_emb_input],
    outputs=[ats_output, domain_output, rsg_output],
    name="unified_heads_only"
)

# --- Transfer weights from full model to heads-only model ---
print("  Transferring weights by layer name...")
transferred = 0
skipped = 0
for layer in heads_model.layers:
    if layer.count_params() > 0:
        try:
            full_layer = full_model.get_layer(layer.name)
            layer.set_weights(full_layer.get_weights())
            transferred += 1
        except ValueError:
            skipped += 1
            print(f"    [SKIP] {layer.name} -- not in full model")

print(f"  Transferred: {transferred} layers, Skipped: {skipped}")

# Verify heads-only model
heads_model.summary()
total_params = heads_model.count_params()
print(f"\n  Heads-only params: {total_params:,}")

# =====================================================================
# TASK 3: Parity check (full model vs heads-only model)
# =====================================================================
print("\n[3/5] Verifying heads-only parity with full model...")

# Load test data
cal_df = pd.read_csv(str(ATS_CSV)).dropna().head(50)
test_resumes = cal_df["resume_text"].astype(str).values
test_jds     = cal_df["jd_text"].astype(str).values

N_PARITY = min(50, len(test_resumes))
ats_diffs = []
dom_diffs = []
rsg_diffs = []

for idx in range(N_PARITY):
    resume = str(test_resumes[idx])
    jd = str(test_jds[idx])

    # Full model (string input -> encoder -> heads)
    full_out = full_model(
        [tf.constant([resume]), tf.constant([jd])],
        training=False
    )
    full_ats = float(full_out[0].numpy()[0][0]) * 100
    full_dom = full_out[1].numpy()[0]
    full_rsg = full_out[2].numpy()[0]

    # Get embeddings from the encoder
    encoder = full_model.get_layer("mobile_use_encoder")
    resume_emb = encoder(tf.constant([resume]))
    jd_emb     = encoder(tf.constant([jd]))

    # Heads-only model (embedding input -> heads)
    heads_out = heads_model(
        [resume_emb, jd_emb],
        training=False
    )
    heads_ats = float(heads_out[0].numpy()[0][0]) * 100
    heads_dom = heads_out[1].numpy()[0]
    heads_rsg = heads_out[2].numpy()[0]

    ats_diff = abs(full_ats - heads_ats)
    ats_diffs.append(ats_diff)

    dom_diffs.append(np.mean(np.abs(full_dom - heads_dom)))
    rsg_diffs.append(np.mean(np.abs(full_rsg - heads_rsg)))

    if idx < 5:
        print(f"  Sample {idx}: Full={full_ats:.2f}, Heads={heads_ats:.2f}, "
              f"diff={ats_diff:.6f}")

mean_diff_kh = np.mean(ats_diffs)
max_diff_kh  = np.max(ats_diffs)
print(f"\n  Keras-to-Heads Parity:")
print(f"    ATS:    mean_diff={mean_diff_kh:.6f}, max_diff={max_diff_kh:.6f}")
print(f"    Domain: mean_diff={np.mean(dom_diffs):.6f}")
print(f"    RSG:    mean_diff={np.mean(rsg_diffs):.6f}")

if max_diff_kh > 0.01:
    print(f"  [!!] WARNING: weight transfer parity issue!")
else:
    print(f"  [OK] Weight transfer parity: PERFECT")

# =====================================================================
# TASK 4: TFLite Conversion (Float16)
# =====================================================================
print("\n[4/5] Converting heads-only model to TFLite (Float16)...")

# Save to SavedModel first
sm_path = str(SAVED_MODEL_PATH)
heads_model.save(sm_path)
print(f"  SavedModel saved: {sm_path}")

# Float16 conversion
converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]

tflite_model = converter.convert()
with open(str(FLOAT16_PATH), "wb") as f:
    f.write(tflite_model)

size_mb = FLOAT16_PATH.stat().st_size / 1e6
print(f"  [OK] Float16 TFLite saved: {FLOAT16_PATH.name}")
print(f"  File size: {size_mb:.1f} MB")

if size_mb < 30:
    print(f"  [OK] SIZE GATE PASSED (< 30 MB for heads-only)")
elif size_mb < 60:
    print(f"  [OK] SIZE GATE PASSED (< 60 MB fallback)")
else:
    print(f"  [WARN] Size {size_mb:.1f} MB exceeds targets")

# =====================================================================
# TASK 5: TFLite Parity Check
# =====================================================================
print("\n[5/5] TFLite parity check...")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=str(FLOAT16_PATH))
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  TFLite inputs:")
for d in input_details:
    print(f"    {d['name']}: shape={d['shape']}, dtype={d['dtype']}")
print(f"  TFLite outputs:")
for d in output_details:
    print(f"    {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

# Run parity on 50 samples
ats_diffs_tfl = []

for idx in range(N_PARITY):
    resume = str(test_resumes[idx])
    jd = str(test_jds[idx])

    # Get embeddings
    encoder = full_model.get_layer("mobile_use_encoder")
    resume_emb = encoder(tf.constant([resume])).numpy().astype(np.float32)
    jd_emb     = encoder(tf.constant([jd])).numpy().astype(np.float32)

    # Keras heads prediction
    heads_out = heads_model([resume_emb, jd_emb], training=False)
    keras_ats = float(heads_out[0].numpy()[0][0]) * 100

    # TFLite prediction
    # Match input tensors by name
    for d in input_details:
        if "resume" in d["name"]:
            interpreter.set_tensor(d["index"], resume_emb)
        elif "jd" in d["name"]:
            interpreter.set_tensor(d["index"], jd_emb)

    interpreter.invoke()

    # Get outputs - match by shape
    tfl_ats = None
    for d in output_details:
        out = interpreter.get_tensor(d["index"])
        if out.shape[-1] == 1:
            tfl_ats = float(out[0][0]) * 100

    if tfl_ats is not None:
        diff = abs(keras_ats - tfl_ats)
        ats_diffs_tfl.append(diff)
        if idx < 5:
            print(f"  Sample {idx}: Keras={keras_ats:.2f}, TFLite={tfl_ats:.2f}, diff={diff:.4f}")

mean_diff = np.mean(ats_diffs_tfl) if ats_diffs_tfl else float("nan")
max_diff  = np.max(ats_diffs_tfl)  if ats_diffs_tfl else float("nan")

print(f"\n  --- TFLite Parity Results ---")
print(f"  ATS Score:  mean_diff={mean_diff:.4f} pts,  max_diff={max_diff:.4f} pts")
print(f"  Samples checked: {len(ats_diffs_tfl)}")

parity_pass = max_diff < 2.0 if not np.isnan(max_diff) else False
print(f"\n  Parity: {'PASS' if parity_pass else 'FAIL'} (tolerance: +/- 2.0 pts)")

# =====================================================================
# REPORT
# =====================================================================
print("\n" + "=" * 65)
print("  B-4 TFLITE CONVERSION SUMMARY")
print("=" * 65)

size_pass = size_mb < 60

print(f"  Architecture:      Heads-only (embedding inputs)")
print(f"  Quantization:      Float16")
print(f"  TFLite file:       {FLOAT16_PATH.name}")
print(f"  File size:         {size_mb:.1f} MB")
print(f"  Size gate:         {'PASS' if size_pass else 'FAIL'}")
print(f"  Parity mean diff:  {mean_diff:.4f} pts")
print(f"  Parity max diff:   {max_diff:.4f} pts")
print(f"  Parity gate:       {'PASS' if parity_pass else 'FAIL'}")
print(f"  Ready for Flutter: {'YES' if parity_pass and size_pass else 'NO'}")
print(f"\n  Full path: {FLOAT16_PATH}")

print(f"\n  Inference pipeline for Flutter:")
print(f"    1. Client: SentencePiece tokenize + USE Lite v2 encode (separate)")
print(f"    2. Client: Feed 512-dim embeddings to this TFLite model")
print(f"    3. Model outputs: [ATS_score(1), Domain_probs(7), RSG_probs(46)]")

# DEFINITION OF DONE
print("\n  DEFINITION OF DONE:")
print(f"  [{'OK' if FLOAT16_PATH.exists() else '!!'}] "
      f"{FLOAT16_PATH.name} produced")
print(f"  [{'OK' if size_pass else '!!'}] File size < 60 MB: "
      f"{size_mb:.1f} MB")
print(f"  [{'OK' if parity_pass else '!!'}] Parity < 2.0 pts: "
      f"max={max_diff:.4f} pts")

# Save summary
summary = {
    "stage": "B-4",
    "source_weights": str(B3C_WEIGHTS),
    "architecture": "heads-only (embedding inputs)",
    "quantization_type": "Float16",
    "tflite_file": FLOAT16_PATH.name,
    "tflite_path": str(FLOAT16_PATH),
    "file_size_mb": round(size_mb, 1),
    "size_gate_pass": size_pass,
    "parity_samples": len(ats_diffs_tfl),
    "parity_mean_diff": round(float(mean_diff), 4),
    "parity_max_diff": round(float(max_diff), 4),
    "parity_pass": bool(parity_pass),
    "ready_for_flutter": bool(parity_pass and size_pass),
    "input_spec": {
        "resume_embedding": "float32[batch, 512]",
        "jd_embedding": "float32[batch, 512]"
    },
    "output_spec": {
        "ats_score": "float32[batch, 1] -- sigmoid",
        "domain_probs": "float32[batch, 7] -- softmax",
        "rsg_template": "float32[batch, 46] -- softmax"
    }
}

with open(str(SUMMARY_JSON), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved: {SUMMARY_JSON}")

print("\n" + "=" * 65)
print("  B-4 TFLITE CONVERSION COMPLETE")
print("=" * 65)
