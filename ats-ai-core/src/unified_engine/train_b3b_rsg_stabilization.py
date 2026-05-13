"""
B-3b: RSG Head Architectural Stabilization
=============================================
Prior state: B-3 complete. RSG accuracy reached ~60%. Overfitting detected.
This pass: Added BatchNorm (rsg_bn3) + Dropout (rsg_drop3) after rsg_dense3.
           Load B-3 weights (existing dense layers keep 60%-accurate state).
           Train RSG head only with lr=1e-4 for fine-grained stabilization.

Config:
  Optimizer:      Adam  lr=1e-4  (lower -- fine-grained stabilization)
  ATS loss:       MSE              weight=0.0
  Domain loss:    SparseCCE        weight=0.0
  RSG loss:       SparseCCE        weight=1.0
  Epochs:         10
  Encoder:        FROZEN
  ATS head:       FROZEN
  Domain head:    FROZEN
  RSG head:       TRAINABLE
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # MUST be before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.unified_engine.unified_model import build_unified_model
from src.config import RSG_CSV_PATH

# -- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# B-3 weights (source -- architecture has changed, load with by_name=True)
B3_WEIGHTS = PROJECT_ROOT / "model" / "ats_model" / "unified_model_B3_RSG_fixed.h5"

# RSG data
RSG_CSV = RSG_CSV_PATH

# ATS data (for regression monitoring only)
ATS_CSV = PROJECT_ROOT / "data" / "labeled" / "merged_final.csv"

# Label mapping
UNIFIED_MODEL_DIR = PROJECT_ROOT / "model" / "unified_model"
MAPPING_JSON = UNIFIED_MODEL_DIR / "rsg_label_mapping.json"

# Output
OUTPUT_DIR = PROJECT_ROOT / "model" / "ats_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
B3B_WEIGHTS  = OUTPUT_DIR / "unified_model_B3b_final.h5"
B3B_LOG_CSV  = OUTPUT_DIR / "training_log_B3b.csv"

# -- Config ---------------------------------------------------------------
BATCH_SIZE   = 32
LR           = 1e-4      # Lower LR for fine-grained stabilization
MAX_EPOCHS   = 10
ES_PATIENCE  = 5
VAL_SPLIT    = 0.20
SEED         = 42
RSG_ACC_GATE = 0.65  # Stop early if we hit 65%

print("=" * 65)
print("  B-3b: RSG HEAD ARCHITECTURAL STABILIZATION")
print("=" * 65)

# -- 1. Build model (new architecture with rsg_bn3, rsg_drop3) -----------
print("\n[1/7] Building model (updated RSG head)...")
model = build_unified_model()

# Print new layers to confirm architecture change
rsg_layer_names = [l.name for l in model.layers if l.name.startswith("rsg")]
print(f"  RSG layers: {rsg_layer_names}")

# -- 2. Load B-3 weights (by_name=True -- new BN/Dropout layers init fresh)
print(f"\n[2/7] Loading B-3 weights: {B3_WEIGHTS}")
print("  Using by_name=True -- new layers (rsg_bn3, rsg_drop3) init fresh")
model.load_weights(str(B3_WEIGHTS), by_name=True, skip_mismatch=True)
print("  B-3 weights loaded successfully.")

# -- 3. Surgical Freezing Logic -------------------------------------------
print("\n[3/7] Applying surgical freeze...")

ENCODER_LAYERS = ["mobile_use_encoder"]
ATS_LAYERS     = ["ats_dense1", "ats_drop1", "ats_dense2", "ats_drop2", "ats_score"]
DOMAIN_LAYERS  = ["dom_dense1", "dom_drop1", "dom_dense2", "dom_drop2", "domain_probs"]
RSG_LAYERS     = ["rsg_dense1", "rsg_bn1", "rsg_drop1", "rsg_dense2", "rsg_bn2",
                  "rsg_drop2", "rsg_dense3", "rsg_bn3", "rsg_drop3", "rsg_template"]

# Freeze ALL layers first
for layer in model.layers:
    layer.trainable = False

# Unfreeze ONLY RSG head layers
for name in RSG_LAYERS:
    try:
        layer = model.get_layer(name)
        layer.trainable = True
        print(f"  [OK] UNFROZEN: {name}")
    except ValueError:
        print(f"  [!!] WARNING: Layer '{name}' not found!")

# -- Verification: Print freeze status ------------------------------------
print("\n  --- Freeze Status ---")
frozen_names    = []
trainable_names = []
for layer in model.layers:
    if layer.trainable and layer.count_params() > 0:
        trainable_names.append(layer.name)
    elif layer.count_params() > 0:
        frozen_names.append(layer.name)

print(f"  FROZEN layers with params ({len(frozen_names)}):")
for n in frozen_names:
    p = model.get_layer(n).count_params()
    print(f"    [LOCKED] {n:25s}  params={p:>10,}")

print(f"\n  TRAINABLE layers ({len(trainable_names)}):")
for n in trainable_names:
    p = model.get_layer(n).count_params()
    print(f"    [OPEN]   {n:25s}  params={p:>10,}")

# model.summary()
print("\n  --- Model Summary ---")
model.summary()

total_params     = model.count_params()
trainable_params = sum(
    tf.keras.backend.count_params(w) for w in model.trainable_weights
)
non_trainable    = total_params - trainable_params
print(f"\n  Total params:         {total_params:>12,}")
print(f"  Trainable params:     {trainable_params:>12,}")
print(f"  Non-trainable params: {non_trainable:>12,}")

# Sanity assertions
for name in ENCODER_LAYERS + ATS_LAYERS + DOMAIN_LAYERS:
    try:
        assert not model.get_layer(name).trainable, \
            f"FATAL: {name} must be frozen for B-3b!"
    except ValueError:
        pass

print("\n  [OK] Freeze verification passed.")

# -- 4. Load RSG label mapping --------------------------------------------
print("\n[4/7] Loading RSG label mapping...")
with open(str(MAPPING_JSON)) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}
num_classes = mapping["num_classes"]
print(f"  RSG classes: {num_classes}")

# -- 5. Load RSG data -----------------------------------------------------
print("\n[5/7] Loading RSG data...")
rsg_df = pd.read_csv(str(RSG_CSV)).dropna()
print(f"  Raw RSG rows: {len(rsg_df)}")

profile_texts    = rsg_df["profile_text"].astype(str).values
template_ids_raw = rsg_df["template_index"].astype(int).values

valid_mask       = np.array([int(tid) in id_to_idx for tid in template_ids_raw])
profile_texts_f  = profile_texts[valid_mask]
template_indices = np.array([id_to_idx[int(tid)] for tid in template_ids_raw[valid_mask]],
                            dtype="int32")

print(f"  After mapping filter: {len(profile_texts_f)} samples")
print(f"  Unique classes: {len(np.unique(template_indices))}")

# Train/Val split
rsg_idx = np.arange(len(profile_texts_f))
rsg_train_idx, rsg_val_idx = train_test_split(
    rsg_idx, test_size=VAL_SPLIT, random_state=SEED, stratify=template_indices
)
print(f"  RSG Train: {len(rsg_train_idx)}  Val: {len(rsg_val_idx)}")

def make_rsg_dataset(idxs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        profile_texts_f[idxs], template_indices[idxs]
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda prof, tmpl: (
        {"resume_text": prof, "jd_text": prof},
        {"ats_score": tf.zeros([tf.shape(prof)[0], 1]),
         "domain_probs": tf.zeros([tf.shape(prof)[0]], dtype=tf.int32),
         "rsg_template": tmpl}
    ))
    return ds

rsg_train_ds = make_rsg_dataset(rsg_train_idx, shuffle=True)
rsg_val_ds   = make_rsg_dataset(rsg_val_idx,   shuffle=False)

# -- Load ATS data for regression monitoring ------------------------------
print("\n[6/7] Loading ATS regression-check data...")
ats_df = pd.read_csv(str(ATS_CSV)).dropna()
ats_resume   = ats_df["resume_text"].astype(str).values
ats_jd       = ats_df["jd_text"].astype(str).values
ats_scores   = (ats_df["score"].astype(float) / 100.0).values.astype("float32")
ats_domains  = ats_df["domain_index"].astype(int).values.astype("int32")

_, ats_reg_idx = train_test_split(
    np.arange(len(ats_df)), test_size=VAL_SPLIT, random_state=SEED, stratify=ats_domains
)
ats_reg_ds = tf.data.Dataset.from_tensor_slices((
    ats_resume[ats_reg_idx], ats_jd[ats_reg_idx],
    ats_scores[ats_reg_idx], ats_domains[ats_reg_idx]
)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print(f"  ATS regression-check set: {len(ats_reg_idx)} samples")

# Baseline ATS/Domain metrics
print("  Computing baseline ATS/Domain metrics (pre-B3b)...")
ats_mae_fn   = tf.keras.losses.MeanAbsoluteError()
domain_ce_fn = tf.keras.losses.SparseCategoricalCrossentropy()

baseline_ats_maes, baseline_dom_accs = [], []
for r, jd, ats_t, dom_t in ats_reg_ds:
    ats_out, dom_out, _ = model([r, jd], training=False)
    mae = ats_mae_fn(tf.expand_dims(ats_t, 1), ats_out)
    baseline_ats_maes.append(float(mae))
    dom_pred = tf.argmax(dom_out, axis=1, output_type=tf.int32)
    dom_acc  = tf.reduce_mean(tf.cast(dom_pred == dom_t, tf.float32))
    baseline_dom_accs.append(float(dom_acc))

baseline_ats_mae = np.mean(baseline_ats_maes) * 100
baseline_dom_acc = np.mean(baseline_dom_accs) * 100
print(f"  Baseline ATS MAE (0-100):   {baseline_ats_mae:.2f}")
print(f"  Baseline Domain Acc:        {baseline_dom_acc:.1f}%")

# -- 6. Compile -- RSG-only loss ------------------------------------------
print("\n[7/7] Compiling & training...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss={
        "ats_score":    "mse",
        "domain_probs": "sparse_categorical_crossentropy",
        "rsg_template": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "ats_score":    0.0,
        "domain_probs": 0.0,
        "rsg_template": 1.0,
    },
    metrics={
        "ats_score":    ["mae"],
        "domain_probs": ["accuracy"],
        "rsg_template": ["accuracy"],
    }
)
print(f"  Compiled: lr={LR}, loss_weights=[ats=0.0, dom=0.0, rsg=1.0]")

# Re-confirm freeze after compile
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder became trainable after compile!"

# -- 7. Training ----------------------------------------------------------
print(f"\n  Starting B-3b stabilization training...")
print(f"  Max epochs:     {MAX_EPOCHS}")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Learning rate:  {LR}")
print(f"  EarlyStopping:  patience={ES_PATIENCE} on val_rsg_template_accuracy")
print(f"  Gate check:     stop if val_rsg_acc >= {RSG_ACC_GATE*100:.0f}%")
print(f"  Output:         {B3B_WEIGHTS}")
print()


class GateCheckCallback(tf.keras.callbacks.Callback):
    """Stop training early if RSG val accuracy exceeds the gate threshold."""
    def __init__(self, gate_acc):
        super().__init__()
        self.gate_acc = gate_acc

    def on_epoch_end(self, epoch, logs=None):
        val_rsg_acc = logs.get("val_rsg_template_accuracy", 0.0)
        if val_rsg_acc >= self.gate_acc:
            print(f"\n  >> GATE CHECK PASSED: val_rsg_acc={val_rsg_acc*100:.1f}% "
                  f">= {self.gate_acc*100:.0f}% -- stopping training.")
            self.model.stop_training = True


callbacks = [
    GateCheckCallback(gate_acc=RSG_ACC_GATE),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_rsg_template_accuracy",
        patience=ES_PATIENCE,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(B3B_WEIGHTS),
        monitor="val_rsg_template_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(str(B3B_LOG_CSV), separator=","),
]

# FINAL FREEZE CHECK
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder not frozen at training start!"

history = model.fit(
    rsg_train_ds,
    validation_data=rsg_val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -- Post-training freeze verification ------------------------------------
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder was unfrozen during training!"
print("\nEncoder freeze check post-training: CONFIRMED FROZEN")

# -- Regression Guard: ATS/Domain drift check ------------------------------
print("\n" + "=" * 65)
print("  REGRESSION GUARD -- ATS/Domain Drift Check")
print("=" * 65)

post_ats_maes, post_dom_accs = [], []
for r, jd, ats_t, dom_t in ats_reg_ds:
    ats_out, dom_out, _ = model([r, jd], training=False)
    mae = ats_mae_fn(tf.expand_dims(ats_t, 1), ats_out)
    post_ats_maes.append(float(mae))
    dom_pred = tf.argmax(dom_out, axis=1, output_type=tf.int32)
    dom_acc  = tf.reduce_mean(tf.cast(dom_pred == dom_t, tf.float32))
    post_dom_accs.append(float(dom_acc))

post_ats_mae = np.mean(post_ats_maes) * 100
post_dom_acc = np.mean(post_dom_accs) * 100

ats_drift = post_ats_mae - baseline_ats_mae
dom_drift = post_dom_acc - baseline_dom_acc

print(f"  ATS MAE  -- Pre: {baseline_ats_mae:.2f}  Post: {post_ats_mae:.2f}  delta={ats_drift:+.2f}")
print(f"  Domain   -- Pre: {baseline_dom_acc:.1f}%  Post: {post_dom_acc:.1f}%  delta={dom_drift:+.1f}%")

if abs(ats_drift) < 0.5:
    print("  ATS drift:    [OK] PASS (< 0.5 MAE change)")
else:
    print(f"  ATS drift:    [!!] WARNING -- {abs(ats_drift):.2f} MAE change detected!")

if abs(dom_drift) < 2.0:
    print("  Domain drift: [OK] PASS (< 2% accuracy change)")
else:
    print(f"  Domain drift: [!!] WARNING -- {abs(dom_drift):.1f}% accuracy change detected!")

# -- Training Summary -----------------------------------------------------
print("\n" + "=" * 65)
print("  B-3b RSG STABILIZATION -- TRAINING SUMMARY")
print("=" * 65)

h = history.history
epochs_trained = len(h["loss"])

val_rsg_acc_key = next((k for k in h if "val_rsg_template_accuracy" in k), None)

if val_rsg_acc_key:
    best_rsg_val_acc = max(h[val_rsg_acc_key])
    best_epoch = int(np.argmax(h[val_rsg_acc_key])) + 1
else:
    best_rsg_val_acc = 0.0
    best_epoch = epochs_trained

print(f"  Total epochs trained:    {epochs_trained}")
print(f"  Best epoch:              {best_epoch}")
print(f"  Best RSG val accuracy:   {best_rsg_val_acc*100:.1f}%")
print(f"  Final RSG val accuracy:  {h[val_rsg_acc_key][-1]*100:.1f}%")

# Epoch details
print("\n  --- Epoch 1 ---")
for k, v in h.items():
    print(f"    {k}: {v[0]:.6f}")

print(f"\n  --- Epoch {epochs_trained} ---")
for k, v in h.items():
    print(f"    {k}: {v[-1]:.6f}")

# NaN check
for k, v in h.items():
    if any(np.isnan(x) for x in v):
        print(f"\n  WARNING: NaN detected in {k}!")

print("=" * 65)

# -- Definition of Done ---------------------------------------------------
print("\n  DEFINITION OF DONE:")
print(f"  [{'OK' if 'rsg_bn3' in rsg_layer_names else '!!'}] Model summary reflects "
      f"new layers (rsg_bn3, rsg_drop3)")
print(f"  [{'OK' if best_rsg_val_acc >= 0.65 else '!!'}] RSG val accuracy >= 65%: "
      f"{best_rsg_val_acc*100:.1f}%")

# Verify weights file
if B3B_WEIGHTS.exists():
    wt_size = B3B_WEIGHTS.stat().st_size
    print(f"  [OK] {B3B_WEIGHTS.name} saved: {wt_size / 1e6:.1f} MB")
    assert wt_size > 0, "FATAL: weights file is empty!"
else:
    print(f"  [!!] {B3B_WEIGHTS.name} -- NOT FOUND!")

# Verify log CSV
if B3B_LOG_CSV.exists():
    log_df = pd.read_csv(str(B3B_LOG_CSV))
    print(f"  [OK] {B3B_LOG_CSV.name} saved: {len(log_df)} rows")

print(f"\n  Post-B3b ATS MAE:        {post_ats_mae:.2f}  (drift={ats_drift:+.2f})")
print(f"  Post-B3b Domain Acc:     {post_dom_acc:.1f}%  (drift={dom_drift:+.1f}%)")

print("\n" + "=" * 65)
if best_rsg_val_acc >= RSG_ACC_GATE:
    print(f"  [OK] B-3b COMPLETE -- RSG val accuracy {best_rsg_val_acc*100:.1f}% >= 65%")
    print("  Ready for Stage B-4 (TFLite Conversion).")
else:
    print(f"  [WARN] RSG val accuracy {best_rsg_val_acc*100:.1f}% < 65% target.")
    print("  Consider:")
    print("    - More training epochs")
    print("    - Label smoothing in loss function")
    print("    - Data augmentation for RSG dataset")

print("\nB-3b RSG ARCHITECTURAL STABILIZATION COMPLETE.")
