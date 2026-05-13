"""
B-2: Retraining Execution — USE Lite v2 Head Tuning
====================================================
Trains ATS + Domain heads with the frozen USELiteEncoder.
RSG head is present in the graph but not optimised this cycle
(no RSG labels in merged_final.csv).

Config:
  Optimizer:      Adam  lr=1e-4
  ATS loss:       MSE              weight=1.0
  Domain loss:    SparseCCE        weight=0.5
  RSG loss:       None (no labels) weight=0.0
  Epochs:         10  (EarlyStopping patience=3 on val_loss)
  Val split:      20%
  Encoder:        FROZEN (trainable=False)
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # MUST be before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.unified_engine.unified_model import build_unified_model

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
ATS_CSV      = PROJECT_ROOT / "data" / "labeled" / "merged_final.csv"
OUTPUT_DIR   = PROJECT_ROOT / "model" / "ats_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_FILE = OUTPUT_DIR / "unified_model_lite_v2.h5"
LOG_CSV      = OUTPUT_DIR / "training_log_lite_v2.csv"

# ── Config ─────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
LR           = 1e-4
MAX_EPOCHS   = 10
ES_PATIENCE  = 3
VAL_SPLIT    = 0.20
SEED         = 42

print("=" * 65)
print("  B-2: RETRAINING EXECUTION — USE Lite v2 HEAD TUNING")
print("=" * 65)

# ── 1. Build model ────────────────────────────────────────────────────
print("\n[1/6] Building model...")
model = build_unified_model()

# REGRESSION GUARD: Double-check encoder is frozen
enc = model.get_layer("mobile_use_encoder")
assert not enc.trainable, "FATAL: mobile_use_encoder must be frozen!"
print(f"  mobile_use_encoder.trainable = {enc.trainable}  — CONFIRMED FROZEN")

trainable_layers = [l.name for l in model.layers if l.trainable and l.count_params() > 0]
frozen_layers    = [l.name for l in model.layers if not l.trainable]
print(f"  Trainable layers ({len(trainable_layers)}): {trainable_layers}")
print(f"  Frozen layers    ({len(frozen_layers)}): {frozen_layers}")

# ── 2. Compile ────────────────────────────────────────────────────────
print("\n[2/6] Compiling model...")
# NOTE: domain labels are integer indices → sparse_categorical_crossentropy
# NOTE: RSG labels not available in dataset → loss=None
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss={
        "ats_score":    "mean_absolute_error",
        "domain_probs": "sparse_categorical_crossentropy",
        "rsg_template": None,
    },
    loss_weights={
        "ats_score":    1.0,
        "domain_probs": 0.5,
    },
    metrics={
        "ats_score":    ["mae"],
        "domain_probs": ["accuracy"],
    }
)
print("  Compiled successfully — no AttributeError.")

# Re-confirm freeze after compile
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder became trainable after compile!"

# ── 3. Load data ──────────────────────────────────────────────────────
print("\n[3/6] Loading data...")
df = pd.read_csv(str(ATS_CSV)).dropna()
print(f"  Loaded {len(df)} rows from {ATS_CSV.name}")
print(f"  Columns: {list(df.columns)}")

resume_texts  = df["resume_text"].astype(str).values
jd_texts      = df["jd_text"].astype(str).values
ats_scores    = (df["score"].astype(float) / 100.0).values.astype("float32")
domain_labels = df["domain_index"].astype(int).values.astype("int32")

# Sanity checks
print(f"  ATS score range (0-1): [{ats_scores.min():.3f}, {ats_scores.max():.3f}]")
print(f"  Domain label range:    [{domain_labels.min()}, {domain_labels.max()}]")
print(f"  Domain label counts:   {np.bincount(domain_labels)}")

# NaN guard
assert not np.any(np.isnan(ats_scores)), "FATAL: NaN in ATS scores!"

# ── 4. Train/Val split ────────────────────────────────────────────────
idx = np.arange(len(df))
train_idx, val_idx = train_test_split(
    idx, test_size=VAL_SPLIT, random_state=SEED, stratify=domain_labels
)
print(f"\n  Train: {len(train_idx)}  Val: {len(val_idx)}")
print(f"  Split: {len(train_idx)/len(df):.0%} / {len(val_idx)/len(df):.0%}")

# Build tf.data pipelines
def make_dataset(idxs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        resume_texts[idxs], jd_texts[idxs],
        ats_scores[idxs], domain_labels[idxs]
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda r, j, a, d: (
        {"resume_text": r, "jd_text": j},
        {"ats_score": tf.expand_dims(a, 1), "domain_probs": d}
    ))
    return ds

train_ds = make_dataset(train_idx, shuffle=True)
val_ds   = make_dataset(val_idx,   shuffle=False)

# ── 5. Callbacks ──────────────────────────────────────────────────────
print("\n[4/6] Configuring callbacks...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=ES_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(WEIGHTS_FILE),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(str(LOG_CSV), separator=","),
]

# ── 6. Train ──────────────────────────────────────────────────────────
print("\n[5/6] Starting training...")
print(f"  Max epochs:     {MAX_EPOCHS}")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Learning rate:  {LR}")
print(f"  EarlyStopping:  patience={ES_PATIENCE} on val_loss")
print(f"  Weights save:   {WEIGHTS_FILE}")
print(f"  Log CSV:        {LOG_CSV}")
print()

# FINAL FREEZE CHECK before fit()
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder not frozen at training start!"

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ── Post-training encoder freeze check ─────────────────────────────────
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder was unfrozen during training!"
print("\nEncoder freeze check post-training: CONFIRMED FROZEN")

# ── 7. Training Summary ───────────────────────────────────────────────
print("\n[6/6] Training summary...")
h = history.history

val_losses = h["val_loss"]
best_epoch = int(np.argmin(val_losses)) + 1

# Find metric keys
ats_mae_val_key = next((k for k in h if "val_ats_score_mae" in k), None)
dom_acc_val_key = next((k for k in h if "val_domain_probs_accuracy" in k), None)

print()
print("=" * 65)
print("  B-2 TRAINING SUMMARY — USE Lite v2 Head Tuning")
print("=" * 65)
print(f"  Total epochs trained:  {len(val_losses)}")
print(f"  Best epoch:            {best_epoch}")
print(f"  Best val_loss:         {val_losses[best_epoch - 1]:.6f}")

if ats_mae_val_key:
    val_ats_mae = h[ats_mae_val_key][best_epoch - 1] * 100
    print(f"  Best val_ats_mae:      {val_ats_mae:.2f}  (0-100 scale)")

if dom_acc_val_key:
    val_dom_acc = h[dom_acc_val_key][best_epoch - 1] * 100
    print(f"  Best val_domain_acc:   {val_dom_acc:.1f}%")

print(f"\n  Weights saved:         {WEIGHTS_FILE}")
print(f"  Log CSV:               {LOG_CSV}")

# --- First epoch log ---
print("\n--- Epoch 1 ---")
for k, v in h.items():
    print(f"  {k}: {v[0]:.6f}")

# --- Last epoch log ---
last = len(val_losses)
print(f"\n--- Epoch {last} ---")
for k, v in h.items():
    print(f"  {k}: {v[-1]:.6f}")

# NaN check
for k, v in h.items():
    if any(np.isnan(x) for x in v):
        print(f"\n  WARNING: NaN detected in {k}!")

print("=" * 65)

# Verify weights file
if WEIGHTS_FILE.exists():
    wt_size = WEIGHTS_FILE.stat().st_size
    print(f"\n  unified_model_lite_v2.h5 size: {wt_size / 1e6:.1f} MB")
    assert wt_size > 0, "FATAL: weights file is empty!"
else:
    print("\n  ERROR: Weight file not found!")

# Verify log CSV
if LOG_CSV.exists():
    log_df = pd.read_csv(str(LOG_CSV))
    print(f"  training_log_lite_v2.csv rows: {len(log_df)}")
    print(f"  Columns: {list(log_df.columns)}")

print("\nB-2 COMPLETE — Head tuning finished.")
