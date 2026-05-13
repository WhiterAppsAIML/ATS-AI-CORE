"""
M-2: Full retraining cycle 1 — MobileUSE encoder (frozen)
Config: Same as USE v4 run that produced MAE 5.09
Split:  75 / 15 / 10  (train / val / test)
Heads:  ATS (MAE) + Domain (SparseCCE).  RSG head present but not optimised.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # MUST be before any TF import

import sys
sys.path.insert(0, ".")

import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.unified_engine.unified_model import build_unified_model

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ATS_CSV      = PROJECT_ROOT / "data" / "labeled" / "merged_final.csv"
OUTPUT_DIR   = PROJECT_ROOT / "model" / "ats_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_WEIGHTS = OUTPUT_DIR / "best_model_mobileuse_cycle1.h5"
LOG_CSV      = OUTPUT_DIR / "training_log_mobileuse_cycle1.csv"
# ───────────────────────────────────────────────────────────────────────

# ── Config (same as USE v4 run) ────────────────────────────────────────
BATCH_SIZE   = 32
INIT_LR      = 1e-4
MAX_EPOCHS   = 150
PATIENCE     = 15     # early stopping patience
LR_PATIENCE  = 8      # ReduceLROnPlateau patience
LR_FACTOR    = 0.5
MIN_LR       = 1e-6
# ───────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  M-2: FULL RETRAINING — MobileUSE CYCLE 1")
print("=" * 65)

# ── 1. Build model ────────────────────────────────────────────────────
print("\n[1/5] Building unified model...")
model = build_unified_model()

# ── 2. Confirm MobileUSE frozen ───────────────────────────────────────
print("[2/5] Confirming MobileUSE encoder is frozen...")
enc = model.get_layer("mobile_use_encoder")
assert not enc.trainable, "FATAL: MobileUSE encoder must be frozen!"
print(f"  mobile_use_encoder.trainable = {enc.trainable}  — CONFIRMED FROZEN")

trainable_names = [l.name for l in model.layers if l.trainable]
frozen_names    = [l.name for l in model.layers if not l.trainable]
print(f"  Trainable layers ({len(trainable_names)}): {trainable_names}")
print(f"  Frozen layers    ({len(frozen_names)}): {frozen_names}")

# ── 3. Compile ────────────────────────────────────────────────────────
print("\n[3/5] Compiling model (ATS MAE + Domain SparseCCE)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    loss={
        "ats_score":    "mean_absolute_error",
        "domain_probs": "sparse_categorical_crossentropy",
        "rsg_template": None,    # RSG not trained this cycle
    },
    metrics={
        "ats_score":    ["mae"],
        "domain_probs": ["accuracy"],
    }
)
print("  Compiled.")

# Re-confirm encoder still frozen after compile
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder became trainable after compile!"

# ── 4. Load data — 75/15/10 split ─────────────────────────────────────
print("\n[4/5] Loading data...")
df = pd.read_csv(str(ATS_CSV)).dropna()
print(f"  Loaded {len(df)} rows from {ATS_CSV.name}")

resume_texts  = df["resume_text"].astype(str).values
jd_texts      = df["jd_text"].astype(str).values
ats_scores    = (df["score"].astype(float) / 100.0).values.astype("float32")
domain_labels = df["domain_index"].astype(int).values.astype("int32")

# 75/15/10 split (train/val/test)
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.25, random_state=42)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.40, random_state=42)
#  0.25 * 0.60 = 0.15 (val),  0.25 * 0.40 = 0.10 (test)

print(f"  Split — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")
print(f"  Split ratios: {len(train_idx)/len(df):.2f} / {len(val_idx)/len(df):.2f} / {len(test_idx)/len(df):.2f}")

# Build tf.data pipelines
def make_dataset(idxs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        resume_texts[idxs], jd_texts[idxs],
        ats_scores[idxs], domain_labels[idxs]
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # Reformat into (inputs, outputs) for model.fit
    ds = ds.map(lambda r, j, a, d: (
        {"resume_text": r, "jd_text": j},
        {"ats_score": tf.expand_dims(a, 1), "domain_probs": d}
    ))
    return ds

train_ds = make_dataset(train_idx, shuffle=True)
val_ds   = make_dataset(val_idx,   shuffle=False)

# ── 5. Callbacks ───────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_WEIGHTS),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(str(LOG_CSV), separator=",")
]

# ── Training ───────────────────────────────────────────────────────────
print("\n[5/5] Starting training...")
print(f"  Max epochs:     {MAX_EPOCHS}")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Initial LR:     {INIT_LR}")
print(f"  LR schedule:    ReduceLROnPlateau (factor={LR_FACTOR}, patience={LR_PATIENCE})")
print(f"  Early stopping: patience={PATIENCE} on val_loss")
print(f"  Weights save:   {BEST_WEIGHTS}")
print(f"  Log CSV:        {LOG_CSV}")
print()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ── Final encoder freeze check ─────────────────────────────────────────
assert not model.get_layer("mobile_use_encoder").trainable, \
    "FATAL: encoder was unfrozen during training!"
print("\nMobileUSE encoder frozen check post-training: CONFIRMED FROZEN")

# ── Training summary ───────────────────────────────────────────────────
h = history.history
# CSVLogger uses column names from Keras — find the right keys
ats_mae_key     = [k for k in h if "ats_score_mae" in k and "val" not in k][0]
val_ats_mae_key = [k for k in h if "val_ats_score_mae" in k][0]
dom_acc_key     = [k for k in h if "domain_probs_accuracy" in k and "val" not in k][0]
val_dom_acc_key = [k for k in h if "val_domain_probs_accuracy" in k][0]

val_losses  = h["val_loss"]
best_epoch  = int(np.argmin(val_losses)) + 1
best_val_loss    = val_losses[best_epoch - 1]
best_val_ats_mae = h[val_ats_mae_key][best_epoch - 1] * 100   # 0-100 scale
best_val_dom_acc = h[val_dom_acc_key][best_epoch - 1] * 100

print()
print("=" * 65)
print("  M-2 TRAINING SUMMARY — MobileUSE Cycle 1")
print("=" * 65)
print(f"  Total epochs trained: {len(val_losses)}")
print(f"  Best epoch:           {best_epoch}")
print(f"  Best val_loss:        {best_val_loss:.6f}")
print(f"  Best val_ats_mae:     {best_val_ats_mae:.2f}  (0-100 scale)")
print(f"  Best val_domain_acc:  {best_val_dom_acc:.1f}%")
print(f"  Weights saved:        {BEST_WEIGHTS}")
print(f"  Log CSV:              {LOG_CSV}")
print("=" * 65)

# Verify weights file is non-empty
wt_size = BEST_WEIGHTS.stat().st_size if BEST_WEIGHTS.exists() else 0
print(f"\n  best_model_mobileuse_cycle1.h5 size: {wt_size / 1e6:.1f} MB")
assert wt_size > 0, "FATAL: weights file is empty!"

# Verify log CSV exists and has correct columns
log_df = pd.read_csv(str(LOG_CSV))
print(f"  training_log_mobileuse_cycle1.csv rows: {len(log_df)}")
print(f"  Columns: {list(log_df.columns)}")

print("\nM-2 COMPLETE — MobileUSE cycle 1 training finished.")
