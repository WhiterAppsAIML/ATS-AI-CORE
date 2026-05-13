"""INJECTION-R3 — RSG Boost Pass

Freeze encoder + ATS/Domain heads; train only rsg_* layers with boosted RSG loss.
DriftGuard halts if ATS val MAE exceeds 8.5 (0-100 scale).

Run from project root:
    python scripts/r3_rsg_boost.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "ats-ai-core"))

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")

from sklearn.metrics import f1_score

from src.config import MINILM_MODEL_NAME

# ── Paths ─────────────────────────────────────────────────────────────────────
UNIFIED_MODEL_DIR = ROOT / "model" / "unified_model"
UNIFIED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

R2_WEIGHTS  = str(UNIFIED_MODEL_DIR / "new_unified_model" / "r2_best_weights.h5")
R3_BEST     = str(UNIFIED_MODEL_DIR / "r3_best_weights.h5")
R3_LOG      = str(UNIFIED_MODEL_DIR / "r3_training_log.csv")
DATA_DIR    = ROOT / "data" / "tokenized"
SPLITS_JSON = UNIFIED_MODEL_DIR / "data_splits.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR            = 5e-4
BATCH_SIZE    = 32
MAX_EPOCHS    = 25
PATIENCE      = 6
W_RSG         = 0.60   # boosted

# Gates
ATS_MAE_CEILING = 8.5
RSG_TARGET_ACC  = 0.62
DOMAIN_F1_FLOOR = 0.78

print("=" * 60)
print("  INJECTION-R3 — RSG Boost Pass")
print("=" * 60)
print(f"\n  RSG loss weight : {W_RSG}  (boosted)  |  encoder: frozen")
print(f"  LR: {LR}  |  max epochs: {MAX_EPOCHS}  |  patience: {PATIENCE}")
print(f"  ATS MAE ceiling: {ATS_MAE_CEILING}  |  RSG target: >{RSG_TARGET_ACC}")

# ── Load tokenized data ───────────────────────────────────────────────────────
print("\n[1/7] Loading tokenized data...")
ats_data = np.load(str(DATA_DIR / "ats_tokenized.npz"))
rsg_data = np.load(str(DATA_DIR / "rsg_tokenized.npz"))

with open(SPLITS_JSON) as f:
    splits = json.load(f)

ats_vl_idx = np.array(splits["ats_val"])
rsg_tr_idx = np.array(splits["rsg_train"])
rsg_vl_idx = np.array(splits["rsg_val"])

ATS_KEYS = (
    "resume_input_ids", "resume_attention_mask",
    "jd_input_ids", "jd_attention_mask",
    "ats_scores", "domain_labels",
)
RSG_KEYS = ("profile_input_ids", "profile_attention_mask", "rsg_labels")

ats_vl = {k: ats_data[k][ats_vl_idx] for k in ATS_KEYS}
rsg_tr = {k: rsg_data[k][rsg_tr_idx] for k in RSG_KEYS}
rsg_vl = {k: rsg_data[k][rsg_vl_idx] for k in RSG_KEYS}

n_rsg_tr = len(rsg_tr_idx)
print(f"  ATS  val ={len(ats_vl_idx):,}  (drift guard only)")
print(f"  RSG  train={n_rsg_tr:,}  val={len(rsg_vl_idx):,}")

# ── Build model and load R2 weights ───────────────────────────────────────────
print(f"\n[2/7] Building model and loading R2 weights...")
from transformers import TFAutoModel
from src.unified_engine.unified_model_minilm import build_unified_minilm_model

bert_model = TFAutoModel.from_pretrained(MINILM_MODEL_NAME, from_pt=True)
bert_model.trainable = False

model = build_unified_minilm_model(bert_model)
model.load_weights(R2_WEIGHTS)
print(f"  Loaded: {R2_WEIGHTS}")

# ── Freeze all; unfreeze only rsg_* layers ────────────────────────────────────
print("\n[3/7] Applying freeze strategy (only rsg_* layers trainable)...")
for layer in model.layers:
    layer.trainable = False

rsg_layer_names = []
for layer in model.layers:
    if layer.name.startswith("rsg_"):
        layer.trainable = True
        rsg_layer_names.append(layer.name)

trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
frozen    = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)
print(f"  Trainable layers : {rsg_layer_names}")
print(f"  Trainable params : {trainable:,}  |  Frozen: {frozen:,}")

# ── Measure R2 baseline (before any R3 gradient) ─────────────────────────────
print("\n[4/7] Measuring R2 baseline...")

def eval_ats(data):
    n = len(data["ats_scores"])
    ats_preds, dom_preds = [], []
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        ap, dp, _ = model(
            [data["resume_input_ids"][s:e],
             data["resume_attention_mask"][s:e],
             data["jd_input_ids"][s:e],
             data["jd_attention_mask"][s:e]],
            training=False,
        )
        ats_preds.append(ap.numpy())
        dom_preds.append(dp.numpy())
    ats_preds = np.concatenate(ats_preds).squeeze(-1)
    dom_preds = np.concatenate(dom_preds)
    mae_100  = float(np.mean(np.abs(ats_preds - data["ats_scores"]))) * 100.0
    dom_true = data["domain_labels"]
    dom_f1   = f1_score(dom_true, np.argmax(dom_preds, 1), average="macro", zero_division=0)
    return mae_100, dom_f1


def eval_rsg(data):
    m = len(data["rsg_labels"])
    rsg_preds = []
    for s in range(0, m, BATCH_SIZE):
        e    = min(s + BATCH_SIZE, m)
        pids = data["profile_input_ids"][s:e]
        pmsk = data["profile_attention_mask"][s:e]
        _, _, rp = model([pids, pmsk, pids, pmsk], training=False)
        rsg_preds.append(rp.numpy())
    rsg_preds = np.concatenate(rsg_preds)
    acc = float(np.mean(np.argmax(rsg_preds, 1) == data["rsg_labels"]))
    ce  = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            data["rsg_labels"], rsg_preds
        ).numpy()
    ))
    return acc, ce


r2_val_mae, r2_dom_f1   = eval_ats(ats_vl)
r2_rsg_acc, _           = eval_rsg(rsg_vl)
print(f"  R2 val ATS MAE : {r2_val_mae:.2f}")
print(f"  R2 val Dom F1  : {r2_dom_f1:.4f}")
print(f"  R2 val RSG Acc : {r2_rsg_acc:.4f}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
_sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def train_step_rsg(p_ids, p_mask, rsg_y):
    with tf.GradientTape() as tape:
        _, _, rsg_p = model([p_ids, p_mask, p_ids, p_mask], training=True)
        loss = W_RSG * _sce(rsg_y, rsg_p)
    optimizer.apply_gradients(
        zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
    )
    return float(loss)


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n[5/7] Training RSG head (max {MAX_EPOCHS} epochs, patience={PATIENCE})...\n")

rng              = np.random.default_rng(42)
best_rsg_acc     = -1.0
patience_counter = 0
best_epoch       = -1
log_rows         = []

batches_per_epoch = -(-n_rsg_tr // BATCH_SIZE)   # ceiling division
print(f"  RSG batches/epoch: {batches_per_epoch}\n")

for epoch in range(MAX_EPOCHS):
    perm       = rng.permutation(n_rsg_tr)
    batch_idxs = [perm[i:i + BATCH_SIZE] for i in range(0, n_rsg_tr, BATCH_SIZE)]

    epoch_loss = 0.0
    for bidx in batch_idxs:
        epoch_loss += train_step_rsg(
            rsg_tr["profile_input_ids"][bidx],
            rsg_tr["profile_attention_mask"][bidx],
            rsg_tr["rsg_labels"][bidx],
        )
    train_loss = epoch_loss / len(batch_idxs)

    if np.isnan(train_loss):
        print(f"\nHARD STOP — NaN train_loss at epoch {epoch + 1}")
        sys.exit(1)

    # Validation
    val_rsg_acc, val_rsg_ce = eval_rsg(rsg_vl)
    val_ats_mae, val_dom_f1 = eval_ats(ats_vl)

    print(
        f"Epoch {epoch+1:3d}/{MAX_EPOCHS}  "
        f"train={train_loss:.4f}  "
        f"RSG_Acc={val_rsg_acc:.4f}  ATS_MAE={val_ats_mae:.2f}  Dom_F1={val_dom_f1:.4f}"
    )

    log_rows.append({
        "epoch":         epoch + 1,
        "train_loss":    round(train_loss, 6),
        "val_rsg_acc":   round(val_rsg_acc, 4),
        "val_ats_mae":   round(val_ats_mae, 4),
        "val_dom_f1":    round(val_dom_f1, 4),
    })

    # DriftGuard hard stop
    if val_ats_mae > ATS_MAE_CEILING:
        print(
            f"\n  *** DRIFT GUARD — HARD STOP at epoch {epoch + 1}: "
            f"ATS MAE {val_ats_mae:.2f} > {ATS_MAE_CEILING} ceiling. "
            "Restoring best checkpoint. ***"
        )
        if best_epoch >= 0:
            model.load_weights(R3_BEST)
        break

    # NaN val guard
    if np.isnan(val_rsg_acc):
        print(f"\nHARD STOP — NaN val_rsg_acc at epoch {epoch + 1}")
        sys.exit(1)

    # Checkpoint on best RSG val accuracy
    if val_rsg_acc > best_rsg_acc:
        best_rsg_acc     = val_rsg_acc
        best_epoch       = epoch + 1
        patience_counter = 0
        model.save_weights(R3_BEST)
        print(f"  [Saved] RSG_Acc={val_rsg_acc:.4f}  ATS_MAE={val_ats_mae:.2f}  -> epoch {epoch + 1}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping (patience={PATIENCE}) at epoch {epoch + 1}")
            break

# ── Write CSV log ─────────────────────────────────────────────────────────────
if log_rows:
    with open(R3_LOG, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n  Training log -> {R3_LOG}")

# ── Final evaluation ──────────────────────────────────────────────────────────
print(f"\n[6/7] Loading best weights and running final evaluation...")
if best_epoch >= 0:
    model.load_weights(R3_BEST)

final_rsg_acc, _ = eval_rsg(rsg_vl)
final_ats_mae, final_dom_f1 = eval_ats(ats_vl)

ats_pass    = final_ats_mae <= ATS_MAE_CEILING
rsg_pass    = final_rsg_acc >= RSG_TARGET_ACC
domain_pass = final_dom_f1 >= DOMAIN_F1_FLOOR

print("\n" + "=" * 60)
print("  INJECTION-R3 GATE RESULTS")
print("=" * 60)
print(f"  RSG val accuracy   : {final_rsg_acc:.4f}  "
      f"{'PASS' if rsg_pass    else 'FAIL'}  (target > {RSG_TARGET_ACC})")
print(f"  ATS val MAE (0-100): {final_ats_mae:.3f}  "
      f"{'PASS' if ats_pass    else 'FAIL'}  (ceiling <= {ATS_MAE_CEILING})")
print(f"  Domain F1 (macro)  : {final_dom_f1:.4f}  "
      f"{'PASS' if domain_pass else 'FAIL'}  (floor >= {DOMAIN_F1_FLOOR})")
print(f"\n  R2 baseline RSG Acc: {r2_rsg_acc:.4f}  "
      f"->  delta: {final_rsg_acc - r2_rsg_acc:+.4f}")
print(f"  Best epoch         : {best_epoch}")
print(f"  Output             : {R3_BEST}")
print("=" * 60)

if ats_pass and rsg_pass and domain_pass:
    print("\nR3 COMPLETE — proceed to R4")
else:
    print("\nR3 INCOMPLETE — one or more gates failed. Review metrics above.")
