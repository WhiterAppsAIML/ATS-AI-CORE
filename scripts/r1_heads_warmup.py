"""INJECTION-R1: Heads Warm-up

Trains ATS, Domain, and RSG heads with the MiniLM encoder frozen.
Run from project root: python scripts/r1_heads_warmup.py
"""

import json
import math
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
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
from sklearn.metrics import f1_score

from src.config import MINILM_MODEL_NAME

# ── Paths ─────────────────────────────────────────────────────────────────────
UNIFIED_MODEL_DIR = ROOT / "model" / "unified_model"
UNIFIED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = str(UNIFIED_MODEL_DIR / "r1_best_weights.h5")
DATA_DIR        = ROOT / "data" / "tokenized"
SPLITS_JSON     = UNIFIED_MODEL_DIR / "data_splits.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR            = 1e-3
BATCH_SIZE    = 32
MAX_EPOCHS    = 40
PATIENCE      = 8
W_ATS         = 0.35
W_DOM         = 0.35
W_RSG         = 0.30
DOM_CLS_W     = [1.4, 0.8, 0.9, 1.0, 1.5, 0.9, 1.0]   # indexed by domain label 0–6

# ── Load tokenized data ───────────────────────────────────────────────────────
print("Loading tokenized data ...")
ats_data = np.load(str(DATA_DIR / "ats_tokenized.npz"))
rsg_data = np.load(str(DATA_DIR / "rsg_tokenized.npz"))

with open(SPLITS_JSON) as f:
    splits = json.load(f)

ats_tr_idx = np.array(splits["ats_train"])
ats_vl_idx = np.array(splits["ats_val"])
rsg_tr_idx = np.array(splits["rsg_train"])
rsg_vl_idx = np.array(splits["rsg_val"])

ATS_TR_KEYS = ("resume_input_ids", "resume_attention_mask",
               "jd_input_ids", "jd_attention_mask", "ats_scores", "domain_labels")
RSG_TR_KEYS = ("profile_input_ids", "profile_attention_mask", "rsg_labels")

ats_tr = {k: ats_data[k][ats_tr_idx] for k in ATS_TR_KEYS}
ats_vl = {k: ats_data[k][ats_vl_idx] for k in ATS_TR_KEYS}
rsg_tr = {k: rsg_data[k][rsg_tr_idx] for k in RSG_TR_KEYS}
rsg_vl = {k: rsg_data[k][rsg_vl_idx] for k in RSG_TR_KEYS}

n_ats_tr = len(ats_tr_idx)
n_rsg_tr = len(rsg_tr_idx)
print(f"  ATS  train={n_ats_tr:,}  val={len(ats_vl_idx):,}")
print(f"  RSG  train={n_rsg_tr:,}  val={len(rsg_vl_idx):,}")

# ── Build model ───────────────────────────────────────────────────────────────
print(f"Loading MiniLM encoder ({MINILM_MODEL_NAME}) ...")
from transformers import TFAutoModel
from src.unified_engine.unified_model_minilm import build_unified_minilm_model

bert_model = TFAutoModel.from_pretrained(MINILM_MODEL_NAME, from_pt=True)
bert_model.trainable = False

model = build_unified_minilm_model(bert_model)

for layer in model.layers:
    if "minilm" in layer.name or "encoder" in layer.name:
        layer.trainable = False

trainable   = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
frozen      = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)
print(f"  Trainable params: {trainable:,}  |  Frozen: {frozen:,}")

# ── Loss helpers ──────────────────────────────────────────────────────────────
dom_w_tf  = tf.constant(DOM_CLS_W, dtype=tf.float32)
_sce      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def _ats_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - tf.squeeze(y_pred, axis=-1)))


def _dom_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    w  = tf.gather(dom_w_tf, tf.cast(y_true, tf.int32))
    return tf.reduce_mean(ce * w)


def _rsg_loss(y_true, y_pred):
    return _sce(y_true, y_pred)


# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)


# ── Train steps ───────────────────────────────────────────────────────────────
def train_step_ats(r_ids, r_mask, j_ids, j_mask, ats_y, dom_y):
    with tf.GradientTape() as tape:
        ats_p, dom_p, _ = model([r_ids, r_mask, j_ids, j_mask], training=True)
        l_ats  = _ats_mae(ats_y, ats_p)
        l_dom  = _dom_loss(dom_y, dom_p)
        loss   = W_ATS * l_ats + W_DOM * l_dom
    optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                  model.trainable_variables))
    return float(loss), float(l_ats), float(l_dom)


def train_step_rsg(p_ids, p_mask, rsg_y):
    with tf.GradientTape() as tape:
        _, _, rsg_p = model([p_ids, p_mask, p_ids, p_mask], training=True)
        l_rsg = _rsg_loss(rsg_y, rsg_p)
        loss  = W_RSG * l_rsg
    optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                  model.trainable_variables))
    return float(loss), float(l_rsg)


# ── Validation ────────────────────────────────────────────────────────────────
def run_validation():
    # --- ATS val ---
    n    = len(ats_vl_idx)
    ats_preds, dom_preds = [], []
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        ap, dp, _ = model(
            [ats_vl["resume_input_ids"][s:e],
             ats_vl["resume_attention_mask"][s:e],
             ats_vl["jd_input_ids"][s:e],
             ats_vl["jd_attention_mask"][s:e]],
            training=False,
        )
        ats_preds.append(ap.numpy())
        dom_preds.append(dp.numpy())

    ats_preds = np.concatenate(ats_preds, axis=0).squeeze(-1)
    dom_preds = np.concatenate(dom_preds, axis=0)

    val_ats_mae_01  = float(np.mean(np.abs(ats_preds - ats_vl["ats_scores"])))
    val_ats_mae_100 = val_ats_mae_01 * 100.0

    dom_true    = ats_vl["domain_labels"]
    dom_classes = np.argmax(dom_preds, axis=1)
    val_dom_f1  = f1_score(dom_true, dom_classes, average="macro", zero_division=0)
    val_dom_ce  = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(dom_true, dom_preds).numpy()
    ))

    # --- RSG val ---
    m    = len(rsg_vl_idx)
    rsg_preds = []
    for s in range(0, m, BATCH_SIZE):
        e    = min(s + BATCH_SIZE, m)
        pids = rsg_vl["profile_input_ids"][s:e]
        pmsk = rsg_vl["profile_attention_mask"][s:e]
        _, _, rp = model([pids, pmsk, pids, pmsk], training=False)
        rsg_preds.append(rp.numpy())

    rsg_preds   = np.concatenate(rsg_preds, axis=0)
    rsg_true    = rsg_vl["rsg_labels"]
    rsg_classes = np.argmax(rsg_preds, axis=1)
    val_rsg_acc = float(np.mean(rsg_classes == rsg_true))
    val_rsg_ce  = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(rsg_true, rsg_preds).numpy()
    ))

    # Combined scalar for early stopping
    val_loss = W_ATS * val_ats_mae_01 + W_DOM * val_dom_ce + W_RSG * val_rsg_ce
    return val_loss, val_ats_mae_100, val_dom_f1, val_rsg_acc


# ── Training loop ──────────────────────────────────────────────────────────────
rng              = np.random.default_rng(42)
best_val_loss    = float("inf")
patience_counter = 0
best_epoch       = -1

ats_batches_per_epoch = math.ceil(n_ats_tr / BATCH_SIZE)
rsg_batches_per_epoch = math.ceil(n_rsg_tr / BATCH_SIZE)

print(f"\nStarting training: max_epochs={MAX_EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
print(f"  ATS batches/epoch: {ats_batches_per_epoch}  RSG batches/epoch: {rsg_batches_per_epoch}\n")

for epoch in range(MAX_EPOCHS):
    ats_perm = rng.permutation(n_ats_tr)
    rsg_perm = rng.permutation(n_rsg_tr)

    ats_batch_idxs = [ats_perm[i:i+BATCH_SIZE] for i in range(0, n_ats_tr, BATCH_SIZE)]
    rsg_batch_idxs = [rsg_perm[i:i+BATCH_SIZE] for i in range(0, n_rsg_tr, BATCH_SIZE)]

    # Cycle RSG batches to match ATS count
    rsg_cycle = (rsg_batch_idxs * (len(ats_batch_idxs) // len(rsg_batch_idxs) + 1))
    rsg_cycle = rsg_cycle[:len(ats_batch_idxs)]

    epoch_loss = 0.0
    for aidx, ridx in zip(ats_batch_idxs, rsg_cycle):
        # ATS + Domain step
        _, la, ld = train_step_ats(
            ats_tr["resume_input_ids"][aidx],
            ats_tr["resume_attention_mask"][aidx],
            ats_tr["jd_input_ids"][aidx],
            ats_tr["jd_attention_mask"][aidx],
            ats_tr["ats_scores"][aidx].astype(np.float32),
            ats_tr["domain_labels"][aidx],
        )
        epoch_loss += W_ATS * la + W_DOM * ld

        # RSG step (profile as both resume and JD)
        _rsg_total, l_rsg = train_step_rsg(
            rsg_tr["profile_input_ids"][ridx],
            rsg_tr["profile_attention_mask"][ridx],
            rsg_tr["rsg_labels"][ridx],
        )
        epoch_loss += W_RSG * l_rsg

    train_loss = epoch_loss / len(ats_batch_idxs)

    if np.isnan(train_loss):
        print(f"HARD STOP — NaN train_loss at epoch {epoch + 1}")
        sys.exit(1)

    val_loss, val_mae, val_f1, val_acc = run_validation()

    if np.isnan(val_loss):
        print(f"HARD STOP — NaN val_loss at epoch {epoch + 1}")
        sys.exit(1)

    print(
        f"Epoch {epoch+1:3d}/{MAX_EPOCHS}  "
        f"train={train_loss:.4f}  val={val_loss:.4f}  "
        f"ATS_MAE={val_mae:.2f}  Dom_F1={val_f1:.4f}  RSG_Acc={val_acc:.4f}"
    )

    # Hard gate: ATS MAE > 15.0 after epoch 10
    if epoch >= 9 and val_mae > 15.0:
        print(f"HARD STOP — ATS MAE {val_mae:.2f} > 15.0 after epoch {epoch + 1}")
        sys.exit(1)

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_epoch       = epoch + 1
        patience_counter = 0
        model.save_weights(CHECKPOINT_PATH)
        print(f"  [Saved] best weights → epoch {epoch + 1}  val_loss={val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping (patience={PATIENCE}) at epoch {epoch + 1}")
            break

# ── Final summary ──────────────────────────────────────────────────────────────
model.load_weights(CHECKPOINT_PATH)
_, final_mae, final_f1, final_acc = run_validation()

print("\n" + "=" * 60)
print("  R1 HEADS WARM-UP SUMMARY")
print("=" * 60)
print(f"  Best epoch    : {best_epoch}")
print(f"  ATS val MAE   : {final_mae:.2f}  (target <10.0)  "
      f"{'PASS' if final_mae < 10.0 else 'soft miss'}")
print(f"  Domain val F1 : {final_f1:.4f}  (target >0.72)  "
      f"{'PASS' if final_f1 > 0.72 else 'soft miss'}")
print(f"  RSG val Acc   : {final_acc:.4f}  (target >0.42)  "
      f"{'PASS' if final_acc > 0.42 else 'soft miss'}")
print(f"  Checkpoint    : {CHECKPOINT_PATH}")
print("=" * 60)
print("R1 COMPLETE — proceed to R2")
