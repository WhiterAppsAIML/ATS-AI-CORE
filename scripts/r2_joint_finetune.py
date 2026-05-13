"""INJECTION-R2: Joint Fine-tuning

Unfreezes MiniLM encoder and trains end-to-end with differential LR:
  Encoder layers  → Adam(2e-5)
  Head layers     → Adam(5e-4)

Hard stops:
  - NaN loss at any epoch
  - ATS val MAE increases by > 3.0 pts from R1 baseline (regression guard)

Run from project root: python scripts/r2_joint_finetune.py
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "ats-ai-core"))

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

R1_WEIGHTS      = str(UNIFIED_MODEL_DIR / "r1_best_weights.h5")
R2_BEST_PATH    = str(UNIFIED_MODEL_DIR / "r2_best_weights.h5")
R2_CKPT_PATH    = str(UNIFIED_MODEL_DIR / "r2_best_in_progress.weights.h5")
R2_LOG_PATH     = str(UNIFIED_MODEL_DIR / "r2_training_log.csv")
DATA_DIR        = ROOT / "data" / "tokenized"
SPLITS_JSON     = UNIFIED_MODEL_DIR / "data_splits.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
ENC_LR        = 2e-5
HEAD_LR       = 5e-4
BATCH_SIZE    = 32
MAX_EPOCHS    = 50
PATIENCE      = 10
W_ATS         = 0.35
W_DOM         = 0.35
W_RSG         = 0.30
DOM_CLS_W     = [1.4, 0.8, 0.9, 1.0, 1.5, 0.9, 1.0]

# Hard-stop thresholds
MAE_REGRESSION_LIMIT = 3.0   # on 0-100 scale, relative to R1 baseline

# Target gates
GATE_VAL_MAE  = 8.0
GATE_TEST_MAE = 8.5
GATE_DOM_F1   = 0.82
GATE_RSG_ACC  = 0.55

DOMAIN_NAMES = ["IT", "Non-IT", "Design", "Healthcare", "Finance", "Legal", "Edu"]

# ── Load tokenized data ───────────────────────────────────────────────────────
print("=" * 60)
print("  Stage R-2: Joint Fine-tuning (MiniLM encoder unfrozen)")
print("=" * 60)
print(f"\n  Loss weights — ATS:{W_ATS}  DOM:{W_DOM}  RSG:{W_RSG}  (canonical)")
print(f"  LR — encoder:{ENC_LR}  heads:{HEAD_LR}")

print("\n[1/7] Loading tokenized data...")
ats_data = np.load(str(DATA_DIR / "ats_tokenized.npz"))
rsg_data = np.load(str(DATA_DIR / "rsg_tokenized.npz"))

with open(SPLITS_JSON) as f:
    splits = json.load(f)

ats_tr_idx = np.array(splits["ats_train"])
ats_vl_idx = np.array(splits["ats_val"])
ats_ts_idx = np.array(splits["ats_test"])
rsg_tr_idx = np.array(splits["rsg_train"])
rsg_vl_idx = np.array(splits["rsg_val"])

ATS_KEYS = (
    "resume_input_ids", "resume_attention_mask",
    "jd_input_ids", "jd_attention_mask",
    "ats_scores", "domain_labels",
)
RSG_KEYS = ("profile_input_ids", "profile_attention_mask", "rsg_labels")

ats_tr = {k: ats_data[k][ats_tr_idx] for k in ATS_KEYS}
ats_vl = {k: ats_data[k][ats_vl_idx] for k in ATS_KEYS}
ats_ts = {k: ats_data[k][ats_ts_idx] for k in ATS_KEYS}
rsg_tr = {k: rsg_data[k][rsg_tr_idx] for k in RSG_KEYS}
rsg_vl = {k: rsg_data[k][rsg_vl_idx] for k in RSG_KEYS}

n_ats_tr = len(ats_tr_idx)
n_rsg_tr = len(rsg_tr_idx)
print(f"  ATS  train={n_ats_tr:,}  val={len(ats_vl_idx):,}  test={len(ats_ts_idx):,}")
print(f"  RSG  train={n_rsg_tr:,}  val={len(rsg_vl_idx):,}")

# ── Build model and load R1 weights ───────────────────────────────────────────
print(f"\n[2/7] Building model and loading R1 weights...")
from transformers import TFAutoModel
from src.unified_engine.unified_model_minilm import build_unified_minilm_model

bert_model = TFAutoModel.from_pretrained(MINILM_MODEL_NAME, from_pt=True)
bert_model.trainable = False

model = build_unified_minilm_model(bert_model)
model.load_weights(R1_WEIGHTS)
print(f"  Loaded: {R1_WEIGHTS}")

# ── Loss helpers ──────────────────────────────────────────────────────────────
dom_w_tf = tf.constant(DOM_CLS_W, dtype=tf.float32)
_sce     = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def _ats_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - tf.squeeze(y_pred, axis=-1)))


def _dom_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    w  = tf.gather(dom_w_tf, tf.cast(y_true, tf.int32))
    return tf.reduce_mean(ce * w)


def _rsg_loss(y_true, y_pred):
    return _sce(y_true, y_pred)


# ── Validation helpers ────────────────────────────────────────────────────────

def eval_ats(data):
    """Evaluate ATS+Domain on a pre-sliced data dict. Returns (mae_100, dom_f1, dom_ce)."""
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
    dom_ce   = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(dom_true, dom_preds).numpy()
    ))
    return mae_100, dom_f1, dom_ce, dom_preds


def eval_rsg(data):
    """Evaluate RSG head on a pre-sliced data dict. Returns (acc, ce)."""
    m = len(data["rsg_labels"])
    rsg_preds = []
    for s in range(0, m, BATCH_SIZE):
        e    = min(s + BATCH_SIZE, m)
        pids = data["profile_input_ids"][s:e]
        pmsk = data["profile_attention_mask"][s:e]
        _, _, rp = model([pids, pmsk, pids, pmsk], training=False)
        rsg_preds.append(rp.numpy())

    rsg_preds = np.concatenate(rsg_preds)
    rsg_true  = data["rsg_labels"]
    acc = float(np.mean(np.argmax(rsg_preds, 1) == rsg_true))
    ce  = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(rsg_true, rsg_preds).numpy()
    ))
    return acc, ce


def composite_val_loss(mae_01, dom_ce, rsg_ce):
    return W_ATS * mae_01 + W_DOM * dom_ce + W_RSG * rsg_ce


# ── Measure R1 baseline before any R2 gradient ───────────────────────────────
print("\n[3/7] Measuring R1 baseline (before any R2 gradient)...")
r1_val_mae, r1_dom_f1, _, _ = eval_ats(ats_vl)
r1_rsg_acc, _               = eval_rsg(rsg_vl)
print(f"  R1 val ATS MAE : {r1_val_mae:.2f}")
print(f"  R1 val Dom F1  : {r1_dom_f1:.4f}")
print(f"  R1 val RSG Acc : {r1_rsg_acc:.4f}")
MAE_HARD_STOP = r1_val_mae + MAE_REGRESSION_LIMIT
print(f"  Regression hard-stop gate: val MAE > {MAE_HARD_STOP:.2f}")

# ── Unfreeze encoder; set up differential-LR optimizers ──────────────────────
print("\n[4/7] Unfreezing encoder with differential LR...")
encoder_layer = model.get_layer("minilm_encoder")
encoder_layer.trainable   = True
encoder_layer._bert.trainable = True

# Build a stable set of encoder variable ids AFTER setting trainable=True
_enc_var_ids = {id(v) for v in encoder_layer.trainable_variables}

encoder_opt = tf.keras.optimizers.legacy.Adam(learning_rate=ENC_LR)
head_opt    = tf.keras.optimizers.legacy.Adam(learning_rate=HEAD_LR)

trainable = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
frozen    = sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)
print(f"  Trainable params: {trainable:,}  |  Frozen: {frozen:,}")
print(f"  Encoder opt: Adam({ENC_LR})  |  Head opt: Adam({HEAD_LR})")


# ── Train step helpers ────────────────────────────────────────────────────────

def _apply_differential(grads):
    enc_gv  = [(g, v) for g, v in zip(grads, model.trainable_variables)
               if g is not None and id(v) in _enc_var_ids]
    head_gv = [(g, v) for g, v in zip(grads, model.trainable_variables)
               if g is not None and id(v) not in _enc_var_ids]
    if enc_gv:
        encoder_opt.apply_gradients(enc_gv)
    if head_gv:
        head_opt.apply_gradients(head_gv)


def train_step_ats(r_ids, r_mask, j_ids, j_mask, ats_y, dom_y):
    with tf.GradientTape() as tape:
        ats_p, dom_p, _ = model([r_ids, r_mask, j_ids, j_mask], training=True)
        l_ats = _ats_mae(ats_y, ats_p)
        l_dom = _dom_loss(dom_y, dom_p)
        loss  = W_ATS * l_ats + W_DOM * l_dom
    _apply_differential(tape.gradient(loss, model.trainable_variables))
    return float(loss), float(l_ats), float(l_dom)


def train_step_rsg(p_ids, p_mask, rsg_y):
    with tf.GradientTape() as tape:
        _, _, rsg_p = model([p_ids, p_mask, p_ids, p_mask], training=True)
        l_rsg = _rsg_loss(rsg_y, rsg_p)
        loss  = W_RSG * l_rsg
    _apply_differential(tape.gradient(loss, model.trainable_variables))
    return float(loss), float(l_rsg)


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n[5/7] Training (max {MAX_EPOCHS} epochs, patience={PATIENCE})...\n")

rng              = np.random.default_rng(42)
best_val_loss    = float("inf")
patience_counter = 0
best_epoch       = -1
log_rows         = []

ats_batches_per_epoch = math.ceil(n_ats_tr / BATCH_SIZE)
rsg_batches_per_epoch = math.ceil(n_rsg_tr / BATCH_SIZE)
print(f"  ATS batches/epoch: {ats_batches_per_epoch}  "
      f"RSG batches/epoch: {rsg_batches_per_epoch}\n")

for epoch in range(MAX_EPOCHS):
    ats_perm = rng.permutation(n_ats_tr)
    rsg_perm = rng.permutation(n_rsg_tr)

    ats_batch_idxs = [ats_perm[i:i + BATCH_SIZE] for i in range(0, n_ats_tr, BATCH_SIZE)]
    rsg_batch_idxs = [rsg_perm[i:i + BATCH_SIZE] for i in range(0, n_rsg_tr, BATCH_SIZE)]

    # Cycle RSG batches to match ATS count
    rsg_cycle = (rsg_batch_idxs * (len(ats_batch_idxs) // len(rsg_batch_idxs) + 1))
    rsg_cycle = rsg_cycle[:len(ats_batch_idxs)]

    epoch_ats_loss = 0.0
    epoch_rsg_loss = 0.0

    for aidx, ridx in zip(ats_batch_idxs, rsg_cycle):
        _, la, ld = train_step_ats(
            ats_tr["resume_input_ids"][aidx],
            ats_tr["resume_attention_mask"][aidx],
            ats_tr["jd_input_ids"][aidx],
            ats_tr["jd_attention_mask"][aidx],
            ats_tr["ats_scores"][aidx].astype(np.float32),
            ats_tr["domain_labels"][aidx],
        )
        epoch_ats_loss += W_ATS * la + W_DOM * ld

        _, l_rsg = train_step_rsg(
            rsg_tr["profile_input_ids"][ridx],
            rsg_tr["profile_attention_mask"][ridx],
            rsg_tr["rsg_labels"][ridx],
        )
        epoch_rsg_loss += W_RSG * l_rsg

    train_loss = (epoch_ats_loss + epoch_rsg_loss) / len(ats_batch_idxs)

    if np.isnan(train_loss):
        print(f"\nHARD STOP — NaN train_loss at epoch {epoch + 1}")
        sys.exit(1)

    # Validation
    val_mae, val_dom_f1, val_dom_ce, _ = eval_ats(ats_vl)
    val_rsg_acc, val_rsg_ce            = eval_rsg(rsg_vl)
    val_loss = composite_val_loss(val_mae / 100.0, val_dom_ce, val_rsg_ce)

    if np.isnan(val_loss):
        print(f"\nHARD STOP — NaN val_loss at epoch {epoch + 1}")
        sys.exit(1)

    print(
        f"Epoch {epoch+1:3d}/{MAX_EPOCHS}  "
        f"train={train_loss:.4f}  val={val_loss:.4f}  "
        f"ATS_MAE={val_mae:.2f}  Dom_F1={val_dom_f1:.4f}  RSG_Acc={val_rsg_acc:.4f}"
    )

    log_rows.append({
        "epoch":          epoch + 1,
        "train_loss":     round(train_loss, 6),
        "val_loss":       round(val_loss, 6),
        "val_ats_mae":    round(val_mae, 4),
        "val_dom_f1":     round(val_dom_f1, 4),
        "val_rsg_acc":    round(val_rsg_acc, 4),
    })

    # Regression hard stop
    if val_mae > MAE_HARD_STOP:
        print(
            f"\n  *** HARD STOP at epoch {epoch + 1}: "
            f"val ATS MAE {val_mae:.2f} > {MAE_HARD_STOP:.2f} "
            f"(R1 baseline {r1_val_mae:.2f} + {MAE_REGRESSION_LIMIT:.1f}). "
            "Restoring best checkpoint. ***"
        )
        if best_epoch >= 0:
            model.load_weights(R2_CKPT_PATH)
        sys.exit(1)

    # Checkpoint
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_epoch       = epoch + 1
        patience_counter = 0
        model.save_weights(R2_CKPT_PATH)
        print(f"  [Saved] best val_loss={val_loss:.4f}  ATS_MAE={val_mae:.2f} → epoch {epoch + 1}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping (patience={PATIENCE}) at epoch {epoch + 1}")
            break

# ── Write CSV log ─────────────────────────────────────────────────────────────
if log_rows:
    with open(R2_LOG_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n  Training log saved: {R2_LOG_PATH}")

# ── Save final best weights ───────────────────────────────────────────────────
print(f"\n[6/7] Saving r2_best_weights.h5...")
model.load_weights(R2_CKPT_PATH)
model.save_weights(R2_BEST_PATH)
print(f"  Saved: {R2_BEST_PATH}")

# ── Final evaluation (val + test) ─────────────────────────────────────────────
print(f"\n[7/7] Final evaluation...")

val_mae_final, val_dom_f1_final, _, _ = eval_ats(ats_vl)
val_rsg_acc_final, _                  = eval_rsg(rsg_vl)

test_mae_final, test_dom_f1_final, _, test_dom_preds = eval_ats(ats_ts)
per_dom_f1 = f1_score(
    ats_ts["domain_labels"],
    np.argmax(test_dom_preds, 1),
    average=None,
    labels=list(range(7)),
    zero_division=0,
)
test_rsg_acc_final, _ = eval_rsg(rsg_vl)   # RSG uses val split (no test split)

print("\n" + "=" * 60)
print("  STAGE R-2 — FINAL METRICS")
print("=" * 60)
print(f"  ATS val MAE  (0-100): {val_mae_final:.2f}  "
      f"{'PASS' if val_mae_final < GATE_VAL_MAE else 'FAIL'} (gate < {GATE_VAL_MAE})")
print(f"  ATS test MAE (0-100): {test_mae_final:.2f}  "
      f"{'PASS' if test_mae_final < GATE_TEST_MAE else 'FAIL'} (gate < {GATE_TEST_MAE})")
print(f"  Domain val F1 (macro): {val_dom_f1_final:.4f}  "
      f"{'PASS' if val_dom_f1_final > GATE_DOM_F1 else 'FAIL'} (gate > {GATE_DOM_F1})")
print(f"  RSG val Accuracy:      {val_rsg_acc_final:.4f}  "
      f"{'PASS' if val_rsg_acc_final > GATE_RSG_ACC else 'FAIL'} (gate > {GATE_RSG_ACC})")
print()
print("  Per-domain test F1:")
for i, (name, f1v) in enumerate(zip(DOMAIN_NAMES, per_dom_f1)):
    print(f"    [{i}] {name:12s}: {f1v:.4f}")
print()
print(f"  Best epoch    : {best_epoch}")
print(f"  Best val loss : {best_val_loss:.4f}")
print(f"  R1 baseline MAE: {r1_val_mae:.2f}  →  delta: {val_mae_final - r1_val_mae:+.2f}")
print(f"  Output        : {R2_BEST_PATH}")
print("=" * 60)
print()
print("R2 COMPLETE — proceed to R3")
