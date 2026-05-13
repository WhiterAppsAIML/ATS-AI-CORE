"""
Stage R-4: Joint Fine-Tuning

Load R-3 weights, freeze encoder, unfreeze all heads, alternate ATS+Domain/RSG
batches at Adam(5e-6) with canonical loss weights (0.35/0.35/0.30).

Hard stop: if val_ats_mae > 6.5 (0-100 scale) at end of any epoch, restore best
checkpoint and break.

Output: model/unified_model/r4_joint_best.weights.h5
"""

import csv
import json
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "ats-ai-core")))

from src.unified_engine.data_loader import load_ats_data, load_rsg_data
from src.unified_engine.unified_model import build_unified_model
from src.config import (
    LABELED_DIR, RSG_CSV_PATH, RSG_MAPPING_JSON,
    SCORE_LOSS_WEIGHT, DOMAIN_LOSS_WEIGHT, RSG_LOSS_WEIGHT,
    DOMAIN_CLASS_WEIGHTS,
)

# ── Constants ────────────────────────────────────────────────────────────────
R2_BASELINE_ATS_MAE = 0.04849   # 0-1 scale (4.85 on 0-100), carried from train_r3.py
R2_BASELINE_DOM_ACC = 0.8500

R4_LR             = 5e-6
R4_MAX_EPOCHS     = 30
R4_BATCH_SIZE     = 32
R4_EARLY_PATIENCE = 8           # epochs of no val-MAE improvement before early stop
R4_MAE_HARD_STOP  = 6.5 / 100.0  # 0.065 on 0-1 scale → 6.5 on 0-100

# Canonical loss weights — MUST NOT carry forward R-2 values (1.0/0.5)
ATS_W = SCORE_LOSS_WEIGHT    # 0.35
DOM_W = DOMAIN_LOSS_WEIGHT   # 0.35
RSG_W = RSG_LOSS_WEIGHT      # 0.30

assert (round(ATS_W, 4), round(DOM_W, 4), round(RSG_W, 4)) == (0.35, 0.35, 0.30), (
    f"FATAL: R-4 must use canonical weights (0.35/0.35/0.30), got {ATS_W}/{DOM_W}/{RSG_W}"
)

DOMAIN_NAMES = ["IT", "Non-IT", "Design", "Healthcare", "Finance", "Legal", "Edu"]

FRESHER_KEYWORDS = [
    "fresher", "entry level", "entry-level", "0 years experience",
    "0-1 years", "recent graduate", "final year", "fresh graduate",
    "no prior experience", "no experience",
]


# ── Dataset helpers ──────────────────────────────────────────────────────────

def _make_ats_raw_ds(idxs, r_texts, jd_texts, ats_scores, domain_labels, shuffle=False):
    """5-tuple dataset for the custom training loop: (r, jd, ats_score, dom, dom_weight)."""
    class_wts = np.array(
        [DOMAIN_CLASS_WEIGHTS.get(int(d), 1.0) for d in domain_labels[idxs]],
        dtype="float32",
    )
    ds = tf.data.Dataset.from_tensor_slices((
        r_texts[idxs],
        jd_texts[idxs],
        ats_scores[idxs].astype("float32"),
        domain_labels[idxs].astype("int32"),
        class_wts,
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=42, reshuffle_each_iteration=True)
    return ds.batch(R4_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _make_rsg_raw_ds(idxs, profiles, rsg_labels, shuffle=False):
    """2-tuple RSG dataset for the custom training loop: (profile, rsg_label)."""
    ds = tf.data.Dataset.from_tensor_slices((
        profiles[idxs],
        rsg_labels[idxs].astype("int32"),
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=42, reshuffle_each_iteration=True)
    return ds.batch(R4_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _make_ats_eval_ds(idxs, r_texts, jd_texts, ats_scores, domain_labels):
    """Dict-format ATS eval dataset compatible with model.predict()."""
    ds = tf.data.Dataset.from_tensor_slices((
        r_texts[idxs],
        jd_texts[idxs],
        ats_scores[idxs].astype("float32"),
        domain_labels[idxs].astype("int32"),
    ))
    ds = ds.batch(R4_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.map(lambda r, j, a, d: (
        {"resume_text": r, "jd_text": j},
        {"ats_score": tf.expand_dims(a, 1), "domain_probs": d},
    ))


def _make_rsg_eval_ds(idxs, profiles, rsg_labels):
    """Dict-format RSG eval dataset. Profile fed to both inputs per data contract."""
    ds = tf.data.Dataset.from_tensor_slices((
        profiles[idxs],
        rsg_labels[idxs].astype("int32"),
    ))
    ds = ds.batch(R4_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.map(lambda p, lbl: (
        {"resume_text": p, "jd_text": p},
        {"rsg_template": lbl},
    ))


# ── Fresher fairness ─────────────────────────────────────────────────────────

def _fresher_fairness(model, r_texts, jd_texts, test_idx):
    """
    Compute mean predicted ATS score for fresher vs experienced samples in the
    test set. Returns (gap, n_fresher, n_experienced) where gap is on 0-100 scale.
    Returns (None, n, m) if fewer than 10 fresher samples are found.
    """
    is_fresher = np.array([
        any(kw in r_texts[i].lower() for kw in FRESHER_KEYWORDS)
        for i in test_idx
    ])
    n_fresher = int(is_fresher.sum())
    if n_fresher < 10:
        return None, n_fresher, len(test_idx) - n_fresher

    all_scores = []
    r_sub, jd_sub = r_texts[test_idx], jd_texts[test_idx]
    for start in range(0, len(test_idx), 64):
        ats_out, _, _ = model(
            [tf.constant(r_sub[start:start + 64]),
             tf.constant(jd_sub[start:start + 64])],
            training=False,
        )
        all_scores.append(ats_out.numpy().flatten())

    scores = np.concatenate(all_scores) * 100.0   # scale to 0-100
    gap = float(scores[~is_fresher].mean()) - float(scores[is_fresher].mean())
    return gap, n_fresher, int((~is_fresher).sum())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stage R-4: Joint Fine-Tuning")
    print("=" * 60)
    print(f"\n  Loss weights -- ATS:{ATS_W}  DOM:{DOM_W}  RSG:{RSG_W}  (canonical)")

    # ── 1. Build model, load R-3 weights ──────────────────────────────────────
    print("\n[1/8] Building model and loading R-3 weights...")
    model = build_unified_model()
    r3_path = "model/unified_model/r3_rsg_warmup_best.weights.h5"
    model.load_weights(r3_path)
    print(f"  Loaded: {r3_path}")

    # ── 2. Freeze encoder; unfreeze ALL head layers ────────────────────────────
    print("\n[2/8] Applying R-4 freeze strategy...")
    model.get_layer("mobile_use_encoder").trainable = False
    for layer in model.layers:
        if layer.name != "mobile_use_encoder":
            layer.trainable = True

    trainable = [l.name for l in model.layers if l.trainable]
    assert not model.get_layer("mobile_use_encoder").trainable, \
        "FATAL: MobileUSE encoder must remain frozen in R-4."
    print(f"  Frozen:    [mobile_use_encoder]")
    print(f"  Trainable: {trainable}")

    # ── 3. Optimizer and loss functions ───────────────────────────────────────
    print("\n[3/8] Setting up optimizer and losses...")
    optimizer  = tf.keras.optimizers.Adam(learning_rate=R4_LR)
    ats_mae_fn = tf.keras.losses.MeanAbsoluteError()
    dom_ce_fn  = tf.keras.losses.SparseCategoricalCrossentropy()
    rsg_ce_fn  = tf.keras.losses.SparseCategoricalCrossentropy()
    print(f"  Adam(lr={R4_LR}) | ATS:MAE  DOM:SparseCCE  RSG:SparseCCE")

    # ── 4. Load data and splits ───────────────────────────────────────────────
    print("\n[4/8] Loading data and splits...")
    r_texts, jd_texts, ats_scores, domain_labels = load_ats_data(
        str(LABELED_DIR / "merged_final.csv")
    )
    print(f"  ATS pairs: {len(r_texts):,}")

    profiles, tids = load_rsg_data(str(RSG_CSV_PATH))
    with open(RSG_MAPPING_JSON) as fh:
        mapping = json.load(fh)
    id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}
    valid_mask     = np.array([int(t) in id_to_idx for t in tids])
    profiles_valid = profiles[valid_mask]
    rsg_labels     = np.array([id_to_idx[int(t)] for t in tids[valid_mask]])
    print(f"  RSG valid samples: {len(profiles_valid)}")

    with open("model/unified_model/data_splits.json") as fh:
        splits = json.load(fh)
    ats_tr  = np.array(splits["ats_train"])
    ats_val = np.array(splits["ats_val"])
    ats_tst = np.array(splits["ats_test"])
    rsg_tr  = np.array(splits["rsg_train"])
    rsg_val = np.array(splits["rsg_val"])

    ats_train_ds = _make_ats_raw_ds(
        ats_tr, r_texts, jd_texts, ats_scores, domain_labels, shuffle=True
    )
    rsg_train_ds = _make_rsg_raw_ds(rsg_tr, profiles_valid, rsg_labels, shuffle=True)
    ats_val_ds   = _make_ats_eval_ds(ats_val, r_texts, jd_texts, ats_scores, domain_labels)
    rsg_val_ds   = _make_rsg_eval_ds(rsg_val, profiles_valid, rsg_labels)
    ats_test_ds  = _make_ats_eval_ds(ats_tst, r_texts, jd_texts, ats_scores, domain_labels)

    # ── 5. Initial validation from R-3 weights (before any R-4 gradient) ──────
    print("\n[5/8] Initial validation (R-3 baseline before any R-4 training)...")
    init_ats_pred, _, _ = model.predict(ats_val_ds, verbose=0)
    init_mae_100 = mean_absolute_error(
        ats_scores[ats_val], init_ats_pred.flatten()
    ) * 100.0
    print(f"  R-3 val ATS MAE: {init_mae_100:.2f}")

    best_val_mae = init_mae_100 / 100.0
    best_ckpt    = "model/unified_model/r4_best_in_progress.weights.h5"
    model.save_weights(best_ckpt)

    # ── 6. Custom alternating training loop ───────────────────────────────────
    print(f"\n[6/8] Custom alternating training loop (max {R4_MAX_EPOCHS} epochs)...")
    print(f"  Hard-stop gate: val MAE > {R4_MAE_HARD_STOP * 100:.1f}")
    print(f"  Early-stop patience: {R4_EARLY_PATIENCE} epochs\n")

    log_rows      = []
    epochs_no_imp = 0
    hard_stop     = False

    for epoch in range(1, R4_MAX_EPOCHS + 1):
        rsg_iter = iter(rsg_train_ds.repeat())  # infinite RSG iterator
        ats_ls, dom_ls, rsg_ls = [], [], []

        for r_b, jd_b, ats_t, dom_t, dom_w in ats_train_ds:

            # ── ATS + Domain batch ──────────────────────────────────────────
            with tf.GradientTape() as tape:
                ats_out, dom_out, _ = model([r_b, jd_b], training=True)
                ats_loss = ats_mae_fn(tf.expand_dims(ats_t, 1), ats_out)
                dom_loss = dom_ce_fn(dom_t, dom_out, sample_weight=dom_w)
                loss_ad  = ATS_W * ats_loss + DOM_W * dom_loss
            optimizer.apply_gradients(
                zip(tape.gradient(loss_ad, model.trainable_variables),
                    model.trainable_variables)
            )
            ats_ls.append(float(ats_loss))
            dom_ls.append(float(dom_loss))

            # ── RSG batch (interleaved) ─────────────────────────────────────
            prof_b, rsg_t = next(rsg_iter)
            with tf.GradientTape() as tape:
                _, _, rsg_out = model([prof_b, prof_b], training=True)
                rsg_loss = RSG_W * rsg_ce_fn(rsg_t, rsg_out)
            optimizer.apply_gradients(
                zip(tape.gradient(rsg_loss, model.trainable_variables),
                    model.trainable_variables)
            )
            rsg_ls.append(float(rsg_loss))

        # ── Epoch-end validation ──────────────────────────────────────────────
        ats_pred, dom_pred, _ = model.predict(ats_val_ds, verbose=0)
        _, _, rsg_pred_val    = model.predict(rsg_val_ds, verbose=0)

        val_mae     = mean_absolute_error(ats_scores[ats_val], ats_pred.flatten())
        val_mae_100 = val_mae * 100.0
        val_dom_acc = float(np.mean(np.argmax(dom_pred, 1) == domain_labels[ats_val]))
        val_rsg_acc = float(np.mean(np.argmax(rsg_pred_val, 1) == rsg_labels[rsg_val]))

        print(
            f"  Ep {epoch:02d}/{R4_MAX_EPOCHS} | "
            f"train ATS MAE:{np.mean(ats_ls):.4f}  "
            f"val ATS MAE:{val_mae_100:.2f} | "
            f"val Dom:{val_dom_acc * 100:.1f}%  "
            f"val RSG:{val_rsg_acc * 100:.1f}%"
        )

        log_rows.append({
            "epoch": epoch,
            "train_ats_mae": float(np.mean(ats_ls)),
            "train_dom_loss": float(np.mean(dom_ls)),
            "train_rsg_loss": float(np.mean(rsg_ls)),
            "val_ats_mae_100": val_mae_100,
            "val_dom_acc": val_dom_acc,
            "val_rsg_acc": val_rsg_acc,
        })

        # ── Regression hard stop ──────────────────────────────────────────────
        if val_mae > R4_MAE_HARD_STOP:
            print(
                f"\n  *** HARD STOP at epoch {epoch}: "
                f"val MAE {val_mae_100:.2f} > {R4_MAE_HARD_STOP * 100:.1f}. "
                "Restoring best checkpoint. ***"
            )
            model.load_weights(best_ckpt)
            hard_stop = True
            break

        # ── Best checkpoint ───────────────────────────────────────────────────
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            model.save_weights(best_ckpt)
            print(f"    -> New best val MAE: {val_mae_100:.2f} -- checkpoint saved.")
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            print(f"    -> No improvement ({epochs_no_imp}/{R4_EARLY_PATIENCE})")
            if epochs_no_imp >= R4_EARLY_PATIENCE:
                print(
                    f"\n  Early stopping: {R4_EARLY_PATIENCE} epochs without improvement."
                )
                model.load_weights(best_ckpt)
                break

    # ── Write CSV log ──────────────────────────────────────────────────────────
    if log_rows:
        log_path = "model/unified_model/r4_training_log.csv"
        with open(log_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\n  Training log saved: {log_path}")

    # ── 7. Save final best weights ─────────────────────────────────────────────
    print("\n[7/8] Saving r4_joint_best.weights.h5...")
    model.load_weights(best_ckpt)  # ensure best weights are active
    final_path = "model/unified_model/r4_joint_best.weights.h5"
    model.save_weights(final_path)
    print(f"  Saved: {final_path}")

    # ── Post-training encoder freeze assertion ─────────────────────────────────
    assert not model.get_layer("mobile_use_encoder").trainable, \
        "FATAL: MobileUSE encoder is trainable after R-4 -- weights may be corrupted."
    print("  [Check] Encoder frozen post-training: PASS")

    # ── 8. Final test set evaluation ───────────────────────────────────────────
    print("\n[8/8] Final evaluation on held-out test set...")
    ats_test_pred, dom_test_pred, _ = model.predict(ats_test_ds, verbose=0)
    ats_test_true = ats_scores[ats_tst]
    dom_test_true = domain_labels[ats_tst]

    test_mae_100 = mean_absolute_error(
        ats_test_true, ats_test_pred.flatten()
    ) * 100.0
    test_dom_f1  = f1_score(
        dom_test_true, np.argmax(dom_test_pred, 1), average="macro"
    )
    per_dom_f1   = f1_score(
        dom_test_true, np.argmax(dom_test_pred, 1),
        average=None, labels=list(range(7)),
    )

    _, _, rsg_val_out = model.predict(rsg_val_ds, verbose=0)
    rsg_final_acc = float(
        np.mean(np.argmax(rsg_val_out, 1) == rsg_labels[rsg_val])
    )

    # ── Fresher fairness evaluation ────────────────────────────────────────────
    print("\n  Evaluating fresher fairness gap...")
    gap, n_fresh, n_exp = _fresher_fairness(model, r_texts, jd_texts, ats_tst)

    # ── Print gates ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE R-4 -- FINAL TEST METRICS")
    print("=" * 60)
    print(f"  ATS MAE (0-100):    {test_mae_100:.2f}  "
          f"{'PASS' if test_mae_100 < 8.0 else 'FAIL'} (gate < 8.0)")
    print(f"  R-4 tight gate:     {test_mae_100:.2f}  "
          f"{'PASS' if test_mae_100 < 6.5 else 'WARN'} (ideal < 6.5)")
    print(f"  Domain F1 (macro):  {test_dom_f1:.4f}  "
          f"{'PASS' if test_dom_f1 > 0.85 else 'FAIL'} (gate > 0.85)")
    print(f"  RSG val accuracy:   {rsg_final_acc:.4f}  "
          f"{'PASS' if rsg_final_acc >= 0.50 else 'FAIL'} (gate >= 0.50)")
    print()
    print("  Per-domain F1 (gate > 0.80 each):")
    for i, (name, f1_val) in enumerate(zip(DOMAIN_NAMES, per_dom_f1)):
        flag = "PASS" if f1_val > 0.80 else "FAIL"
        print(f"    [{i}] {name:12s}: {f1_val:.4f}  {flag}")
    print()
    if gap is not None:
        gate_str = "PASS" if gap <= 20.0 else "FAIL"
        print(f"  Fresher gap:        {gap:.1f} pts  {gate_str} (gate <= 20 pts)")
        print(f"    Fresher samples: {n_fresh}  |  Experienced: {n_exp}")
    else:
        print(f"  Fresher gap:        N/A (< 10 fresher samples in test set)")
    print()
    print(f"  Hard stop triggered: {'YES' if hard_stop else 'NO'}")
    print(f"  Best val MAE:        {best_val_mae * 100:.2f}")
    print(f"  Output:              {final_path}")
    print("=" * 60)
    print()
    print("  Stage R-4 Complete.")
    print("  DO NOT proceed to T-1 conversion until all gates above are PASS.")


if __name__ == "__main__":
    main()
