"""
Stage R-3: RSG Head Warmup
- Load R-2 weights
- Freeze ALL layers, then unfreeze only rsg_* layers
- Train RSG head in isolation
- Verify ATS/Domain drift is within tolerance
"""
import json
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ats-ai-core')))

from src.unified_engine.data_loader import load_ats_data, load_rsg_data
from src.unified_engine.unified_model import build_unified_model
from src.config import LABELED_DIR, RSG_CSV_PATH, RSG_MAPPING_JSON

# ── R-2 Baseline metrics (from best val_loss epoch 33) ──
R2_BASELINE_VAL_ATS_MAE = 0.04849  # on 0-1 scale => 4.85 on 0-100
R2_BASELINE_VAL_DOM_ACC = 0.8500

def make_rsg_dataset(idxs, profile_texts, rsg_labels, shuffle=False):
    """RSG data contract: profile_text goes to BOTH resume_text and jd_text inputs."""
    ds = tf.data.Dataset.from_tensor_slices((
        profile_texts[idxs],
        rsg_labels[idxs].astype("int32")
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda p, lbl: (
        {"resume_text": p, "jd_text": p},  # duplicate profile to both inputs
        {"rsg_template": lbl}
    ))
    return ds

def make_ats_dataset_for_eval(idxs, r_texts, jd_texts, ats_scores, domain_labels):
    """ATS dataset for drift evaluation (no sample weights needed, inference only)."""
    ds = tf.data.Dataset.from_tensor_slices((
        r_texts[idxs], jd_texts[idxs],
        ats_scores[idxs].astype("float32"),
        domain_labels[idxs].astype("int32")
    ))
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda r, j, a, d: (
        {"resume_text": r, "jd_text": j},
        {"ats_score": tf.expand_dims(a, 1), "domain_probs": d}
    ))
    return ds

def main():
    print("=" * 60)
    print("  Stage R-3: RSG Head Warmup")
    print("=" * 60)

    # ── 1. Build model and load R-2 weights ──
    print("\n[1/6] Building model and loading R-2 weights...")
    model = build_unified_model()
    model.load_weights("model/unified_model/r2_ats_domain_best.weights.h5")
    print("  R-2 weights loaded successfully.")

    # ── 2. Freeze strategy: freeze ALL, then unfreeze rsg_* only ──
    print("\n[2/6] Applying freeze strategy...")
    for layer in model.layers:
        layer.trainable = False

    rsg_layer_names = []
    for layer in model.layers:
        if layer.name.startswith("rsg_"):
            layer.trainable = True
            rsg_layer_names.append(layer.name)

    print(f"  Trainable (RSG only): {rsg_layer_names}")
    frozen = [l.name for l in model.layers if not l.trainable]
    print(f"  Frozen layers: {len(frozen)} total")

    # ── 3. Compile with RSG-only loss ──
    print("\n[3/6] Compiling model (RSG-only loss)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss={
            "ats_score": None,
            "domain_probs": None,
            "rsg_template": "sparse_categorical_crossentropy",
        },
        metrics={"rsg_template": ["accuracy"]}
    )

    # ── 4. Load RSG data and splits ──
    print("\n[4/6] Loading RSG data...")
    profiles, tids = load_rsg_data(str(RSG_CSV_PATH))
    with open(RSG_MAPPING_JSON) as f:
        mapping = json.load(f)
    id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}

    # Map template IDs to contiguous class indices
    valid_mask = np.array([int(t) in id_to_idx for t in tids])
    profiles_valid = profiles[valid_mask]
    rsg_labels = np.array([id_to_idx[int(t)] for t in tids[valid_mask]])
    print(f"  Valid RSG samples: {len(profiles_valid)}")
    print(f"  RSG classes: {len(set(rsg_labels))}")

    with open("model/unified_model/data_splits.json") as f:
        splits = json.load(f)

    rsg_train_idx = np.array(splits["rsg_train"])
    rsg_val_idx = np.array(splits["rsg_val"])

    train_ds = make_rsg_dataset(rsg_train_idx, profiles_valid, rsg_labels, shuffle=True)
    val_ds = make_rsg_dataset(rsg_val_idx, profiles_valid, rsg_labels, shuffle=False)

    # ── 5. Train RSG head ──
    print("\n[5/6] Training RSG head (15 epochs max)...")
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rsg_template_accuracy", patience=5, mode="max",
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "model/unified_model/r3_rsg_warmup_best.weights.h5",
            save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger("model/unified_model/r3_training_log.csv"),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
    )

    # ── 6. Post-training: RSG accuracy + ATS/Domain drift check ──
    print("\n[6/6] Post-training regression guard...")

    # RSG final val accuracy
    rsg_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    rsg_val_acc = rsg_eval.get("rsg_template_accuracy", rsg_eval.get("accuracy", 0))
    print(f"  RSG val accuracy: {rsg_val_acc:.4f}")

    # ATS/Domain drift check using ATS val set
    print("  Loading ATS validation data for drift check...")
    r_texts, jd_texts, ats_scores, domain_labels = load_ats_data(str(LABELED_DIR / "merged_final.csv"))

    ats_val_idx = np.array(splits["ats_val"])
    ats_val_ds = make_ats_dataset_for_eval(ats_val_idx, r_texts, jd_texts, ats_scores, domain_labels)

    ats_pred, dom_pred, _ = model.predict(ats_val_ds, verbose=0)
    ats_true = ats_scores[ats_val_idx]
    dom_true = domain_labels[ats_val_idx]

    current_ats_mae = mean_absolute_error(ats_true, ats_pred.flatten())
    current_dom_acc = np.mean(np.argmax(dom_pred, axis=1) == dom_true)

    ats_mae_drift = abs(current_ats_mae - R2_BASELINE_VAL_ATS_MAE) * 100  # on 0-100 scale
    dom_acc_drift = abs(current_dom_acc - R2_BASELINE_VAL_DOM_ACC) * 100  # percentage points

    print(f"\n  --- DRIFT ANALYSIS ---")
    print(f"  R-2 Baseline ATS val MAE:    {R2_BASELINE_VAL_ATS_MAE * 100:.2f}")
    print(f"  Current ATS val MAE:         {current_ats_mae * 100:.2f}")
    print(f"  ATS MAE drift:               {ats_mae_drift:.2f} (gate: < 0.50)")
    print(f"  R-2 Baseline Domain val acc: {R2_BASELINE_VAL_DOM_ACC * 100:.2f}%")
    print(f"  Current Domain val acc:      {current_dom_acc * 100:.2f}%")
    print(f"  Domain acc drift:            {dom_acc_drift:.2f}pp (gate: < 2.00pp)")

    print(f"\n  --- FINAL RESULTS ---")
    print(f"  RSG val accuracy:  {rsg_val_acc:.4f}  {'PASS' if rsg_val_acc >= 0.50 else 'FAIL'} (gate >= 0.50)")
    print(f"  ATS MAE drift:     {ats_mae_drift:.2f}   {'PASS' if ats_mae_drift < 0.50 else 'FAIL'} (gate < 0.50)")
    print(f"  Domain acc drift:  {dom_acc_drift:.2f}pp  {'PASS' if dom_acc_drift < 2.0 else 'FAIL'} (gate < 2.00pp)")
    print(f"\n  Stage R-3 Complete.")

if __name__ == "__main__":
    main()
