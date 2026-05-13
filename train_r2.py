import json
import os
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, f1_score

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ats-ai-core')))

from src.unified_engine.data_loader import load_ats_data
from src.unified_engine.unified_model import build_unified_model
from src.config import LABELED_DIR

def make_ats_dataset(idxs, r_texts, jd_texts, ats_scores, domain_labels, shuffle=False):
    # Domain class weights mapping
    domain_class_weights = {0: 1.4, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.5, 5: 0.9, 6: 1.0}
    weights = np.array([domain_class_weights.get(int(d), 1.0) for d in domain_labels[idxs]], dtype="float32")
    
    ds = tf.data.Dataset.from_tensor_slices((
        r_texts[idxs], jd_texts[idxs],
        ats_scores[idxs].astype("float32"),
        domain_labels[idxs].astype("int32"),
        weights
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda r, j, a, d, w: (
        {"resume_text": r, "jd_text": j},
        {"ats_score": tf.expand_dims(a, 1), "domain_probs": d},
        {"domain_probs": w}
    ))
    return ds

def main():
    print("=== Stage R-2: ATS + Domain Head Training ===")
    
    # 1. Load data
    print("Loading ATS data...")
    r_texts, jd_texts, ats_scores, domain_labels = load_ats_data(str(LABELED_DIR / "merged_final.csv"))
    
    # 2. Load splits
    splits_path = Path("model/unified_model/data_splits.json")
    with open(splits_path) as f:
        splits = json.load(f)
    
    train_idx = np.array(splits["ats_train"])
    val_idx = np.array(splits["ats_val"])
    test_idx = np.array(splits["ats_test"])
    
    train_ds = make_ats_dataset(train_idx, r_texts, jd_texts, ats_scores, domain_labels, shuffle=True)
    val_ds = make_ats_dataset(val_idx, r_texts, jd_texts, ats_scores, domain_labels, shuffle=False)
    test_ds = make_ats_dataset(test_idx, r_texts, jd_texts, ats_scores, domain_labels, shuffle=False)
    
    # 3. Build model and apply freeze strategy
    model = build_unified_model()
    
    # Freeze encoder and RSG layers
    model.get_layer("mobile_use_encoder").trainable = False
    
    for layer in model.layers:
        if layer.name.startswith("rsg_"):
            layer.trainable = False
        elif layer.name.startswith("ats_") or layer.name.startswith("dom_") or layer.name in ["ats_score", "domain_probs"]:
            layer.trainable = True
            
    # Verify freeze strategy
    trainable_layers = [l.name for l in model.layers if l.trainable]
    print(f"Trainable layers: {trainable_layers}")
    
    # 4. Compile
    R2_ATS_WEIGHT = 1.0
    R2_DOMAIN_WEIGHT = 0.5
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "ats_score": "mean_absolute_error",
            "domain_probs": "sparse_categorical_crossentropy",
            "rsg_template": None,
        },
        loss_weights={"ats_score": R2_ATS_WEIGHT, "domain_probs": R2_DOMAIN_WEIGHT},
        metrics={"ats_score": ["mae"], "domain_probs": ["accuracy"]}
    )
    
    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("model/unified_model/r2_ats_domain_best.weights.h5", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger("model/unified_model/r2_training_log.csv")
    ]
    
    # 6. Train
    domain_class_weights = {
        0: 1.4, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.5, 5: 0.9, 6: 1.0
    }
    
    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=60,
        callbacks=callbacks
    )
    
    # 7. Evaluate on test set
    print("\nEvaluating on test set...")
    ats_pred, dom_pred, _ = model.predict(test_ds, verbose=0)
    
    ats_true = ats_scores[test_idx]
    dom_true = domain_labels[test_idx]
    
    mae = mean_absolute_error(ats_true, ats_pred.flatten()) * 100
    macro_f1 = f1_score(dom_true, np.argmax(dom_pred, axis=1), average="macro")
    
    print(f"--- TEST SET RESULTS ---")
    print(f"ATS MAE: {mae:.2f}")
    print(f"Domain F1 (Macro): {macro_f1:.4f}")
    
    print("Stage R-2 Complete.")

if __name__ == "__main__":
    main()
