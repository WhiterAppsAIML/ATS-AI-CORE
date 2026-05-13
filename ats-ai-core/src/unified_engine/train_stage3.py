import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # MUST be before any TF import

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.unified_engine.unified_model import build_unified_model
from src.unified_engine.data_loader import load_rsg_data
from src.config import RSG_CSV_PATH, RSG_MAPPING_JSON

# ── Paths ──────────────────────────────────────────────────────────────
RSG_CSV      = RSG_CSV_PATH
MAPPING_JSON = RSG_MAPPING_JSON
STAGE2_BEST  = Path(r"model\unified_model\best_unified_weights.h5")
OUTPUT_DIR   = Path(r"model\unified_model")
SAVE_PATH    = OUTPUT_DIR / "unified_stage3_weights.h5"
# ───────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load mapping ───────────────────────────────────────────────────────
with open(MAPPING_JSON) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}

print("=== STAGE 3: RSG HEAD FINE-TUNING ===")
print()
print("Stage 2 results being preserved:")
print("  ATS MAE    : 4.33  (frozen — will not change)")
print("  Domain acc : 86.70% (frozen — will not change)")
print("  RSG acc    : 5.50%  (fixing now with correct training data)")
print()

# ── Build model and load Stage 2 best weights ─────────────────────────
print("Building unified model...")
model = build_unified_model()

print(f"Loading Stage 2 best weights: {STAGE2_BEST}")
model.load_weights(str(STAGE2_BEST))
print("Stage 2 weights loaded — ATS and Domain heads are now initialised.")

# ── Freeze EVERYTHING except RSG head ─────────────────────────────────
# This is critical — ATS and Domain weights must not change at all
for layer in model.layers:
    layer.trainable = False

rsg_layer_names = [
    "rsg_dense1", "rsg_bn1",   "rsg_drop1",
    "rsg_dense2", "rsg_bn2",   "rsg_drop2",
    "rsg_dense3", "rsg_bn3",   "rsg_drop3", "rsg_template"
]
unfrozen = []
for name in rsg_layer_names:
    try:
        model.get_layer(name).trainable = True
        unfrozen.append(name)
    except ValueError:
        print(f"WARNING: Layer {name} not found in model")

print(f"\nUnfrozen RSG layers ({len(unfrozen)}): {unfrozen}")
frozen_count = sum(1 for l in model.layers if not l.trainable)
print(f"Frozen layers: {frozen_count} (includes USE encoder, ATS head, Domain head)")

# ── Compile — RSG loss only ────────────────────────────────────────────
# Use lower LR than Stage 1 (5e-5 instead of 1e-4)
# RSG head already has weights from transfer — needs gentle fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss={
        "ats_score":    None,
        "domain_probs": None,
        "rsg_template": "sparse_categorical_crossentropy"
    },
    metrics={"rsg_template": ["accuracy"]}
)

# ── Load RSG data with CORRECT input pairing ───────────────────────────
# KEY FIX: profile_text is used as BOTH resume_text AND jd_text input.
# This is the correct format — the RSG head encodes the profile and
# predicts which template family best fits that candidate.
# This is exactly how the standalone RSG model was trained.
print("\nLoading RSG training data...")
profile_texts, template_ids = load_rsg_data(str(RSG_CSV))

# Filter to only IDs in the mapping and remap to 0-45 indices
valid_mask = np.array([int(tid) in id_to_idx for tid in template_ids])
profile_texts_f = profile_texts[valid_mask]
template_indices = np.array([
    id_to_idx[int(tid)] for tid in template_ids[valid_mask]
])

print(f"RSG samples: {len(profile_texts_f)} valid (of {len(profile_texts)} total)")
print(f"Label range: {template_indices.min()} to {template_indices.max()} (expected 0-45)")

# 80/20 split — same seed as standalone RSG training
split     = int(0.8 * len(profile_texts_f))
train_txt = profile_texts_f[:split]
val_txt   = profile_texts_f[split:]
train_lbl = template_indices[:split]
val_lbl   = template_indices[split:]

print(f"Train: {len(train_txt)}  |  Val: {len(val_txt)}")

# ── Callbacks ──────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_rsg_template_accuracy",
        patience=5,
        restore_best_weights=True,
        mode="max",          # maximise accuracy
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "stage3_best.weights.h5"),
        monitor="val_rsg_template_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1
    )
]

# ── Train Stage 3 ──────────────────────────────────────────────────────
print("\nStarting Stage 3 RSG fine-tuning (max 30 epochs)...")
print("Expected: RSG accuracy recovers to 65-75%")
print("ATS and Domain weights: completely frozen\n")

history = model.fit(
    x={
        "resume_text": train_txt,   # profile_text → resume input
        "jd_text":     train_txt    # profile_text → jd input (same — RSG context)
    },
    y={"rsg_template": train_lbl},
    validation_data=(
        {
            "resume_text": val_txt,
            "jd_text":     val_txt
        },
        {"rsg_template": val_lbl}
    ),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ── Results ────────────────────────────────────────────────────────────
best_val_acc = max(history.history.get("val_rsg_template_accuracy", [0]))
final_val_acc = history.history.get("val_rsg_template_accuracy", [0])[-1]

print(f"\n=== STAGE 3 RESULT ===")
print(f"Best RSG val_accuracy  : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
print(f"Final RSG val_accuracy : {final_val_acc:.4f}  ({final_val_acc*100:.1f}%)")
print()

# ── Save final weights ─────────────────────────────────────────────────
# Save the model with restored RSG weights
# ATS and Domain weights are UNCHANGED from Stage 2 best checkpoint
model.save_weights(str(SAVE_PATH))
print(f"Saved: {SAVE_PATH}")
print()
print("This file contains:")
print("  ATS head    : Stage 2 best weights (MAE 4.33) — UNCHANGED")
print("  Domain head : Stage 2 best weights (86.70%)   — UNCHANGED")
print(f"  RSG head    : Stage 3 fine-tuned ({best_val_acc*100:.1f}%)")
print()

# ── Pass/Fail assessment ───────────────────────────────────────────────
print("=== STAGE 3 GATE CHECK ===")
if best_val_acc >= 0.65:
    print(f"RSG accuracy {best_val_acc*100:.1f}% >= 65% — PASS")
    print("Proceed to INJECTION-2-EVAL")
elif best_val_acc >= 0.50:
    print(f"RSG accuracy {best_val_acc*100:.1f}% >= 50% — ACCEPTABLE")
    print("Show to Sai. May proceed to INJECTION-2-EVAL with awareness.")
else:
    print(f"RSG accuracy {best_val_acc*100:.1f}% < 50% — LOW")
    print("Show to Sai before proceeding. Do NOT run INJECTION-2-EVAL yet.")

print()
print("Send this full output to Sai.")
