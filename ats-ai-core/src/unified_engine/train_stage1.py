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
from src.config import ATS_MODEL_DIR, RSG_CSV_PATH, RSG_MAPPING_JSON, UNIFIED_MODEL_DIR

# ── Paths ──────────────────────────────────────────────────────────────
RSG_CSV        = RSG_CSV_PATH
MAPPING_JSON   = RSG_MAPPING_JSON
RSG_CHECKPOINT = UNIFIED_MODEL_DIR / "unified_with_rsg_weights.h5"
ATS_WEIGHTS    = ATS_MODEL_DIR / "final_model_weights.h5"
OUTPUT_DIR     = UNIFIED_MODEL_DIR
# ───────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load label mapping ─────────────────────────────────────────────────
with open(MAPPING_JSON) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}

print("=== STAGE 1: RSG HEAD WARMUP ===\n")

# ── Build model and load RSG checkpoint ────────────────────────────────
print("Building unified model...")
model = build_unified_model()

print(f"Loading RSG checkpoint: {RSG_CHECKPOINT}")
model.load_weights(str(RSG_CHECKPOINT))
print("RSG checkpoint loaded — RSG head has pre-trained weights.")

# ── Load ATS production weights on top ────────────────────────────────
# Transfer ATS and Domain head weights by layer name matching
print(f"\nLoading ATS production weights: {ATS_WEIGHTS}")
layer_name_map = {
    "sim_dense1": "ats_dense1",
    "sim_drop1": "ats_drop1",
    "sim_dense2": "ats_dense2",
    "sim_drop2": "ats_drop2",
    "ats_score": "ats_score",
    "dom_dense1": "dom_dense1",
    "dom_drop1": "dom_drop1",
    "dom_dense2": "dom_dense2",
    "dom_drop2": "dom_drop2",
    "domain_logits": "domain_probs",
}

if ATS_WEIGHTS.exists():
    ats_source = tf.keras.models.load_model(str(ATS_WEIGHTS), compile=False)
    transferred = 0
    for ats_name, unified_name in layer_name_map.items():
        try:
            ats_layer = ats_source.get_layer(ats_name)
            unified_layer = model.get_layer(unified_name)
            if ats_layer.get_weights() and unified_layer.get_weights():
                src_shapes = [w.shape for w in ats_layer.get_weights()]
                dst_shapes = [w.shape for w in unified_layer.get_weights()]
                if src_shapes == dst_shapes:
                    unified_layer.set_weights(ats_layer.get_weights())
                    transferred += 1
                    print(f"  OK   {ats_name:<16} → {unified_name}")
                else:
                    print(f"  SKIP {ats_name}: shape mismatch {src_shapes} → {dst_shapes}")
        except (ValueError, KeyError) as e:
            print(f"  SKIP {ats_name}: {e}")
    print(f"ATS/Domain weights transferred: {transferred} layers")
else:
    print("WARNING: ATS weights not found — ATS head starts from random weights")

# ── Freeze everything EXCEPT RSG head ──────────────────────────────────
for layer in model.layers:
    layer.trainable = False

# RSG head layer names (confirmed from model inspection)
rsg_layer_names = [
    "rsg_dense1", "rsg_bn1", "rsg_drop1",
    "rsg_dense2", "rsg_bn2", "rsg_drop2",
    "rsg_dense3", "rsg_bn3", "rsg_drop3", "rsg_template"
]
for name in rsg_layer_names:
    try:
        model.get_layer(name).trainable = True
    except ValueError:
        print(f"WARNING: Layer {name} not found — check unified_model.py")

trainable = [l.name for l in model.layers if l.trainable]
frozen    = [l.name for l in model.layers if not l.trainable]
print(f"\nTrainable ({len(trainable)}): {trainable}")
encoder_name = "mobile_use_encoder"
print(f"Frozen    ({len(frozen)}): {[l.name for l in model.layers if not l.trainable and l.name != encoder_name][:5]}... {encoder_name}")

# ── Compile — RSG loss only ────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        "ats_score":    None,
        "domain_probs": None,
        "rsg_template": "sparse_categorical_crossentropy"
    },
    metrics={"rsg_template": ["accuracy"]}
)

# ── Load RSG data and remap labels to model indices ────────────────────
print("\nLoading RSG training data...")
profile_texts, template_ids = load_rsg_data(str(RSG_CSV))

# Remap original template IDs → model output indices (0-45)
template_indices = np.array([
    id_to_idx[int(tid)] for tid in template_ids
    if int(tid) in id_to_idx
])
profile_texts_filtered = np.array([
    pt for pt, tid in zip(profile_texts, template_ids)
    if int(tid) in id_to_idx
])

split   = int(0.8 * len(profile_texts_filtered))
train_x = (profile_texts_filtered[:split], profile_texts_filtered[:split])
val_x   = (profile_texts_filtered[split:],  profile_texts_filtered[split:])
train_y = {"rsg_template": template_indices[:split]}
val_y   = {"rsg_template": template_indices[split:]}

print(f"RSG samples — train: {split}  |  val: {len(profile_texts_filtered)-split}")

# ── Train Stage 1 ──────────────────────────────────────────────────────
print("\nTraining RSG head (10 epochs)...")
history = model.fit(
    x={"resume_text": train_x[0], "jd_text": train_x[1]},
    y=train_y,
    validation_data=(
        {"resume_text": val_x[0], "jd_text": val_x[1]},
        val_y
    ),
    epochs=10,
    batch_size=32,
    verbose=1
)

final_val_acc = history.history.get("val_rsg_template_accuracy", [0])[-1]

# ── Save Stage 1 checkpoint ────────────────────────────────────────────
ckpt = OUTPUT_DIR / "stage1_checkpoint.weights.h5"
model.save_weights(str(ckpt))
print(f"\nStage 1 checkpoint saved: {ckpt}")

# ── Result ─────────────────────────────────────────────────────────────
print(f"\n=== STAGE 1 RESULT ===")
print(f"RSG val_accuracy (final epoch): {final_val_acc:.4f}")
if final_val_acc >= 0.50:
    print("STATUS: PASS — proceed to Stage 2")
elif final_val_acc >= 0.30:
    print("STATUS: BORDERLINE — show to Sai before Stage 2")
else:
    print("STATUS: FAIL — val_accuracy below 0.30 — STOP, report to Sai")
print("\nSend this output to Sai before running INJECTION-2-STAGE2.")
