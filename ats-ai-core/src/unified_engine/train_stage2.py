import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # MUST be before any TF import

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.unified_engine.unified_model import build_unified_model
from src.unified_engine.data_loader import load_ats_data, load_rsg_data
from src.config import (
    LABELED_DIR,
    RSG_CSV_PATH,
    RSG_MAPPING_JSON,
    UNIFIED_MODEL_DIR as CONFIG_UNIFIED_MODEL_DIR,
    SCORE_LOSS_WEIGHT,
    DOMAIN_LOSS_WEIGHT,
    RSG_LOSS_WEIGHT,
)

# ── Absolute project root — works regardless of working directory ─────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED_MODEL_DIR = str(CONFIG_UNIFIED_MODEL_DIR)
os.makedirs(UNIFIED_MODEL_DIR, exist_ok=True)

# ── Paths — update ATS_CSV before running ─────────────────────────────
ATS_CSV      = str(LABELED_DIR / "merged_final.csv")
RSG_CSV      = str(RSG_CSV_PATH)
MAPPING_JSON = str(RSG_MAPPING_JSON)
STAGE1_CKP   = os.path.join(UNIFIED_MODEL_DIR, "stage1_checkpoint.weights.h5")
# ───────────────────────────────────────────────────────────────────────

# ── Load label mapping ─────────────────────────────────────────────────
with open(MAPPING_JSON) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}

print("=== STAGE 2: JOINT TRAINING (ALTERNATING BATCHES) ===")
print()
print("Training approach: alternating ATS batches + RSG batches per epoch.")
print("Each head only trains on semantically matched data.")
print("No label mismatch possible.")
print()

# ── Build model and load Stage 1 checkpoint ───────────────────────────
model = build_unified_model()
print(f"Loading Stage 1 checkpoint: {STAGE1_CKP}")
model.load_weights(STAGE1_CKP)
print("Checkpoint loaded.\n")

# ── Phase 1: Freeze ATS head, train RSG head only ──────────────────────
# This prevents disturbing pretrained ATS weights while RSG head warms up
for layer in model.layers:
    layer.trainable = False
model.get_layer("mobile_use_encoder").trainable = False

# Unfreeze only RSG-related layers for warmup phase
rsg_layers = [
    "rsg_dense1", "rsg_bn1", "rsg_drop1",
    "rsg_dense2", "rsg_bn2", "rsg_drop2",
    "rsg_dense3", "rsg_bn3", "rsg_drop3", "rsg_template",
]
for name in rsg_layers:
    try:
        model.get_layer(name).trainable = True
    except ValueError:
        print(f"WARNING: Layer '{name}' not found")

trainable = [l.name for l in model.layers if l.trainable]
print(f"Phase 1 trainable layers (RSG warmup): {trainable}")
print(f"USE encoder: frozen")
print(f"ATS/Domain heads: frozen (will unfreeze after warmup)\n")

# ── Optimiser and loss functions ──────────────────────────────────────
# Reduced LR: 1e-5 instead of 1e-4 to avoid disturbing pretrained weights
WARMUP_LR = 1e-5
FINETUNE_LR = 5e-6
WARMUP_EPOCHS = 10

optimizer = tf.keras.optimizers.Adam(learning_rate=WARMUP_LR)
print(f"Learning rates: warmup={WARMUP_LR}, fine-tune={FINETUNE_LR}")
ats_mae_fn   = tf.keras.losses.MeanAbsoluteError()
domain_ce_fn = tf.keras.losses.SparseCategoricalCrossentropy()
rsg_ce_fn    = tf.keras.losses.SparseCategoricalCrossentropy()

ATS_W    = SCORE_LOSS_WEIGHT
DOMAIN_W = DOMAIN_LOSS_WEIGHT
RSG_W    = RSG_LOSS_WEIGHT

# ── Load and prepare ATS data ──────────────────────────────────────────
print("Loading ATS data...")
r_texts, jd_texts, ats_scores, domain_labels = load_ats_data(str(ATS_CSV))
ats_n = len(r_texts)

ats_idx = np.arange(ats_n)
tr_ats, val_ats = train_test_split(ats_idx, test_size=0.2, random_state=42)

print(f"ATS — train: {len(tr_ats)}  val: {len(val_ats)}")

# tf.data for ATS batches
BATCH = 32
ats_train_ds = tf.data.Dataset.from_tensor_slices((
    r_texts[tr_ats], jd_texts[tr_ats],
    ats_scores[tr_ats].astype("float32"),
    domain_labels[tr_ats].astype("int32")
)).shuffle(10000, seed=42).batch(BATCH).prefetch(tf.data.AUTOTUNE)

ats_val_ds = tf.data.Dataset.from_tensor_slices((
    r_texts[val_ats], jd_texts[val_ats],
    ats_scores[val_ats].astype("float32"),
    domain_labels[val_ats].astype("int32")
)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ── Load and prepare RSG data ──────────────────────────────────────────
print("Loading RSG data...")
profile_texts, template_ids = load_rsg_data(str(RSG_CSV))

# Filter to IDs in mapping and remap to 0-45
valid = np.array([int(tid) in id_to_idx for tid in template_ids])
prof_f  = profile_texts[valid]
tmpl_f  = np.array([id_to_idx[int(tid)] for tid in template_ids[valid]])

split_r = int(0.8 * len(prof_f))
rsg_train_ds = tf.data.Dataset.from_tensor_slices((
    prof_f[:split_r], tmpl_f[:split_r].astype("int32")
)).shuffle(5000, seed=42).batch(BATCH).repeat().prefetch(tf.data.AUTOTUNE)
# .repeat() so RSG iterator never runs out during the longer ATS epoch

rsg_val_ds = tf.data.Dataset.from_tensor_slices((
    prof_f[split_r:], tmpl_f[split_r:].astype("int32")
)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

print(f"RSG — train: {split_r}  val: {len(prof_f)-split_r}")
print()

# ── Training step functions ────────────────────────────────────────────
# Note: no @tf.function — optimizer changes between phases
def train_ats_step(r, jd, ats_true, dom_true):
    """One ATS+Domain batch — RSG output is ignored."""
    with tf.GradientTape() as tape:
        ats_out, dom_out, _ = model([r, jd], training=True)
        ats_loss = ats_mae_fn(tf.expand_dims(ats_true, 1), ats_out)
        dom_loss = domain_ce_fn(dom_true, dom_out)
        total    = ATS_W * ats_loss + DOMAIN_W * dom_loss
    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return ats_loss, dom_loss

# Note: no @tf.function — optimizer changes between phases
def train_rsg_step(prof, tmpl_true):
    """One RSG batch — profile_text as both inputs. ATS/Domain ignored."""
    with tf.GradientTape() as tape:
        _, _, rsg_out = model([prof, prof], training=True)
        rsg_loss = rsg_ce_fn(tmpl_true, rsg_out)
        total    = RSG_W * rsg_loss
    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return rsg_loss

@tf.function
def val_ats_step(r, jd, ats_true, dom_true):
    ats_out, dom_out, _ = model([r, jd], training=False)
    ats_loss = ats_mae_fn(tf.expand_dims(ats_true, 1), ats_out)
    dom_loss = domain_ce_fn(dom_true, dom_out)
    dom_pred = tf.argmax(dom_out, axis=1, output_type=tf.int32)
    dom_acc  = tf.reduce_mean(tf.cast(dom_pred == dom_true, tf.float32))
    return ats_loss, dom_loss, dom_acc

@tf.function
def val_rsg_step(prof, tmpl_true):
    _, _, rsg_out = model([prof, prof], training=False)
    rsg_loss = rsg_ce_fn(tmpl_true, rsg_out)
    rsg_pred = tf.argmax(rsg_out, axis=1, output_type=tf.int32)
    rsg_acc  = tf.reduce_mean(tf.cast(rsg_pred == tmpl_true, tf.float32))
    return rsg_loss, rsg_acc

# ── Training loop ──────────────────────────────────────────────────────
MAX_EPOCHS      = 60
PATIENCE        = 12  # Increased patience for 2-phase training
best_val_loss   = float("inf")
patience_count  = 0
best_epoch      = 0
log             = []
rsg_iter        = iter(rsg_train_ds)   # infinite iterator

ckpt_path  = os.path.join(UNIFIED_MODEL_DIR, "best_unified_weights.h5")
csv_path   = os.path.join(UNIFIED_MODEL_DIR, "unified_training_log.csv")

print(f"Starting 2-phase joint training:")
print(f"  Phase 1 (epochs 1-{WARMUP_EPOCHS}): RSG warmup only, LR={WARMUP_LR}")
print(f"  Phase 2 (epochs {WARMUP_EPOCHS+1}-{MAX_EPOCHS}): All heads, LR={FINETUNE_LR}")
print(f"  Early stopping patience={PATIENCE}")
print(f"ATS batches/epoch: {len(ats_train_ds)}")
print()

for epoch in range(1, MAX_EPOCHS + 1):

    # ── Check if we need to switch to Phase 2 ─────────────────────────
    if epoch == WARMUP_EPOCHS + 1:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Unfreezing all heads, reducing LR to {FINETUNE_LR}")
        print(f"{'='*60}\n")
        
        # Unfreeze ATS and Domain heads
        for layer in model.layers:
            if layer.name != "mobile_use_encoder":
                layer.trainable = True
        
        # Create a fresh optimizer so it builds slots for ALL trainable vars
        # (the old optimizer only has slots for RSG layers from Phase 1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
        optimizer.build(model.trainable_variables)
        
        trainable = [l.name for l in model.layers if l.trainable]
        print(f"Now trainable: {trainable}\n")

    # ── Training ───────────────────────────────────────────────────────
    ats_losses, dom_losses = [], []
    
    # Phase 1: RSG warmup only (skip ATS training)
    if epoch <= WARMUP_EPOCHS:
        # Just iterate through ATS data to keep epoch pacing, but don't train
        for r, jd, ats_t, dom_t in ats_train_ds:
            # Record ATS metrics without training (forward pass only)
            ats_out, dom_out, _ = model([r, jd], training=False)
            al = ats_mae_fn(tf.expand_dims(ats_t, 1), ats_out)
            dl = domain_ce_fn(dom_t, dom_out)
            ats_losses.append(float(al))
            dom_losses.append(float(dl))
            
            # Train RSG only
            try:
                prof, tmpl = next(rsg_iter)
            except StopIteration:
                rsg_iter = iter(rsg_train_ds)
                prof, tmpl = next(rsg_iter)
            train_rsg_step(prof, tmpl)
    else:
        # Phase 2: Train all heads
        for r, jd, ats_t, dom_t in ats_train_ds:
            al, dl = train_ats_step(r, jd, ats_t, dom_t)
            ats_losses.append(float(al))
            dom_losses.append(float(dl))

            # One RSG batch per ATS batch
            try:
                prof, tmpl = next(rsg_iter)
            except StopIteration:
                rsg_iter = iter(rsg_train_ds)
                prof, tmpl = next(rsg_iter)
            train_rsg_step(prof, tmpl)

    # ── Validation ────────────────────────────────────────────────────
    val_ats_losses, val_dom_losses, val_dom_accs = [], [], []
    for r, jd, ats_t, dom_t in ats_val_ds:
        al, dl, da = val_ats_step(r, jd, ats_t, dom_t)
        val_ats_losses.append(float(al))
        val_dom_losses.append(float(dl))
        val_dom_accs.append(float(da))

    val_rsg_losses, val_rsg_accs = [], []
    for prof, tmpl in rsg_val_ds:
        rl, ra = val_rsg_step(prof, tmpl)
        val_rsg_losses.append(float(rl))
        val_rsg_accs.append(float(ra))

    # ── Compute epoch metrics ──────────────────────────────────────────
    e_ats_mae     = np.mean(ats_losses)       * 100  # back to 0-100 scale
    e_dom_loss    = np.mean(dom_losses)
    e_val_ats_mae = np.mean(val_ats_losses)   * 100
    e_val_dom_acc = np.mean(val_dom_accs)
    e_val_rsg_acc = np.mean(val_rsg_accs)
    e_val_rsg_loss= np.mean(val_rsg_losses)

    # Combined val_loss for early stopping
    val_ats_loss_norm = np.mean(val_ats_losses)   # already 0-1 scale
    val_loss = (ATS_W * val_ats_loss_norm
              + DOMAIN_W * np.mean(val_dom_losses)
              + RSG_W * e_val_rsg_loss)

    row = {
        "epoch": epoch,
        "train_ats_mae": round(e_ats_mae, 4),
        "train_dom_loss": round(e_dom_loss, 4),
        "val_ats_mae": round(e_val_ats_mae, 4),
        "val_dom_acc": round(e_val_dom_acc, 4),
        "val_rsg_acc": round(e_val_rsg_acc, 4),
        "val_loss": round(val_loss, 4)
    }
    log.append(row)

    print(f"Epoch {epoch:>3}/{MAX_EPOCHS} | "
          f"val_loss={val_loss:.4f} | "
          f"ATS_MAE={e_val_ats_mae:.2f} | "
          f"Dom_acc={e_val_dom_acc*100:.1f}% | "
          f"RSG_acc={e_val_rsg_acc*100:.1f}%")

    # ── Regression guard ───────────────────────────────────────────────
    if e_val_ats_mae > 8.0:
        print(f"\nREGRESSION GUARD: ATS MAE {e_val_ats_mae:.2f} > 8.0 at epoch {epoch}")
        print("HARD STOP — ATS regression detected. Restoring best weights.")
        model.load_weights(ckpt_path)
        break

    # ── Checkpoint and early stopping ─────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_epoch     = epoch
        patience_count = 0
        model.save_weights(ckpt_path)
        print(f"              ↑ Best checkpoint saved (val_loss={val_loss:.4f})")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            print(f"Best epoch: {best_epoch}")
            model.load_weights(ckpt_path)
            break

# ── Save training log ──────────────────────────────────────────────────
import csv
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=log[0].keys())
    w.writeheader()
    w.writerows(log)
print(f"\nTraining log saved: {csv_path}")

# ── Save final weights ─────────────────────────────────────────────────
final_path = os.path.join(UNIFIED_MODEL_DIR, "unified_final_weights.h5")
model.save_weights(final_path)

# ── Summary ────────────────────────────────────────────────────────────
best_row = log[best_epoch - 1]
print(f"\n=== STAGE 2 TRAINING SUMMARY ===")
print(f"Best epoch        : {best_epoch}")
print(f"val_loss (best)   : {best_row['val_loss']:.4f}")
print(f"val ATS MAE       : {best_row['val_ats_mae']:.2f}  (0-100 scale, target <8.0)")
print(f"val Domain acc    : {best_row['val_dom_acc']*100:.1f}%")
print(f"val RSG acc       : {best_row['val_rsg_acc']*100:.1f}%")
print()

# Inline regression check
if best_row["val_ats_mae"] < 8.0:
    print("ATS regression check : PASS")
else:
    print("ATS regression check : FAIL — report to Sai before eval")

if best_row["val_rsg_acc"] >= 0.50:
    print("RSG accuracy check   : PASS")
else:
    print("RSG accuracy check   : LOW — Stage 3 fine-tuning may be needed")
    print("                       (run INJECTION-2-STAGE3 if RSG acc < 0.50)")

print()
print(f"Best checkpoint : {ckpt_path}")
print(f"Final weights   : {final_path}")
print()
print("Send this output to Sai before running INJECTION-2-EVAL.")
