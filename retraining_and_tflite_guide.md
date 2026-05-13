# Unified Model Retraining & TFLite Conversion — Technical Guide

## 1. Why Retrain From Scratch?

The current model (`stage1_checkpoint.weights.h5`, **1.03 GB**) uses USE v4 (~257M params). Switching to MobileUSE v2 (~25M params) changes the embedding space entirely — every head weight is **incompatible**. A full retrain is mandatory.

**Size reduction:**
| Component | USE v4 (current) | MobileUSE v2 (target) |
|-----------|------------------|-----------------------|
| Encoder | ~980 MB | ~30 MB |
| All 3 Heads | ~1 MB | ~1 MB |
| **Keras total** | **1.03 GB** | **~35 MB** |
| **TFLite (heads-only, F16)** | 1.8 MB | **< 5 MB** |

## 2. Architecture Reference

Source: [unified_model.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/unified_model.py)

```
resume_text ──┐                    ┌─ Concat(emb_r, emb_j, cos, dot) → Dense(256) → Dense(64) → ats_score (1)
              ├─→ MobileUSE v2 ──→├─ jd_emb → Dense(256) → Dense(128) → domain_probs (7)
jd_text ──────┘   (frozen, 512d)   └─ resume_emb → Dense(512) → BN → Dense(256) → BN → Dense(128) → BN → rsg_template (46)
```

**Layer inventory (trainable heads only):**

| Head | Layers | Params | Input |
|------|--------|--------|-------|
| ATS Score | `ats_dense1(256)` → `ats_dense2(64)` → `ats_score(1)` | ~280K | 1026-dim (concat) |
| Domain | `dom_dense1(256)` → `dom_dense2(128)` → `domain_probs(7)` | ~165K | 512-dim (jd_emb) |
| RSG | `rsg_dense1(512)` → `rsg_dense2(256)` → `rsg_dense3(128)` → `rsg_template(46)` + 3×BN | ~465K | 512-dim (resume_emb) |

## 3. Dataset Summary

**ATS data** — [merged_final.csv](file:///c:/Users/saini/Desktop/ats/ats-ai-core/data/labeled/merged_final.csv) (65.7 MB)
- Columns: `resume_text`, `jd_text`, `score` (0-100), `domain_index` (0-6)
- ~68K pairs after NaN drop

**RSG data** — `weak_labels.csv` (external, must copy to `data/labeled/rsg_data.csv`)
- Columns: `profile_text`, `template_index`
- 46 classes after mapping via [rsg_label_mapping.json](file:///c:/Users/saini/Desktop/ats/model/unified_model/rsg_label_mapping.json)

**Config source of truth:** [config.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/config.py)

---

## 4. Stage R-0 — Data Preparation

**Goal:** Copy RSG data locally, validate both datasets, create reproducible splits.

### 4.1 Copy RSG Data
```python
import shutil
from src.config import RSG_CSV_PATH

# External source (will not exist on intern's machine)
EXTERNAL = r"C:\Users\saini\Desktop\rsg\RSG-AI-MODULE-main\data\labeled\weak_labels.csv"
shutil.copy2(EXTERNAL, str(RSG_CSV_PATH))
print(f"Copied to {RSG_CSV_PATH}")
```

### 4.2 Validate ATS Dataset
```python
from src.unified_engine.data_loader import load_ats_data
r, j, scores, domains = load_ats_data(str(LABELED_DIR / "merged_final.csv"))
assert len(r) >= 60000, f"Too few ATS pairs: {len(r)}"
assert scores.min() >= 0 and scores.max() <= 1.0
assert domains.min() >= 0 and domains.max() <= 6
# Print per-domain counts
for d in range(7):
    n = (domains == d).sum()
    flag = " ⚠️" if n < 150 else ""
    print(f"  Domain {d}: {n:,} pairs{flag}")
```

### 4.3 Validate RSG Dataset
```python
import json
from src.unified_engine.data_loader import load_rsg_data
from src.config import RSG_CSV_PATH, RSG_MAPPING_JSON

profiles, tids = load_rsg_data(str(RSG_CSV_PATH))
with open(RSG_MAPPING_JSON) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}

valid = [int(t) in id_to_idx for t in tids]
print(f"RSG valid: {sum(valid)} / {len(tids)}")
assert sum(valid) >= 1000, "Too few valid RSG samples"
```

### 4.4 Create Canonical Splits
```python
from sklearn.model_selection import train_test_split
import json, numpy as np

# ATS: 75/15/10
idx = np.arange(len(r))
tr, temp = train_test_split(idx, test_size=0.25, random_state=42, stratify=domains)
val, test = train_test_split(temp, test_size=0.40, random_state=42)

# RSG: 80/20
rsg_idx = np.arange(sum(valid))
rsg_tr, rsg_val = train_test_split(rsg_idx, test_size=0.20, random_state=42)

splits = {
    "ats_train": tr.tolist(), "ats_val": val.tolist(), "ats_test": test.tolist(),
    "rsg_train": rsg_tr.tolist(), "rsg_val": rsg_val.tolist()
}
with open("model/unified_model/data_splits.json", "w") as f:
    json.dump(splits, f)
```

### 4.5 Gate Checklist
- [ ] `rsg_data.csv` exists locally
- [ ] ATS ≥ 60K pairs, RSG ≥ 1K valid samples
- [ ] Every domain ≥ 150 pairs
- [ ] `data_splits.json` saved and **committed to git** (it contains only integer indices, no data — safe to track and essential for reproducibility across retraining cycles)

---

## 5. Stage R-1 — Smoke Build

**Goal:** Verify `build_unified_model()` works end-to-end with zero training.

```python
from src.unified_engine.unified_model import build_unified_model
import tensorflow as tf

model = build_unified_model()

# Verify encoder
enc = model.get_layer("mobile_use_encoder")
assert not enc.trainable, "Encoder must be frozen"

# 1-batch inference
r = tf.constant(["Python developer with Django experience"])
j = tf.constant(["Looking for backend engineer"])
ats, dom, rsg = model([r, j], training=False)

assert ats.shape == (1, 1), f"ATS shape: {ats.shape}"
assert dom.shape == (1, 7), f"Domain shape: {dom.shape}"
assert rsg.shape == (1, 46), f"RSG shape: {rsg.shape}"
assert 0 <= float(ats[0][0]) <= 1
print("R-1 SMOKE TEST PASSED")
```

### Gate
- [ ] Model builds, 3 outputs, correct shapes and value ranges

---

## 6. Stage R-2 — ATS + Domain Head Training

**Goal:** Train ATS and Domain heads. RSG receives no gradient. Encoder frozen.

### 6.1 Freeze Strategy
```
mobile_use_encoder  → FROZEN
ats_dense1/2, ats_score  → TRAINABLE
dom_dense1/2, domain_probs  → TRAINABLE
rsg_* (all)  → FROZEN (loss=None)
```

### 6.2 Compilation

> [!IMPORTANT]
> **Loss weight rationale (Issue B):** R-2 uses `ats=1.0 / domain=0.5` — NOT the canonical `0.35/0.35/0.30` from config. This is intentional: R-2 trains only ATS+Domain (RSG=None), so the ratio 1.0:0.5 = 2:1 gives ATS slightly more gradient priority during initial head training. The canonical `0.35/0.35/0.30` weights are used exclusively in **Stage R-4** for joint training. The R-4 script **must** import these from config, not carry forward the R-2 values.

```python
# R-2 ONLY weights — these are NOT the canonical joint weights
R2_ATS_WEIGHT = 1.0     # ATS gets priority during head training
R2_DOMAIN_WEIGHT = 0.5  # Domain secondary
# RSG = None (excluded from graph entirely)

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
```

### 6.3 Data Pipeline
```python
def make_ats_dataset(idxs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        r_texts[idxs], jd_texts[idxs],
        ats_scores[idxs].astype("float32"),
        domain_labels[idxs].astype("int32")
    ))
    if shuffle:
        ds = ds.shuffle(len(idxs), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda r, j, a, d: (
        {"resume_text": r, "jd_text": j},
        {"ats_score": tf.expand_dims(a, 1), "domain_probs": d}
    ))
    return ds
```

### 6.4 Callbacks
| Callback | Config |
|----------|--------|
| `EarlyStopping` | `monitor="val_loss"`, `patience=10`, `restore_best_weights=True` |
| `ModelCheckpoint` | `save_best_only=True` → `r2_ats_domain_best.weights.h5` |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=8`, `min_lr=1e-6` |
| `CSVLogger` | → `r2_training_log.csv` |

> [!NOTE]
> **Patience interaction (Issue D):** `ReduceLROnPlateau(patience=8)` will halve the LR before `EarlyStopping(patience=10)` triggers. Since `EarlyStopping` uses `restore_best_weights=True`, the final saved weights come from the best epoch overall — which may be *before* the LR reduction. This is correct behavior: if a LR reduction triggers improvement, those improved epochs will naturally become the new best. When reading training logs, expect to see a LR drop around epoch N, potentially followed by recovery — the `ModelCheckpoint` will capture whichever epoch achieves the lowest `val_loss` regardless of LR state.

### 6.5 Training Parameters

| Param | Value |
|-------|-------|
| Max epochs | 60 |
| Batch size | 32 |
| Initial LR | 1e-4 |
| ATS loss | MAE, weight=1.0 |
| Domain loss | SparseCCE, weight=0.5 |
| Domain class weights | IT=1.4, Non-IT=0.8, Design=0.9, Health=1.0, Finance=1.5, Legal=0.9, Edu=1.0 |

### 6.6 Post-Training Evaluation
```python
from sklearn.metrics import mean_absolute_error, f1_score
# On held-out test set (10%)
ats_pred, dom_pred, _ = model.predict(test_ds, verbose=0)
mae = mean_absolute_error(ats_true, ats_pred.flatten()) * 100
macro_f1 = f1_score(dom_true, np.argmax(dom_pred, 1), average="macro")
```

### 6.7 Gate
- [ ] ATS MAE < 8.0 on test set
- [ ] Domain F1 (macro) > 0.85
- [ ] Per-domain F1 > 0.80 for all 7 domains
- [ ] No NaN in history
- [ ] Encoder frozen post-training

> [!WARNING]
> If gate fails: reduce LR to 5e-5, increase domain class weights for Legal/Education, check for corrupted CSV rows. Do NOT proceed to R-3.

---

## 7. Stage R-3 — RSG Head Warmup

**Goal:** Train RSG head in isolation. ATS/Domain weights from R-2 are completely frozen.

### 7.1 Freeze Strategy
```
mobile_use_encoder  → FROZEN
ats_*, dom_*  → FROZEN (loaded from R-2)
rsg_dense1/2/3, rsg_bn1/2/3, rsg_drop1/2/3, rsg_template  → TRAINABLE
```

### 7.2 Weight Loading
```python
model = build_unified_model()
model.load_weights("model/unified_model/r2_ats_domain_best.weights.h5")

for layer in model.layers:
    layer.trainable = False
for name in ["rsg_dense1","rsg_bn1","rsg_drop1","rsg_dense2","rsg_bn2",
             "rsg_drop2","rsg_dense3","rsg_bn3","rsg_drop3","rsg_template"]:
    model.get_layer(name).trainable = True
```

### 7.3 Compilation
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={"ats_score": None, "domain_probs": None,
          "rsg_template": "sparse_categorical_crossentropy"},
    metrics={"rsg_template": ["accuracy"]}
)
```

### 7.4 RSG Data Contract
- `profile_text` is fed as **both** `resume_text` AND `jd_text` inputs
- RSG head only uses `resume_emb` — JD input is ignored by architecture
- Template IDs remapped to 0-45 via `rsg_label_mapping.json`

### 7.5 Training Parameters

| Param | Value | Rationale |
|-------|-------|-----------|
| Max epochs | 15 | RSG converges fast isolated |
| LR | 5e-4 | Higher OK for single random-init head |
| LR schedule | ReduceLR factor=0.5, patience=3, min=1e-5 |
| Early stopping | patience=5 on `val_rsg_template_accuracy`, mode=max |
| Checkpoint | `r3_rsg_warmup_best.weights.h5` on best val accuracy |

### 7.6 Regression Guard (Post-Training)
```python
# Forward pass on ATS val set — NO gradient
for r, jd, ats_t, dom_t in ats_val_ds:
    ats_out, dom_out, _ = model([r, jd], training=False)
    # compute MAE, domain accuracy

ats_drift = post_mae - r2_baseline_mae
dom_drift = post_acc - r2_baseline_acc
assert abs(ats_drift) < 0.5, f"ATS drifted by {ats_drift}"
assert abs(dom_drift) < 2.0, f"Domain drifted by {dom_drift}%"
```

### 7.7 Gate
- [ ] RSG val_accuracy ≥ 50%
- [ ] ATS MAE drift < 0.5 from R-2
- [ ] Domain accuracy drift < 2% from R-2

> [!TIP]
> If RSG < 50%: increase `rsg_dense1` to 1024, add a 4th dense layer, run 30 epochs, or check for class imbalance in 46 templates.

---

## 8. Stage R-4 — Joint Fine-tuning (Optional)

**Goal:** Polish all heads together at very low LR. Skip if R-2 + R-3 gates pass cleanly.

### 8.1 Freeze Strategy
```
mobile_use_encoder  → FROZEN
ALL heads  → TRAINABLE (at LR=5e-6)
```

> [!IMPORTANT]
> **Fresher fairness reapplication:** The fresher fairness gap (≤ 20 pts) is gated at R-4, not R-2. Since R-4 is the only stage where all heads train simultaneously, any fresher-specific score weighting or curriculum logic from the original training must be reapplied here. If R-4 is skipped, fresher fairness should be evaluated on the R-3 weights as a read-only check.

### 8.2 Alternating Batch Training
Standard `model.fit()` cannot handle the ATS/RSG label mismatch (ATS batches have no RSG labels, RSG batches have no ATS labels). Use a **custom training loop**:

```python
from src.config import SCORE_LOSS_WEIGHT, DOMAIN_LOSS_WEIGHT, RSG_LOSS_WEIGHT

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
ats_mae_fn = tf.keras.losses.MeanAbsoluteError()
dom_ce_fn  = tf.keras.losses.SparseCategoricalCrossentropy()
rsg_ce_fn  = tf.keras.losses.SparseCategoricalCrossentropy()

# CRITICAL (Issue B): Use canonical config weights, NOT R-2 weights
ATS_W  = SCORE_LOSS_WEIGHT    # 0.35
DOM_W  = DOMAIN_LOSS_WEIGHT   # 0.35
RSG_W  = RSG_LOSS_WEIGHT      # 0.30
assert (ATS_W, DOM_W, RSG_W) == (0.35, 0.35, 0.30), \
    f"FATAL: R-4 must use canonical weights, got {ATS_W}/{DOM_W}/{RSG_W}"

# Tightened regression threshold (Issue C): R-4 starts from a good
# checkpoint (best ~5.09 MAE). A guard at 8.0 allows ~3 pts regression
# before triggering — too loose for a polishing stage. Use 6.5 instead.
R4_MAE_HARD_STOP = 6.5

for epoch in range(30):
    for (r, jd, ats_t, dom_t) in ats_train_ds:
        # --- ATS + Domain batch ---
        with tf.GradientTape() as tape:
            ats_out, dom_out, _ = model([r, jd], training=True)
            loss = ATS_W * ats_mae_fn(ats_t, ats_out) + DOM_W * dom_ce_fn(dom_t, dom_out)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # --- RSG batch (interleaved) ---
        prof, tmpl = next(rsg_iter)
        with tf.GradientTape() as tape:
            _, _, rsg_out = model([prof, prof], training=True)
            loss = RSG_W * rsg_ce_fn(tmpl, rsg_out)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # --- Regression guard (tightened for polishing stage) ---
    if val_ats_mae > R4_MAE_HARD_STOP:
        print(f"HARD STOP: MAE {val_ats_mae:.2f} > {R4_MAE_HARD_STOP}. Restoring best.")
        model.load_weights(best_ckpt)
        break
```

### 8.3 Training Parameters

| Param | Value |
|-------|-------|
| Max epochs | 30 |
| LR | 5e-6 (very low — polishing only) |
| Loss weights | **0.35 / 0.35 / 0.30** (canonical, from config — NOT R-2 weights) |
| Early stopping | patience=8 on combined val_loss |
| Regression guard | Hard stop if ATS MAE > **6.5** at any epoch (Issue C: tightened from 8.0) |
| Output | `r4_joint_best.weights.h5` |

### 8.4 Gate
- [ ] ATS MAE < 8.0 (absolute gate) — ideally < 6.5 (R-4 continuation gate)
- [ ] Domain F1 > 0.85, RSG acc ≥ 50%
- [ ] No regression from R-2/R-3 baselines
- [ ] Fresher fairness gap ≤ 20 pts

---

## 8A. Pre-T-1 Gate — Flutter Encoder Handoff Dependency

> [!CAUTION]
> **Issue A: This is a blocking dependency.** The heads-only TFLite model accepts 512-dim embeddings, not raw text. The Flutter client **must** be able to produce these embeddings before the TFLite model is useful. This is a non-trivial integration task — do NOT treat it as a simple handoff.

### What the Flutter team needs to implement

The Flutter/mobile client must produce 512-dimensional `float32` embeddings from raw text **before** calling the TFLite model. There are two viable paths:

| Approach | Complexity | Size Impact | Recommended? |
|----------|------------|-------------|-------------|
| **A) TFHub Mobile SDK** — Load `universal-sentence-encoder-mobile/2` as a separate TFLite model on-device | Medium | +30 MB for encoder TFLite | ✅ Yes |
| **B) SentencePiece + Embedding Lookup** — Tokenize with SentencePiece, run token embeddings through a separate lightweight model | High | +15 MB tokenizer + model | ❌ Complex |

**Recommended path:** Approach A. Convert the MobileUSE encoder itself to a separate TFLite model that takes `string → float32[512]`, and ship it alongside the heads-only TFLite. Total on-device: ~30 MB (encoder) + ~2 MB (heads) ≈ **~32 MB**.

### Handoff checklist (must be confirmed before T-1)
- [ ] Flutter team has confirmed they can load a TFLite model that accepts string input
- [ ] OR Flutter team has confirmed they can produce 512-dim embeddings via another path
- [ ] A `HANDOFF.md` document exists in the repo root specifying the exact input/output contract
- [ ] A sample Flutter integration test exists that feeds an embedding and reads all 3 outputs

> [!WARNING]
> If the Flutter team cannot produce embeddings on-device, the entire heads-only strategy is blocked. Fallback: ship a server-side inference API instead of on-device TFLite.

---

## 9. Stage T-1 — TFLite Conversion

**Goal:** Export heads-only model to TFLite Float16 for mobile deployment.

### 9.1 Why Heads-Only?
`hub.KerasLayer` contains `FlexStaticRegexReplace` and other ops not supported by TFLite. The encoder must run separately on the client (see **§8A** above for the Flutter handoff dependency).

### 9.2 Build Heads-Only Model
```python
from src.config import EMBEDDING_DIM, RSG_NUM_CLASSES

resume_emb = tf.keras.Input(shape=(EMBEDDING_DIM,), name="resume_embedding")
jd_emb     = tf.keras.Input(shape=(EMBEDDING_DIM,), name="jd_embedding")

# Feature engineering (same as full model)
cos = tf.keras.layers.Dot(axes=1, normalize=True, name="cosine_sim")([resume_emb, jd_emb])
dot = tf.keras.layers.Dot(axes=1, normalize=False, name="dot_prod")([resume_emb, jd_emb])
feat = tf.keras.layers.Concatenate(name="ats_features")([resume_emb, jd_emb, cos, dot])

# HEAD 1: ATS (identical layer names)
x1 = tf.keras.layers.Dense(256, activation="relu", name="ats_dense1")(feat)
x1 = tf.keras.layers.Dropout(0.3, name="ats_drop1")(x1)
x1 = tf.keras.layers.Dense(64, activation="relu", name="ats_dense2")(x1)
x1 = tf.keras.layers.Dropout(0.2, name="ats_drop2")(x1)
ats_out = tf.keras.layers.Dense(1, activation="sigmoid", name="ats_score")(x1)

# HEAD 2: Domain
x2 = tf.keras.layers.Dense(256, activation="relu", name="dom_dense1")(jd_emb)
x2 = tf.keras.layers.Dropout(0.3, name="dom_drop1")(x2)
x2 = tf.keras.layers.Dense(128, activation="relu", name="dom_dense2")(x2)
x2 = tf.keras.layers.Dropout(0.2, name="dom_drop2")(x2)
dom_out = tf.keras.layers.Dense(7, activation="softmax", name="domain_probs")(x2)

# HEAD 3: RSG
x3 = tf.keras.layers.Dense(512, activation="relu", name="rsg_dense1")(resume_emb)
x3 = tf.keras.layers.BatchNormalization(name="rsg_bn1")(x3)
x3 = tf.keras.layers.Dropout(0.4, name="rsg_drop1")(x3)
x3 = tf.keras.layers.Dense(256, activation="relu", name="rsg_dense2")(x3)
x3 = tf.keras.layers.BatchNormalization(name="rsg_bn2")(x3)
x3 = tf.keras.layers.Dropout(0.3, name="rsg_drop2")(x3)
x3 = tf.keras.layers.Dense(128, activation="relu", name="rsg_dense3")(x3)
x3 = tf.keras.layers.BatchNormalization(name="rsg_bn3")(x3)
x3 = tf.keras.layers.Dropout(0.3, name="rsg_drop3")(x3)
rsg_out = tf.keras.layers.Dense(RSG_NUM_CLASSES, activation="softmax", name="rsg_template")(x3)

heads_model = tf.keras.Model(
    inputs=[resume_emb, jd_emb],
    outputs=[ats_out, dom_out, rsg_out],
    name="unified_heads_only"
)
```

### 9.3 Weight Transfer
```python
full_model = build_unified_model()
full_model.load_weights("model/unified_model/r4_joint_best.weights.h5")

transferred, skipped = 0, 0
for layer in heads_model.layers:
    if layer.count_params() > 0:
        try:
            full_layer = full_model.get_layer(layer.name)
            layer.set_weights(full_layer.get_weights())
            transferred += 1
        except ValueError:
            skipped += 1
print(f"Transferred: {transferred}, Skipped: {skipped}")
# Expected: all transfer, 0 skips
```

### 9.4 Parity Check — Keras Full vs Heads-Only
```python
for idx in range(50):
    # Full model: string → encoder → heads
    full_out = full_model([tf.constant([resume]), tf.constant([jd])], training=False)
    # Get embeddings
    encoder = full_model.get_layer("mobile_use_encoder")
    r_emb = encoder(tf.constant([resume]))
    j_emb = encoder(tf.constant([jd]))
    # Heads-only: embeddings → heads
    heads_out = heads_model([r_emb, j_emb], training=False)
    diff = abs(float(full_out[0][0][0]) - float(heads_out[0][0][0])) * 100
    assert diff < 0.01, f"Parity fail: {diff:.6f} pts"
```

### 9.5 Float16 Conversion
```python
saved_model_path = "model/unified_model/saved_model_heads"
heads_model.save(saved_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_bytes = converter.convert()
tflite_path = "model/unified_model/unified_mobile_float16.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_bytes)

size_mb = os.path.getsize(tflite_path) / 1e6
print(f"TFLite size: {size_mb:.1f} MB")
assert size_mb < 5.0, f"Size gate FAIL: {size_mb:.1f} MB"
```

### 9.6 Parity Check — Keras Heads vs TFLite
```python
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for idx in range(50):
    r_emb = encoder(tf.constant([resume])).numpy().astype(np.float32)
    j_emb = encoder(tf.constant([jd])).numpy().astype(np.float32)

    # Keras heads
    keras_ats = float(heads_model([r_emb, j_emb], training=False)[0][0][0]) * 100

    # TFLite
    for d in input_details:
        if "resume" in d["name"]: interpreter.set_tensor(d["index"], r_emb)
        elif "jd" in d["name"]:   interpreter.set_tensor(d["index"], j_emb)
    interpreter.invoke()
    tfl_ats = float([interpreter.get_tensor(d["index"]) for d in output_details
                     if interpreter.get_tensor(d["index"]).shape[-1] == 1][0][0][0]) * 100

    diff = abs(keras_ats - tfl_ats)
    assert diff < 2.0, f"TFLite parity fail: {diff:.4f} pts"
```

### 9.7 Final Output Summary
```json
{
  "tflite_file": "unified_mobile_float16.tflite",
  "architecture": "heads-only (embedding inputs)",
  "quantization": "Float16",
  "input_spec": {
    "resume_embedding": "float32[batch, 512]",
    "jd_embedding": "float32[batch, 512]"
  },
  "output_spec": {
    "ats_score": "float32[batch, 1] — sigmoid, multiply by 100",
    "domain_probs": "float32[batch, 7] — softmax, argmax for class",
    "rsg_template": "float32[batch, 46] — softmax, argmax → rsg_label_mapping"
  }
}
```

### 9.8 Flutter Inference Pipeline

> [!IMPORTANT]
> **See §8A for the encoder handoff dependency.** Step 1-2 below are the Flutter team's responsibility and require a separate MobileUSE encoder TFLite or equivalent. This is NOT a trivial dependency.

```
┌─────────────────────────────────────────────────────┐
│  FLUTTER CLIENT (on-device)                         │
│                                                     │
│  Step 1: Load MobileUSE encoder TFLite (~30 MB)     │
│          Input:  raw text string                    │
│          Output: float32[512] embedding             │
│          ⚠️ Flutter team owns this integration      │
│                                                     │
│  Step 2: Encode resume_text → resume_emb [512]      │
│          Encode jd_text     → jd_emb     [512]      │
│                                                     │
│  Step 3: Feed embeddings → heads-only TFLite (~2 MB)│
│          Input:  resume_embedding, jd_embedding      │
│                                                     │
│  Step 4: Read outputs:                              │
│          ats_score × 100 → ATS score (0-100)        │
│          argmax(domain_probs) → domain index (0-6)  │
│          argmax(rsg_template) → template index       │
│              → lookup in rsg_label_mapping.json      │
└─────────────────────────────────────────────────────┘
```

### 9.9 INT8 Quantization — Decision Record

> [!NOTE]
> **Issue E:** `MobileUSE_Retraining_Plan.docx` (Stage M-4) specifies INT8 as primary target with a representative dataset and Float16 as fallback. This guide implements **Float16 only**. Here is why:
>
> - The heads-only model has ~910K params → Float16 TFLite ≈ **1.8 MB**. This is far under the 5 MB gate and the original 30 MB target.
> - INT8 requires a representative dataset for calibration and risks accuracy loss on the softmax heads (Domain, RSG) where small probability differences matter.
> - The size benefit of INT8 over Float16 would be ~0.9 MB → 0.9 MB saved is not worth the calibration complexity and accuracy risk.
>
> **Decision:** Float16 is the production quantization. INT8 is documented but not implemented. If INT8 is ever needed (e.g., for a constrained IoT deployment), add a `--quantization int8` flag to `convert_unified_tflite.py` with a representative dataset of 200 samples from the test split.

This decision will be recorded in the `conversion_summary.json` output:
```json
{
  "quantization_decision": {
    "chosen": "Float16",
    "reason": "Heads-only model is 1.8 MB at Float16 — well under 5 MB gate. INT8 saves ~0.9 MB but risks softmax accuracy.",
    "int8_available": false,
    "int8_blocked_by": "Not needed — Float16 clears all size gates"
  }
}
```

### 9.10 Gate
- [ ] TFLite file < 5 MB (Float16 heads-only)
- [ ] Keras-to-heads parity < 0.01 pts
- [ ] Keras-to-TFLite parity < 2.0 pts
- [ ] All 3 output tensors present with correct shapes
- [ ] `conversion_summary.json` includes INT8 decision record

---

## 10. Consolidated Script Design

All stages will be implemented in a single file:

**[NEW]** `src/unified_engine/train_unified_mobile.py`

```bash
# Full pipeline
python src/unified_engine/train_unified_mobile.py

# Individual stages
python src/unified_engine/train_unified_mobile.py --stage r0
python src/unified_engine/train_unified_mobile.py --stage r1
python src/unified_engine/train_unified_mobile.py --stage r2
python src/unified_engine/train_unified_mobile.py --stage r3
python src/unified_engine/train_unified_mobile.py --stage r4
python src/unified_engine/train_unified_mobile.py --stage t1
```

**Design principles:**
- Each stage loads from the previous stage's checkpoint → idempotent
- All paths resolved from `src/config.py`
- Splits loaded from `data_splits.json` → reproducible (committed to git)
- Freeze assertions before AND after every `model.fit()`
- Regression guards: MAE > 8.0 hard stop in R-2, **MAE > 6.5** in R-4 (tightened for polishing)
- R-4 loss weights **must** be imported from config (`0.35/0.35/0.30`), not carried from R-2
- CSV logging at every stage

---

## 11. Final Verification Matrix

| Gate | Metric | Target | Stage | Notes |
|------|--------|--------|-------|-------|
| A | ATS MAE (0-100) | < 8.0 (R-2), **< 6.5** (R-4) | R-2, R-4 | R-4 uses tighter threshold |
| B | Band Accuracy | > 80% | R-2 | — |
| C | Domain F1 (macro) | > 0.85 | R-2 | — |
| D | Per-domain F1 | > 0.80 each | R-2 | Watch Legal/Finance with R-4 weights |
| E | RSG Accuracy | ≥ 50% | R-3 | — |
| F | Fresher Fairness | ≤ 20 pts gap | R-4 | Reapply fresher weighting in R-4 |
| G | Keras Model Size | < 50 MB | After R-4 | — |
| H | TFLite Size | < 5 MB | T-1 | Float16 heads-only; INT8 not needed |
| I | TFLite Parity | < 2.0 pts max | T-1 | — |
| J | Inference Latency | < 500 ms | R-1 | — |
| K | Flutter Handoff | Encoder produces 512-d emb | Pre-T-1 | **Blocking dependency** — see §8A |
