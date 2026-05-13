# Model Size Reduction Plan: Unified ATS+RSG Keras Model with Mobile Encoder

## Background & Problem Statement

The current unified ATS+RSG model (`best_unified_weights.h5` / `stage1_checkpoint.weights.h5`) is **1.03 GB** due to the Universal Sentence Encoder v4 (USE v4) backbone which contains **~257 million parameters**. The three task heads themselves are tiny (~1.8 MB as TFLite). The goal is to produce a single unified Keras model that:

1. Replaces USE v4 with **MobileUSE v2** (~30-40 MB encoder vs ~980 MB)
2. Retains all 3 heads: **ATS Score**, **Domain Classification**, **RSG Template**
3. Maintains current performance targets (MAE < 8.0, Domain F1 > 0.85, RSG Acc ≥ 50%)
4. Converts cleanly to TFLite < 30 MB

### Current Model Size Breakdown

| Component | Current (USE v4) | Target (MobileUSE v2) |
|-----------|------------------|-----------------------|
| Encoder | ~980 MB (257M params) | ~30-40 MB (~25M params) |
| ATS Head | ~0.3 MB | ~0.3 MB (unchanged) |
| Domain Head | ~0.2 MB | ~0.2 MB (unchanged) |
| RSG Head | ~0.5 MB | ~0.5 MB (unchanged) |
| **Total Keras** | **~1.03 GB** | **~35-45 MB** |
| **Total TFLite (Float16)** | ~491 MB (full) / 1.8 MB (heads-only) | **< 30 MB** |

---

## User Review Required

> [!IMPORTANT]
> **Encoder Change = Full Retrain Required**: Switching from USE v4 to MobileUSE v2 changes the embedding space. All head weights from the current model are **incompatible** — the entire model must be retrained from scratch with the new encoder. There is no shortcut for weight transfer.

> [!WARNING]
> **RSG Data Dependency**: The RSG dataset lives at `C:\Users\saini\Desktop\rsg\RSG-AI-MODULE-main\data\labeled\weak_labels.csv` which is **outside** the project directory. This path must remain accessible during retraining, or the data should be copied into the project.

> [!IMPORTANT]
> **Two config.py files exist** with conflicting values (see Issue #2). A decision is needed on which values are canonical before proceeding.

---

## 🚨 Critical Issues to Fix Before Training

### Issue #1 — Encoder URL Mismatch Across Codebase (HIGH / BLOCKING)

Three **different** encoder URLs are used across the codebase:

| File | Encoder URL | Type |
|------|------------|------|
| [config.py (outer)](file:///c:/Users/saini/Desktop/ats/src/config.py#L21) | `universal-sentence-encoder-lite/2` | USE Lite v2 |
| [config.py (ats-ai-core)](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/config.py#L46) | `universal-sentence-encoder/4` | USE v4 (FULL) |
| [unified_model.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/unified_model.py#L22) | `universal-sentence-encoder-mobile/2` | MobileUSE v2 |

**Impact**: The ATS standalone model (`model.py`) uses USE v4 (~980 MB). The unified model already uses MobileUSE v2 (~30 MB). But `config.py` in the outer project points to USE Lite v2 (a completely different model requiring SentencePiece tokenization). If any code mixes these, weights become incompatible.

**Fix**: Standardize on `MobileUSE v2` everywhere. Update both `config.py` files and `model.py` to use `https://tfhub.dev/google/universal-sentence-encoder-mobile/2`.

---

### Issue #2 — Duplicate & Conflicting config.py Files (HIGH / BLOCKING)

Two separate `config.py` files exist with **different hyperparameters**:

| Parameter | [src/config.py](file:///c:/Users/saini/Desktop/ats/src/config.py) (outer) | [ats-ai-core/src/config.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/config.py) (inner) |
|-----------|-----|-----|
| `USE_LITE_URL` | `encoder-lite/2` | `encoder/4` |
| `SCORE_LOSS_WEIGHT` | **0.7** | **0.35** |
| `DOMAIN_LOSS_WEIGHT` | **0.3** | **0.65** |
| `EPOCHS` | **50** | **60** |
| `MAX_MODEL_SIZE_MB` | **30** | **600** |
| `EARLY_STOPPING_PATIENCE` | _(missing)_ | **10** |
| `DOMAIN_CLASS_WEIGHTS` | _(missing)_ | _(defined)_ |

**Impact**: Which config is imported depends on `sys.path` at runtime. The loss weights (0.7/0.3 vs 0.35/0.65) will produce fundamentally different models. The model size gate (30 MB vs 600 MB) is contradictory.

**Fix**: Consolidate into a single canonical `config.py` inside `ats-ai-core/src/`. Delete or deprecate the outer `src/config.py`.

---

### Issue #3 — Broken ATS Weight Transfer in train_stage1.py (HIGH / BLOCKING)

[train_stage1.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage1.py#L47-L61) contains **undefined variables**:

```python
# Line 47: `layer_name_map` is never defined
for ats_name, unified_name in layer_name_map.items():
    # Line 49: `ats_model` is never defined (should be `ats_source`)
    ats_layer = ats_model.get_layer(ats_name)
```

**Impact**: Stage 1 training will crash with `NameError` at the ATS weight transfer step. The weight transfer from the standalone ATS model to the unified model never actually works.

**Fix**: Define `layer_name_map` dict mapping old ATS layer names to unified layer names, and replace `ats_model` with `ats_source`.

---

### Issue #4 — Encoder Layer Name Inconsistency (MEDIUM / BLOCKING)

The encoder layer is named differently across files:

| File | Layer Name |
|------|-----------|
| [unified_model.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/unified_model.py#L36) | `"mobile_use_encoder"` |
| [train_stage2.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage2.py#L48) | `"use_encoder"` |
| [train_mobileuse_cycle1.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_mobileuse_cycle1.py#L51) | `"use_encoder"` |
| [train_b2_head_tuning.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_b2_head_tuning.py#L58) | `"use_lite_encoder"` |
| [train_b3_rsg_surgical.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_b3_rsg_surgical.py#L87) | `"use_lite_encoder"` |
| [convert_to_tflite_b4.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/scripts/convert_to_tflite_b4.py#L167) | `"use_lite_encoder"` |

**Impact**: `model.get_layer("use_encoder")` in `train_stage2.py` will throw `ValueError` at runtime because `unified_model.py` names it `"mobile_use_encoder"`. The freeze assertions in `train_mobileuse_cycle1.py` (line 51-52) will also crash.

**Fix**: Standardize the encoder layer name to `"mobile_use_encoder"` everywhere (matching `unified_model.py`) or pick one name and update all references.

---

### Issue #5 — Data Loader Column Name Mismatch (MEDIUM / BLOCKING)

[data_loader.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/data_loader.py#L12-L13) expects columns `"ats_score"` and `"domain_label"`:

```python
ats_scores = (df["ats_score"].astype(float) / 100.0).values  # Line 12
domain_labels = df["domain_label"].astype(int).values          # Line 13
```

But [train_mobileuse_cycle1.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_mobileuse_cycle1.py#L87-L88) and [train_stage2.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage2.py#L81) read the same CSV with columns `"score"` and `"domain_index"`:

```python
ats_scores = (df["score"].astype(float) / 100.0).values       # Line 87
domain_labels = df["domain_index"].astype(int).values          # Line 88
```

**Impact**: `load_ats_data()` in `data_loader.py` will crash with `KeyError` because `merged_final.csv` uses `"score"` / `"domain_index"`, not `"ats_score"` / `"domain_label"`. The `validate_production.py` script which calls `load_ats_data()` will also fail.

**Fix**: Update `data_loader.py` to use the correct column names (`score` and `domain_index`), or add column aliasing logic.

---

### Issue #6 — Missing `rsg_bn3` Layer in unified_model.py (MEDIUM)

[unified_model.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/unified_model.py#L75-L77) defines the RSG head with:
```python
x3 = Dense(128, activation="relu", name="rsg_dense3")(x3)
x3 = BatchNormalization(name="rsg_bn3")(x3)
x3 = Dropout(0.3, name="rsg_drop3")(x3)
```

But [train_stage1.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage1.py#L71-L75) lists RSG trainable layers as:
```python
rsg_layer_names = [
    "rsg_dense1", "rsg_bn1", "rsg_drop1",
    "rsg_dense2", "rsg_bn2", "rsg_drop2",
    "rsg_dense3", "rsg_template"
]
```

Missing `"rsg_bn3"` and `"rsg_drop3"` — these layers will remain **frozen** during Stage 1 RSG warmup, degrading RSG head training.

**Fix**: Add `"rsg_bn3"` and `"rsg_drop3"` to the trainable layer list in `train_stage1.py`.

---

### Issue #7 — Hardcoded Absolute Paths (MEDIUM)

Multiple scripts contain hardcoded Windows paths:

| File | Hardcoded Path |
|------|---------------|
| [train_stage1.py:15](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage1.py#L15) | `C:\Users\saini\Desktop\rsg\RSG-AI-MODULE-main\...` |
| [train_stage2.py:21](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage2.py#L21) | `C:\Users\saini\Desktop\ats\ats-ai-core\...` |
| [train_stage3.py:15](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage3.py#L15) | `C:\Users\saini\Desktop\rsg\...` |
| [transfer_rsg_weights.py:14](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/transfer_rsg_weights.py#L14) | `C:\Users\saini\Desktop\rsg\...` |
| [validate_production.py:30](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/validate_production.py#L30) | `C:\Users\saini\Desktop\rsg\...` |
| [train_b3_rsg_surgical.py:49](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_b3_rsg_surgical.py#L49) | `C:\Users\saini\Desktop\rsg\...` |

**Impact**: Will break for any other developer or machine. The intern cannot run any RSG training.

**Fix**: Move RSG data into the project or use environment variables / config for paths.

---

### Issue #8 — Loss Function Inconsistency Across Training Stages (MEDIUM)

| Script | ATS Loss | Rationale |
|--------|----------|-----------|
| [model.py (standalone)](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/ats_engine/model.py#L138) | `mean_absolute_error` | Production ATS model |
| [train_b2_head_tuning.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_b2_head_tuning.py#L74) | `mse` | B-2 unified training |
| [train_mobileuse_cycle1.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_mobileuse_cycle1.py#L65) | `mean_absolute_error` | M-2 MobileUSE training |
| [train_stage2.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage2.py#L71) | custom `MeanAbsoluteError()` | Stage 2 joint training |

**Impact**: MSE vs MAE loss produces different optimization landscapes. MAE is more robust to outliers and should be the canonical choice for ATS scoring (matches the evaluation metric). Using MSE in B-2 may have caused suboptimal weight initialization.

**Fix**: Standardize on MAE for the ATS head across all training scripts.

---

### Issue #9 — Domain Output Name Inconsistency (MEDIUM)

| File | Domain Output Layer Name |
|------|-------------------------|
| [model.py (standalone ATS)](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/ats_engine/model.py#L124) | `"domain_logits"` |
| [unified_model.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/unified_model.py#L64) | `"domain_probs"` |
| [trainer.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/ats_engine/trainer.py#L110) | references `"domain_logits"` |

**Impact**: Weight transfer from standalone ATS model to unified model will fail on the domain head because layer names don't match. The trainer's `_to_tf_dataset()` method outputs `"domain_logits"` keys, which won't match unified model outputs.

**Fix**: Standardize to `"domain_probs"` in the unified model (softmax output → probabilities, not logits).

---

### Issue #10 — Data Limit in data_loader.py (LOW)

[data_loader.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/data_loader.py#L5) has `limit: int = 5000` by default:

```python
def load_ats_data(csv_path: str, limit: int = 5000):
```

**Impact**: Only 5,000 of ~68K ATS samples are used by default. Scripts that call `load_ats_data()` without `limit=None` will train on a tiny fraction of data.

**Fix**: Change default to `limit=None` or remove the limit parameter.

---

### Issue #11 — No Regularization on Dense Layers (LOW)

All Dense layers across the model use **no L1/L2 regularization** (confirmed in [architecture_audit.json](file:///c:/Users/saini/Desktop/ats/ats-ai-core/architecture_audit.json)). Dropout alone may be insufficient for generalization, especially with the smaller MobileUSE encoder which has less representational capacity.

**Fix**: Consider adding L2 regularization (1e-4) to dense layers, especially in the RSG head which has the most parameters.

---

### Issue #12 — Domain Imbalance Not Addressed for MobileUSE Retraining (LOW)

Current domain distribution is heavily skewed:

| Domain | % of Data | Risk |
|--------|-----------|------|
| IT / Software | 35.7% | Over-represented |
| Legal | 1.9% | **Severely under-represented** |
| Education | 3.3% | **Under-represented** |
| Design / Creative | 7.6% | Moderate |

**Impact**: With a smaller encoder, the model has less capacity for tail classes. Domain F1 for Legal/Education may drop below the 0.80 per-domain gate.

**Fix**: Use the existing `DOMAIN_CLASS_WEIGHTS` in config but also consider upsampling Legal and Education domains.

---

### Issue #13 — Unused `sentencepiece` Import in unified_model.py (LOW)

[unified_model.py:14](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/unified_model.py#L14) imports `sentencepiece` but never uses it:

```python
import sentencepiece as spm  # never used
```

MobileUSE v2 accepts raw strings directly — no SentencePiece tokenization needed. This is a false dependency that will cause `ImportError` if the package isn't installed.

**Fix**: Remove the unused import.

---

### Issue #14 — `train_b2_head_tuning.py` Uses Wrong Import Path (LOW)

[train_b2_head_tuning.py:29](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_b2_head_tuning.py#L29):
```python
from unified_model import build_unified_model  # relative import
```

This only works if CWD is `src/unified_engine/`. Other scripts correctly use:
```python
from src.unified_engine.unified_model import build_unified_model
```

**Fix**: Standardize all imports to use absolute project-relative paths.

---

## Proposed Changes

### Phase 1: Encoder Consolidation & Config Unification

#### [MODIFY] [config.py (ats-ai-core)](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/config.py)
- Change `USE_LITE_URL` to `https://tfhub.dev/google/universal-sentence-encoder-mobile/2`
- Set `MAX_MODEL_SIZE_MB = 30.0`
- Add `RSG_NUM_CLASSES = 46`
- Add `RSG_CSV_PATH` and `RSG_MAPPING_JSON` to centralize RSG paths
- Standardize loss weights: `SCORE_LOSS_WEIGHT = 0.35`, `DOMAIN_LOSS_WEIGHT = 0.35`, `RSG_LOSS_WEIGHT = 0.30`

#### [DEPRECATE] [config.py (outer)](file:///c:/Users/saini/Desktop/ats/src/config.py)
- Add deprecation notice pointing to `ats-ai-core/src/config.py`
- Or delete entirely if no code depends on it

#### [MODIFY] [unified_model.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/unified_model.py)
- Import encoder URL from `config.py` instead of hardcoding
- Standardize encoder layer name to `"mobile_use_encoder"`
- Remove unused `sentencepiece` import
- Remove unused `glob` import
- Add L2 regularization to dense layers (optional, based on review)

---

### Phase 2: Fix Broken Training Scripts

#### [MODIFY] [train_stage1.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage1.py)
- Fix `layer_name_map` (undefined variable → define the dict)
- Fix `ats_model` → `ats_source`
- Add `"rsg_bn3"` and `"rsg_drop3"` to RSG trainable layers list
- Replace hardcoded RSG path with config reference
- Fix encoder layer name reference

#### [MODIFY] [train_stage2.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_stage2.py)
- Fix encoder layer name: `"use_encoder"` → `"mobile_use_encoder"`
- Replace hardcoded paths with config references
- Standardize ATS loss to MAE

#### [MODIFY] [data_loader.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/data_loader.py)
- Fix column names: `"ats_score"` → `"score"`, `"domain_label"` → `"domain_index"`
- Change default `limit` to `None`

#### [MODIFY] [train_b2_head_tuning.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_b2_head_tuning.py)
- Fix import path: `from unified_model` → `from src.unified_engine.unified_model`
- Fix encoder layer name reference
- Change ATS loss from `mse` to `mean_absolute_error`

#### [MODIFY] [train_b3_rsg_surgical.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_b3_rsg_surgical.py)
- Fix encoder layer name: `"use_lite_encoder"` → `"mobile_use_encoder"`
- Replace hardcoded RSG path with config reference

#### [MODIFY] [validate_production.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/validate_production.py)
- Replace hardcoded RSG path with config reference

#### [MODIFY] [convert_to_tflite_b4.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/scripts/convert_to_tflite_b4.py)
- Fix encoder layer name: `"use_lite_encoder"` → `"mobile_use_encoder"`

---

### Phase 3: Dataset Pipeline Consolidation

#### [NEW] [prepare_unified_data.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/scripts/prepare_unified_data.py)
- Single script that loads both ATS and RSG datasets
- Validates column names and data types
- Applies domain class balancing (upsampling Legal/Education)
- Outputs clean train/val/test splits with consistent format
- Stores RSG data locally in `ats-ai-core/data/labeled/rsg_data.csv`

---

### Phase 4: Clean Unified Retraining

#### [NEW] [train_unified_mobile.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/unified_engine/train_unified_mobile.py)
- Single clean training script that replaces the fragmented Stage 1/2/3 approach
- Uses `build_unified_model()` with MobileUSE v2 (from config)
- Phase 1: RSG warmup (10 epochs, RSG head only, LR=1e-4)
- Phase 2: Joint training all heads (up to 60 epochs, LR=5e-6)
- Built-in regression guards for ATS MAE > 8.0
- Proper early stopping on combined val_loss
- Saves `best_unified_mobile_weights.h5`

---

### Phase 5: TFLite Conversion & Validation

#### [MODIFY] [convert_to_tflite_b4.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/scripts/convert_to_tflite_b4.py)
- Update to use MobileUSE encoder layer name
- Target heads-only TFLite < 5 MB (Float16)
- Full-model TFLite target < 30 MB (if encoder can convert)
- Add INT8 quantization as fallback

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Should the inference pipeline ([inference.py](file:///c:/Users/saini/Desktop/ats/ats-ai-core/src/ats_engine/inference.py)) be updated to use the unified model (3 heads) instead of the standalone ATS model (2 heads)? This would add RSG template prediction to the API response.

> [!IMPORTANT]
> **Q2**: The RSG dataset is at `C:\Users\saini\Desktop\rsg\RSG-AI-MODULE-main\data\labeled\weak_labels.csv`. Should this be copied into the ATS project directory for portability, or should we keep it as an external dependency?

> [!IMPORTANT]
> **Q3**: The current Domain classification uses `DOMAIN_CLASS_WEIGHTS` with manual overrides (e.g., IT=1.4, Finance=1.5). Should these be recalculated for the MobileUSE retraining, or should we use purely `sklearn.compute_class_weight("balanced")`?

> [!IMPORTANT]
> **Q4**: The outer `src/` directory has its own `config.py`, `training/`, `encoding/`, and other modules that appear to be an older version of the pipeline. Can these be removed/deprecated, or is any code still depending on them?

---

## Verification Plan

### Automated Tests

```bash
# 1. Verify unified model builds successfully with MobileUSE
python -c "from src.unified_engine.unified_model import build_unified_model; m = build_unified_model(); m.summary()"

# 2. Run full training pipeline (smoke test with 200 samples)
python train_unified_mobile.py --smoke-test

# 3. Validate all 3 heads produce correct shapes
python src/unified_engine/validate_production.py

# 4. TFLite conversion + parity check
python scripts/convert_to_tflite_b4.py

# 5. Final size gate
python scripts/validate_encoder_size.py
```

### Performance Gates (Must Pass Before Deployment)

| Gate | Metric | Target | Measurement |
|------|--------|--------|-------------|
| A | ATS MAE (0-100) | < 8.0 | Held-out test set |
| B | Band Accuracy | > 80% | Held-out test set |
| C | Domain F1 (macro) | > 0.85 | Held-out test set |
| D | Per-domain F1 | > 0.80 each | Held-out test set |
| E | RSG Accuracy | ≥ 50% | RSG test split |
| F | Fresher Fairness Gap | ≤ 20 pts | Fresher vs Experienced |
| G | Keras Model Size | < 50 MB | Disk |
| H | TFLite Size (heads-only) | < 5 MB | Disk |
| I | TFLite Parity | < 2.0 pts max diff | 50-sample comparison |
| J | Inference Time | < 500 ms | Single-sample latency |

### Manual Verification
- Run 3 sample inferences (IT, Healthcare, Marketing) and verify reasonable scores
- Compare domain predictions against standalone ATS model
- Verify TFLite model works in isolation with pre-computed embeddings

---

## Execution Timeline (Estimated)

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Phase 1: Config & Encoder Fix | 1-2 hours | None |
| Phase 2: Fix Broken Scripts | 2-3 hours | Phase 1 |
| Phase 3: Dataset Consolidation | 1-2 hours | Phase 2 |
| Phase 4: Retraining | 4-8 hours (GPU) | Phase 3 |
| Phase 5: TFLite + Validation | 1-2 hours | Phase 4 |
| **Total** | **~10-17 hours** | |
