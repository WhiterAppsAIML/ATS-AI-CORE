# EDGE V2 — Stage-Wise Prompt Injections
# Encoder: all-MiniLM-L6-v2 (384-dim, tokenized inputs)
# Target: single INT8 .tflite < 30 MB, no Flex ops
# Usage: Paste the relevant INJECTION block before starting each stage.

---

## INJECTION-E0 — Encoder Validation

### Context
We are migrating from MobileUSE (string-input, 512-dim) to `sentence-transformers/all-MiniLM-L6-v2` (token-input, 384-dim). This stage validates the encoder in isolation before any architecture changes.

### Task
1. **Download & convert MiniLM to TFLite (INT8)**
   - Use `transformers` to load `sentence-transformers/all-MiniLM-L6-v2`
   - Export to SavedModel with fixed input shapes: `input_ids [1,128]`, `attention_mask [1,128]`
   - Convert via `tf.lite.TFLiteConverter` with `TFLITE_BUILTINS_INT8` representative dataset
   - Assert: **zero Flex ops**, encoder `.tflite` size **< 25 MB**

2. **Embedding quality check**
   - Encode 50 resume–JD pairs with both MiniLM (384-dim) and old USE (512-dim)
   - Compute pairwise cosine similarity rankings for each encoder
   - Assert: Spearman rank correlation **> 0.85** between the two rankings

3. **Tokenizer fidelity check**
   - Tokenize 20 sample texts with Python `AutoTokenizer`
   - Compare token IDs against Dart `WordPieceTokenizer` output (provide JSON for manual Dart check)
   - Assert: **100% match** on all 20 samples

### Output Files
- `model/minilm/encoder_only.tflite` — INT8 encoder
- `evaluation/e0_encoder_report.json` — sizes, op list, Spearman ρ, tokenizer match results

### Script Path
`scripts/e0_validate_encoder.py`

### Hard Stop — Do NOT proceed to E1 if:
- Any Flex op detected in encoder `.tflite`
- Encoder size ≥ 25 MB
- Spearman ρ < 0.85
- Any tokenizer mismatch in the 20-sample set

### Definition of Done
- [ ] `encoder_only.tflite` exists, < 25 MB, zero Flex ops
- [ ] `e0_encoder_report.json` written with all metrics passing
- [ ] Print `E0 PASSED — proceed to E1`

---

## INJECTION-E1 — Model Architecture Update

### Context
E0 confirmed MiniLM is viable. Now replace the USE encoder inside the unified model with a MiniLM submodel and verify INT8 conversion of the full 3-head architecture.

### Task
1. **Create `unified_model_minilm.py`** (new file, do NOT modify old `unified_model.py`)
   - 4 input tensors: `resume_input_ids [1,128]`, `resume_attention_mask [1,128]`, `jd_input_ids [1,128]`, `jd_attention_mask [1,128]` — all `int32`
   - Shared MiniLM encoder submodel (frozen), output 384-dim CLS embeddings
   - Feature concat for ATS head: `resume_emb + jd_emb + cosine_sim + dot_prod` = 770 dims
   - **HEAD 1 — ATS Score:** Dense(256)→Dropout(0.3)→Dense(64)→Dropout(0.2)→Dense(1, sigmoid) — name `ats_score`
   - **HEAD 2 — Domain:** Dense(256)→Dropout(0.3)→Dense(128)→Dropout(0.2)→Dense(7, softmax) — name `domain_probs`, fed from `jd_emb`
   - **HEAD 3 — RSG:** Dense(512)→BN→Dropout(0.4)→Dense(256)→BN→Dropout(0.3)→Dense(128)→BN→Dropout(0.3)→Dense(46, softmax) — name `rsg_template`, fed from `resume_emb`

2. **Dry-run INT8 conversion**
   - Convert full model to INT8 TFLite with representative dataset of random `int32` token IDs
   - Assert: zero Flex ops, output shapes `[1,1]`, `[1,7]`, `[1,46]`

3. **Update config** — add to `src/config.py`:
   ```python
   MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
   MINILM_EMBEDDING_DIM = 384
   MINILM_MAX_SEQ_LEN = 128
   ```

### Output Files
- `ats-ai-core/src/unified_engine/unified_model_minilm.py`
- `evaluation/e1_architecture_validation.json` — op list, shapes, size

### Regression Guard
- Do NOT modify `unified_model.py` (the USE-based model remains as fallback)
- Head layer names MUST match exactly: `ats_score`, `domain_probs`, `rsg_template`

### Hard Stop — Do NOT proceed to R0 if:
- Any Flex op in full-model `.tflite`
- Output shapes don't match `[1,1]`, `[1,7]`, `[1,46]`

### Definition of Done
- [ ] `unified_model_minilm.py` builds, `model.summary()` prints correct shapes
- [ ] INT8 conversion succeeds with zero Flex ops
- [ ] `e1_architecture_validation.json` written
- [ ] Print `E1 PASSED — proceed to R0`

---

## INJECTION-R0 — Tokenization Pipeline

### Context
MiniLM requires tokenized inputs (not raw strings). Pre-tokenize all training data to `.npz` files so R1–R4 training scripts load tokens directly.

### Task
1. **Tokenize ATS dataset** (`data/labeled/merged_final.csv`)
   - Use `AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")`
   - For each row: tokenize `resume_text` and `jd_text` separately, `max_length=128`, `padding='max_length'`, `truncation=True`
   - Save as `data/tokenized/ats_tokenized.npz` with keys:
     - `resume_input_ids`, `resume_attention_mask` — shape `[N, 128]`, dtype `int32`
     - `jd_input_ids`, `jd_attention_mask` — shape `[N, 128]`, dtype `int32`
     - `ats_scores` — shape `[N]`, dtype `float32` (0–1 scale)
     - `domain_labels` — shape `[N]`, dtype `int32` (0–6)

2. **Tokenize RSG dataset** (`data/labeled/rsg_data.csv`)
   - Tokenize `profile_text`, save as `data/tokenized/rsg_tokenized.npz`:
     - `profile_input_ids`, `profile_attention_mask` — shape `[M, 128]`
     - `rsg_labels` — shape `[M]`, dtype `int32` (0–45)

3. **Validation checks**
   - Print sequence length distribution (mean, p50, p95, p99) for resume, JD, profile
   - Assert all labels in valid ranges
   - Assert no NaN or empty sequences

4. **Regenerate canonical splits** using `model/unified_model/data_splits.json` indices
   - Verify split sizes match expectations (75/15/10 ATS, 80/20 RSG)

### Output Files
- `data/tokenized/ats_tokenized.npz`
- `data/tokenized/rsg_tokenized.npz`
- `evaluation/r0_tokenization_report.json`

### Script Path
`scripts/r0_tokenize_data.py`

### Definition of Done
- [ ] Both `.npz` files saved, shapes correct
- [ ] Label integrity verified (no out-of-range values)
- [ ] Sequence length stats printed
- [ ] Print `R0 PASSED — proceed to R1`

---

## INJECTION-R1 — Heads Warm-up

### Context
Train only the 3 heads (ATS, Domain, RSG) with the MiniLM encoder **frozen**. This establishes baseline head performance before joint fine-tuning.

### Task
1. **Load tokenized data** from `data/tokenized/*.npz` and splits from `data_splits.json`

2. **Build model** via `unified_model_minilm.py`, freeze all encoder layers:
   ```python
   for layer in model.layers:
       if 'minilm' in layer.name or 'encoder' in layer.name:
           layer.trainable = False
   ```

3. **Training config**
   - Optimizer: Adam, lr=`1e-3`
   - Loss weights: ATS=`0.35`, Domain=`0.35`, RSG=`0.30`
   - Losses: MAE (ATS), SparseCategoricalCrossentropy (Domain, RSG)
   - Batch size: 32, max epochs: 40
   - Early stopping on `val_loss`, patience=8
   - Domain class weights: `{0:1.4, 1:0.8, 2:0.9, 3:1.0, 4:1.5, 5:0.9, 6:1.0}`

4. **Alternating batches** — each epoch alternates ATS batches and RSG batches (RSG uses `profile_input_ids` for BOTH resume and JD inputs)

5. **Checkpoint** — save best weights to `model/unified_model/r1_best_weights.h5`

### Target Gates (soft — proceed even if slightly under)
| Metric | Target |
|--------|--------|
| ATS val MAE (0–100) | < 10.0 |
| Domain val F1 (macro) | > 0.72 |
| RSG val Accuracy | > 0.42 |

### Script Path
`scripts/r1_heads_warmup.py`

### Hard Stop — STOP and report if:
- NaN loss at any epoch
- ATS MAE > 15.0 after epoch 10
- Any checkpoint save error (apply `UNIFIED_MODEL_DIR` absolute path pattern from INJECTION-2A)

### Definition of Done
- [ ] Training completes or early-stops naturally
- [ ] `r1_best_weights.h5` saved
- [ ] Print summary: val ATS MAE, Domain F1, RSG Acc
- [ ] Print `R1 COMPLETE — proceed to R2`

---

## INJECTION-R2 — Joint Fine-tuning

### Context
Unfreeze the MiniLM encoder and train end-to-end with a very low encoder LR. This is the primary training stage.

### Task
1. **Load model**, load `r1_best_weights.h5`

2. **Unfreeze encoder** with differential LR:
   - Encoder layers: lr=`2e-5`
   - Head layers: lr=`5e-4`
   - Use two separate optimizers or a custom LR schedule

3. **Training config**
   - Loss weights: ATS=`0.35`, Domain=`0.35`, RSG=`0.30`
   - Batch size: 32, max epochs: 50
   - Early stopping on `val_loss`, patience=10
   - Same alternating batch strategy as R1

4. **Checkpoint** — save best to `model/unified_model/r2_best_weights.h5`

### Target Gates
| Metric | Target |
|--------|--------|
| ATS val MAE (0–100) | < 8.0 |
| ATS test MAE (0–100) | < 8.5 |
| Domain val F1 (macro) | > 0.82 |
| RSG val Accuracy | > 0.55 |

### Script Path
`scripts/r2_joint_finetune.py`

### Hard Stop
- NaN loss at any epoch
- ATS val MAE increases by > 3.0 from R1 baseline (regression)

### Definition of Done
- [ ] `r2_best_weights.h5` saved
- [ ] All 4 target gates printed with PASS/FAIL
- [ ] Print `R2 COMPLETE — proceed to R3`

---

## INJECTION-R3 — RSG Boost Pass

### Context
RSG accuracy may lag behind. Temporarily upweight RSG loss while keeping encoder frozen to boost RSG without regressing ATS.

### Task
1. **Load model**, load `r2_best_weights.h5`

2. **Freeze encoder again**, unfreeze only `rsg_*` layers

3. **Training config**
   - Loss weights: ATS=`0.20`, Domain=`0.20`, RSG=`0.60` (boosted)
   - Optimizer: Adam, lr=`5e-4`
   - Batch size: 32, max epochs: 25
   - Early stopping on `val_rsg_template_accuracy`, patience=6

4. **Drift guard** — after every epoch, evaluate ATS val MAE. If it exceeds `8.5` (0–100 scale), STOP immediately and save current weights.

5. **Checkpoint** — save best to `model/unified_model/r3_best_weights.h5`

### Target Gates
| Metric | Target | Hard Ceiling |
|--------|--------|-------------|
| RSG val Accuracy | > 0.62 | — |
| ATS val MAE (0–100) | — | ≤ 8.5 |
| Domain val F1 | — | ≥ 0.78 |

### Script Path
`scripts/r3_rsg_boost.py`

### Hard Stop
- ATS val MAE > 8.5 (drift guard triggers)
- NaN loss

### Definition of Done
- [ ] `r3_best_weights.h5` saved
- [ ] RSG accuracy improved from R2
- [ ] ATS MAE still within ceiling
- [ ] Print `R3 COMPLETE — proceed to R4`

---

## INJECTION-R4 — Final Joint Polish

### Context
Restore canonical loss weights and run a final joint pass to lock in all three heads at production-quality metrics.

### Task
1. **Load model**, load `r3_best_weights.h5`

2. **Unfreeze everything** (encoder + all heads)

3. **Training config**
   - Loss weights: ATS=`0.35`, Domain=`0.35`, RSG=`0.30` (canonical)
   - Optimizer: Adam, lr=`1e-5` (very low — polish only)
   - Batch size: 32, max epochs: 30
   - Early stopping on `val_loss`, patience=8

4. **Checkpoint** — save best to `model/unified_model/r4_final_weights.h5`

5. **Run full evaluation** on held-out test set after training:
   - ATS test MAE, per-domain ATS MAE breakdown
   - Domain test F1 (macro + per-class)
   - RSG test accuracy (top-1 and top-3)
   - Fresher fairness gap (ATS MAE for freshers vs experienced, gap must be ≤ 20)

### Final Hard Gates — ALL must pass
| Metric | Gate |
|--------|------|
| ATS val MAE (0–100) | < 6.5 |
| ATS test MAE (0–100) | < 7.0 |
| Domain F1 macro | > 0.85 |
| Domain F1 per-class (all 7) | > 0.80 each |
| RSG val Accuracy | > 0.65 |
| Fresher fairness gap | ≤ 20 |

### Script Path
`scripts/r4_final_polish.py`

### Hard Stop
- Any gate fails → report exact values, do NOT proceed to T1
- NaN loss

### Output Files
- `model/unified_model/r4_final_weights.h5`
- `evaluation/r4_final_eval_report.json` — all metrics + per-domain breakdown

### Definition of Done
- [ ] All 6 gates pass
- [ ] `r4_final_weights.h5` and `r4_final_eval_report.json` saved
- [ ] Print `R4 PASSED ALL GATES — proceed to T1`

---

## INJECTION-T1 — INT8 Export + Metadata

### Context
Convert the final trained model to a production INT8 TFLite file with embedded tokenizer metadata.

### Task
1. **Load model**, load `r4_final_weights.h5`

2. **INT8 conversion**
   - Representative dataset: 200 real samples from training set
   - `supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
   - Input/output types: `tf.int32` (inputs), `tf.float32` (outputs)
   - Assert: zero Flex ops, zero SELECT_TF_OPS

3. **Embed tokenizer metadata**
   - Extract `vocab.txt` from MiniLM tokenizer
   - Use `tflite_support.metadata` to attach vocab + tokenizer config to `.tflite`
   - Metadata must include: vocab file, max_seq_len=128, do_lower_case=true

4. **Parity check**
   - Run 50 samples through both Keras model and TFLite interpreter
   - Assert: ATS score max diff < 2.0 points (on 0–100 scale)
   - Assert: Domain prediction match ≥ 96%
   - Assert: RSG top-1 prediction match ≥ 94%

5. **Size check** — assert final `.tflite` < 30 MB

### Output Files
- `model/tflite/ats_unified_minilm_int8.tflite`
- `model/tflite/vocab.txt`
- `evaluation/t1_conversion_report.json` — parity stats, size, op audit

### Script Path
`scripts/t1_export_tflite.py`

### Hard Stop
- Any Flex op detected
- Parity diff > 2.0 on ATS score
- File size ≥ 30 MB

### Definition of Done
- [ ] `.tflite` file saved, < 30 MB, zero Flex ops
- [ ] Vocab embedded in metadata
- [ ] Parity check passes (50 samples)
- [ ] `t1_conversion_report.json` written
- [ ] Print `T1 PASSED — proceed to T2`

---

## INJECTION-T2 — Flutter Integration Verification

### Context
Final validation that the `.tflite` model works correctly when loaded by the Flutter app's TFLite interpreter.

### Task
1. **Generate integration test data**
   - Pick 20 diverse resume–JD pairs (cover all 7 domains, include 4+ fresher profiles)
   - For each pair: provide raw text, expected Python tokenization, expected Keras outputs
   - Save as `evaluation/t2_test_vectors.json`

2. **Write Dart tokenizer spec** — document exact tokenization steps:
   - Load `vocab.txt` from TFLite metadata
   - Lowercase → WordPiece tokenize → pad/truncate to 128 → convert to `Int32List`
   - Provide 3 worked examples with expected token IDs

3. **Write IO contract** — update `model/tflite/IO_SCHEMA.md`:
   - Input tensor names, shapes, dtypes (4 × `int32 [1,128]`)
   - Output tensor names, shapes, dtypes (3 outputs)
   - Post-processing: multiply `ats_score` × 100, argmax `domain_probs`, argmax `rsg_template`

4. **Latency estimate** — run TFLite inference 100 times on CPU, report:
   - Mean, p50, p95, p99 latency in ms
   - Assert: mean < 500 ms

### Output Files
- `evaluation/t2_test_vectors.json`
- `evaluation/t2_integration_report.json`
- `model/tflite/IO_SCHEMA.md` (updated)

### Script Path
`scripts/t2_flutter_validation.py`

### Definition of Done
- [ ] 20-sample test vectors JSON saved
- [ ] IO_SCHEMA.md updated with MiniLM contract
- [ ] Mean latency < 500 ms
- [ ] Print `T2 COMPLETE — model ready for Flutter integration`

---

## Quick Reference — Stage Sequence

```
E0 (Encoder Validation)
 └→ E1 (Architecture Update)
     └→ R0 (Tokenization Pipeline)
         └→ R1 (Heads Warm-up)         encoder frozen, lr=1e-3
             └→ R2 (Joint Fine-tune)    encoder unfrozen, lr=2e-5
                 └→ R3 (RSG Boost)      encoder frozen, RSG weight=0.60
                     └→ R4 (Final Polish) all unfrozen, lr=1e-5
                         └→ T1 (INT8 Export)
                             └→ T2 (Flutter Integration)
```

## How to Use

```
1. Copy the relevant INJECTION block
2. Paste as system/context prompt before giving the task
3. Say: "Execute INJECTION-E0" (or whichever stage)
4. Agent will produce the script, run it, and report pass/fail
5. Only proceed to next stage if current stage passes all gates
```
