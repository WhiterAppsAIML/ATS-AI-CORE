# Edge Deployment Plan (Simplified) — ATS + RSG Unified Model v2

## Goal
Ship one on-device `.tflite` model (under 30 MB) that takes resume + JD text and returns:
- ATS score
- Domain probabilities (7 classes)
- RSG template probabilities (46 classes)

## Why We Changed Direction
String-input encoders like USE/MobileUSE require Flex ops (`StaticRegexReplace`, etc.), which breaks standard TFLite-only deployment.

So we split responsibilities:
- **Inside TFLite:** token IDs -> MiniLM encoder -> 3 heads
- **Outside graph (but still single file UX):** tokenizer via metadata (`vocab.txt` embedded in `.tflite`)

## Chosen Design
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding size: `384`
- Sequence length: `128`
- Inputs (4):
  - `resume_input_ids [1,128]`
  - `resume_attention_mask [1,128]`
  - `jd_input_ids [1,128]`
  - `jd_attention_mask [1,128]`
- Outputs (3):
  - `ats_score [1,1]`
  - `domain_probs [1,7]`
  - `rsg_template [1,46]`

## Non-Negotiable Gates
- No `SELECT_TF_OPS`
- No Flex ops
- Final `.tflite` size `< 30 MB`
- Keras vs TFLite ATS parity: max diff `< 2.0` points
- Flutter API must remain: `predict(String resumeText, String jdText)`

## Execution Plan

### E0 — Encoder Validation (2 days)
- Convert MiniLM to INT8 TFLite with `TFLITE_BUILTINS_INT8` only
- Check no Flex ops and encoder size `< 25 MB`
- Compare embedding ranking vs prior USE baseline (Spearman `> 0.85`)
- Verify tokenizer fidelity between Python and Dart (100% on sample set)

Exit rule: proceed only if all checks pass.

### E1 — Model Architecture Update (1 day)
- Replace USE layer with MiniLM submodel in `unified_model.py`
- Update config (`EMBEDDING_DIM=384`, `MAX_SEQ_LEN=128`)
- Rewire heads for new dimensions (ATS concat = 770)
- Dry-run INT8 conversion and verify tensor shapes

Exit rule: conversion succeeds, no Flex ops, output shapes correct.

### R0 — Tokenization Pipeline (1 day)
- Pre-tokenize ATS + RSG datasets to `.npz`
- Save `input_ids` + `attention_mask`
- Validate sequence-length distribution and label integrity

Exit rule: tokenized files complete and checks pass.

### R1 — Heads Warm-up (2–3 days)
- Freeze encoder
- Train heads only
- Target: ATS MAE `< 10.0`, Domain F1 `> 0.72`, RSG Acc `> 0.42`

### R2 — Joint Fine-tuning (2–3 days)
- Unfreeze encoder, very low LR
- Train end-to-end
- Target: ATS val/test MAE `< 8.0`, Domain F1 `> 0.82`, RSG Acc `> 0.55`

### R3 — RSG Boost Pass (1–2 days)
- Freeze encoder, upweight RSG temporarily
- Target: RSG Acc `> 0.62` without ATS regression beyond MAE `8.5`

### R4 — Final Joint Polish (1–2 days)
- Restore canonical loss weights
- Final hard gates:
  - ATS val MAE `< 6.5`
  - ATS test MAE `< 7.0`
  - Domain F1 `> 0.85` (all domains `> 0.80`)
  - RSG val Acc `> 0.65`
  - Fresher fairness gap `<= 20`

### T1 — INT8 Export + Metadata (1 day)
- Export full model INT8 (`TFLITE_BUILTINS_INT8` only)
- Embed `vocab.txt` + tokenizer metadata into `.tflite`
- Run parity checks (Keras vs TFLite)
- Ensure final model size `< 30 MB`

### T2 — Flutter Integration (1–2 days)
- Load single `.tflite` in app
- Read tokenizer metadata at runtime
- Verify 20-sample parity on device
- Target inference latency `< 500 ms`

## Key Deliverables
- `e0_encoder_report.json`
- `e1_architecture_validation.json`
- `tokenized_*.npz`
- `r1/r2/r3/r4` best weight files + logs
- `ats_unified_minilm_int8.tflite`
- `conversion_summary_minilm.json`
- `t2_flutter_integration_report.json`

## Expected Outcome
A single-file, Flutter-friendly, INT8 TFLite model around 23–25 MB with no Flex dependency and production-ready ATS/Domain/RSG outputs.
