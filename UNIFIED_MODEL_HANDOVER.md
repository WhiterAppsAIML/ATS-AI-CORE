# Unified Model (ATS+RSG) Handover Document

## 1. Overview
The Unified Model integrates ATS scoring, Domain Classification, and RSG (Resume Section Generation) heads into a single multi-task learning architecture. The primary trained Keras weights file (`best_unified_weights.h5`) exceeds 1 GB (**1.03 GB**). 

This document outlines the training process, performance metrics, and pipeline details to serve as a starting point for subsequent work.

## 2. Training Process & Pipeline Stages
The model was trained and refined across several systematic stages:

- **Stage B-1: USE Lite v2 Encoder Swap** 
  Integrated a 512-dim embedding output to ensure TF Hub compatibility and resolve earlier TFLite conversion roadblocks.
- **Stage B-2: ATS + Domain Head Tuning** 
  Optimized the regression (ATS) and classification (Domain) branches.
- **Stage B-3: RSG Surgical Recovery & Stabilization** 
  Resolved overfitting issues, applied data augmentation, and achieved the 65.5% accuracy gate target, eventually scaling to **>92%** during full evaluation.
- **Stage B-4: TFLite Conversion** 
  Successfully converted the architecture to a heads-only, Float16 TFLite model.

### Training Convergence
The unified training log demonstrates a robust convergence over 13 epochs:
- **Final Validation ATS MAE:** 4.10
- **Final Validation Domain Acc:** 87.3%
- **Final Validation RSG Acc:** 92.07%
- **Final Validation Loss:** 0.1848
- **Final Train ATS MAE:** 4.87
- **Final Train Domain Loss:** 0.41

## 3. Performance Metrics (Final Evaluation)

### 📊 ATS Head
- **MAE:** 4.152
- **RMSE:** 7.8329
- **Band Accuracy:** 85.9%
- **Fresher Fairness Gap:** 6.84 points

### 🎯 Domain Head
- **Macro F1 Score:** 0.8823
- **Per-Domain F1 Scores:** 
  - Design: 0.9615 🏆 (Best)
  - Legal: 0.9255
  - Healthcare: 0.9023
  - IT / Software: 0.8759
  - Finance: 0.8556
  - Non-IT / Management: 0.8330
  - Education: 0.8223

### 📝 RSG Head
- **Overall Accuracy:** 92.05%
- Performance remains exceptionally solid across the supported resume section templates, with most sub-templates achieving 100% accuracy.

## 4. TFLite Deployment Details
Due to the `USELiteEncoder` relying on `tf.py_function` (which is incompatible with TFLite), the model was converted to a **heads-only architecture**. 

- **Input Spec:** Pre-computed 512-dim embeddings for Resume and JD (`float32[batch, 512]`).
- **Output Spec:** 
  - `ats_score`: `float32[batch, 1]` (sigmoid, 0-1)
  - `domain_probs`: `float32[batch, 7]` (softmax)
  - `rsg_template`: `float32[batch, 46]` (softmax)
- **Quantization:** Float16
- **Final File Size:** **1.8 MB**
- **Numerical Parity:** Passed with a maximum difference of **0.0026 points** compared to the Keras baseline.

## 5. Next Steps
- The heavy **1.03 GB** `best_unified_weights.h5` model handles the full Keras E2E pipeline and serves as the source of truth.
- For mobile/Flutter integration, use the **1.8 MB** `unified_model_lite_v2_float16.tflite` file.
- **Inference Requirement:** The client application must handle SentencePiece tokenization and USE Lite v2 encoding to generate the 512-dim embeddings. These embeddings are then passed to the TFLite heads.
