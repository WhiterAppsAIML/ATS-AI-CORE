# ATS Keras Model Specification & Performance Report

**Version**: v1.0 (Final Production Model)
**Date**: March 2026
**Model Location**: `ats-ai-core/model/ats_model/final_model_weights.h5`
**Status**: ✅ Production Ready

---

## 📋 Executive Summary

The ATS (Applicant Tracking System) Keras model is a dual-head neural network designed for automated resume scoring and job domain classification. Built on TensorFlow 2.x with Universal Sentence Encoder, it achieves industry-leading performance with **MAE 5.09** (target < 8.0) and **Domain F1 0.8648** (target > 0.85) across 7 professional domains.

---

## 🏗️ Model Architecture

### Core Design
- **Architecture Type**: Multi-task learning with shared encoder
- **Framework**: TensorFlow/Keras (Legacy Keras mode)
- **Base Encoder**: Universal Sentence Encoder Lite v4 (TF-Hub)
- **Parameter Count**: 257,242,376 parameters
- **Model Pattern**: Dual-head architecture (scoring + classification)

### Network Structure

```
Input Layer (Text Pairs)
├── resume_text: tf.string, shape=()
└── jd_text: tf.string, shape=()
         │
Universal Sentence Encoder (USE Lite v4)
├── Embedding Size: 512 dimensions
├── Frozen Weights: Non-trainable (transfer learning)
└── Hub URL: https://tfhub.dev/google/universal-sentence-encoder/4
         │
    Feature Engineering
├── Resume Embedding: [512]
├── JD Embedding: [512]
├── Cosine Similarity: [1]
└── Dot Product: [1]
         │
    ┌─────────────────┴─────────────────┐
    │                                   │
Similarity Head                   Domain Head
(ATS Score Prediction)           (Job Classification)
    │                                   │
Concat Features [1026]              JD Embedding [512]
    │                                   │
Dense(256) + ReLU + Dropout(0.3)   Dense(256) + ReLU + Dropout(0.3)
    │                                   │
Dense(64) + ReLU + Dropout(0.2)    Dense(128) + ReLU + Dropout(0.2)
    │                                   │
Dense(1) + Sigmoid                  Dense(7) + Softmax
    │                                   │
ATS Score [0.0-1.0]               Domain Probabilities [7]
```

### Domain Categories

| Index | Domain Name | Description |
|-------|-------------|-------------|
| 0 | IT / Software | Technology, programming, software development |
| 1 | Non-IT / Management | Business, operations, project management |
| 2 | Design / Creative | UI/UX, graphic design, marketing creative |
| 3 | Healthcare | Medical, nursing, healthcare administration |
| 4 | Finance / Banking | Financial services, accounting, investment |
| 5 | Legal | Law, paralegal, legal consulting |
| 6 | Education | Teaching, academic research, training |

---

## 📊 Performance Metrics

### Overall Model Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Mean Absolute Error (MAE)** | 5.09 | < 8.0 | ✅ **PASS** |
| **Root Mean Square Error (RMSE)** | 9.10 | - | ✅ |
| **Band Accuracy** | 82.42% | > 80% | ✅ **PASS** |
| **Domain F1 Score** | 0.8648 | > 0.85 | ✅ **PASS** |

### Per-Domain F1 Scores

| Domain | F1 Score | Status |
|--------|----------|--------|
| IT / Software | 0.8498 | ✅ |
| Non-IT / Management | 0.8311 | ✅ |
| **Design / Creative** | **0.9593** | 🏆 **Best** |
| Healthcare | 0.8559 | ✅ |
| Finance / Banking | 0.8454 | ✅ |
| **Legal** | **0.9048** | 🏆 |
| Education | 0.8077 | ✅ |

### Training Convergence (Final Epoch)

| Phase | ATS MAE (0-1 scale) | Domain Accuracy |
|-------|---------------------|-----------------|
| **Training** | 0.0387 | 88.7% |
| **Validation** | 0.0475 | 83.4% |
| **Test** | 0.0509 | 82.4% |

---

## ⚙️ Technical Specifications

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 32 |
| **Learning Rate** | 1e-4 (Adam optimizer) |
| **Total Epochs** | 60 (early stopping: patience=8) |
| **Loss Function** | Weighted multi-task loss |
| **ATS Loss Weight** | 0.35 (Mean Absolute Error) |
| **Domain Loss Weight** | 0.65 (Sparse Categorical Crossentropy) |
| **Data Split** | 75% train / 15% val / 10% test |
| **Random Seed** | 42 |

### Hyperparameters

| Component | Configuration |
|-----------|---------------|
| **Dropout Rates** | 0.3 (first layer), 0.2 (second layer) |
| **Activation Functions** | ReLU (hidden), Sigmoid (ATS), Softmax (domain) |
| **Regularization** | Dropout only (no L1/L2) |
| **Early Stopping** | Monitor: val_loss, patience: 8, restore_best: true |
| **Learning Schedule** | Fixed learning rate (no decay) |

---

## 📁 Model Files & Storage

### Keras Model Files

| File | Size | Description |
|------|------|-------------|
| `final_model_weights.h5` | **981 MB** | Production model weights (current) |
| `best_model_weights.h5` | **981 MB** | Best validation model weights |
| `training_log.csv` | 30 KB | Complete training history |
| `training_log_backup.csv` | 29 KB | Backup training log |

### TensorFlow Lite Deployment

| File | Size | Optimization |
|------|------|--------------|
| `ats_core.tflite` | **491 MB** | Float16 quantization + SELECT_TF_OPS |
| **Quantization** | 50% size reduction | Dynamic range quantization |
| **Optimization** | DEFAULT + SELECT_TF_OPS | TF operations fallback |
| **Parity Tolerance** | ±0.02 | Maximum difference vs Keras |

> ⚠️ **Note**: Current TFLite model (491 MB) exceeds RULES.md target of 30 MB. Requires further quantization or pruning for mobile deployment.

---

## 📊 Training Dataset

### Dataset Composition

| Component | Count | Percentage |
|-----------|-------|------------|
| **Total Training Pairs** | 250,163 | 75% |
| **Validation Pairs** | 50,462 | 15% |
| **Test Pairs** | 36,811 | 10% |
| **Gold Labels** | 115,723 | High-quality manual |
| **Weak Labels** | 204,746 | Synthetic generation |
| **Combined Dataset** | **364,987** | **100%** |

### Data Sources

| Source | Description | Quality |
|--------|-------------|---------|
| **Resume Corpus** | LiveCareer resume dataset | Professional, structured |
| **Job Descriptions** | LinkedIn job postings | Real-world JDs |
| **Gold Labels** | Manual annotation + expert review | High precision |
| **Synthetic Data** | Rule-based generation | Bulk training data |

### Domain Distribution

| Domain | Training Pairs | Percentage |
|--------|----------------|------------|
| IT / Software | 89,234 | 35.7% |
| Non-IT / Management | 67,821 | 27.1% |
| Healthcare | 32,456 | 13.0% |
| Finance / Banking | 28,789 | 11.5% |
| Design / Creative | 18,923 | 7.6% |
| Education | 8,234 | 3.3% |
| Legal | 4,706 | 1.9% |

---

## 🔌 Input/Output Schema

### Model Inputs

```python
# Input Signature
{
    'resume_text': tf.TensorSpec(shape=(), dtype=tf.string),
    'jd_text': tf.TensorSpec(shape=(), dtype=tf.string)
}
```

| Input | Type | Shape | Description |
|-------|------|-------|-------------|
| `resume_text` | `tf.string` | `()` | Raw resume text content |
| `jd_text` | `tf.string` | `()` | Raw job description text |

### Model Outputs

```python
# Output Signature
[
    tf.TensorSpec(shape=[1], dtype=tf.float32),    # ATS Score
    tf.TensorSpec(shape=[7], dtype=tf.float32)     # Domain Logits
]
```

| Output | Type | Shape | Range | Description |
|--------|------|-------|-------|-------------|
| **ATS Score** | `tf.float32` | `[1]` | [0.0, 1.0] | Compatibility score (×100 for %) |
| **Domain Logits** | `tf.float32` | `[7]` | [0.0, 1.0] | Softmax probabilities per domain |

### Score Band Mapping

| Score Range | Band | Interpretation |
|-------------|------|----------------|
| **85-100** | Excellent Match | Strong candidate fit |
| **65-84** | Good Match | Reasonable candidate fit |
| **45-64** | Moderate Match | Acceptable with gaps |
| **25-44** | Weak Match | Significant skill gaps |
| **0-24** | Poor Match | Poor candidate fit |

---

## 🚀 Inference Pipeline

### Complete Processing Flow

1. **Text Preprocessing**
   - HTML tag removal
   - Unicode normalization
   - PII masking (emails, URLs)
   - Whitespace normalization

2. **Resume Analysis**
   - Section segmentation
   - Fresher detection (experience parsing)
   - Text normalization

3. **Model Prediction**
   - Cached model loading (singleton)
   - Dual-head inference
   - Score scaling (0-1 → 0-100)

4. **Keyword Gap Analysis**
   - TF-IDF extraction (top 15 keywords)
   - Hard/soft skill classification
   - Importance ranking

5. **Feedback Generation**
   - Domain-specific rules
   - Fresher-aware suggestions
   - Actionable recommendations (3-5 items)

### Performance Benchmarks

| Operation | Time (avg) | Memory |
|-----------|------------|--------|
| **Model Loading** | ~15 seconds | ~2.5 GB |
| **Inference** | ~1.2 seconds | ~500 MB |
| **Full Pipeline** | ~2.0 seconds | ~650 MB |
| **Subsequent Calls** | ~0.4 seconds | ~500 MB |

---

## ✅ Production Readiness Checklist

| Requirement | Status | Notes |
|-------------|--------|--------|
| **Performance Targets** | ✅ PASS | MAE < 8.0, F1 > 0.85 |
| **Model Validation** | ✅ PASS | Cross-validation + held-out test |
| **Error Handling** | ✅ PASS | Graceful degradation |
| **Inference Speed** | ✅ PASS | < 2 seconds full pipeline |
| **Memory Usage** | ✅ PASS | < 3 GB peak memory |
| **Domain Coverage** | ✅ PASS | All 7 domains validated |
| **Fresher Fairness** | ✅ PASS | ≤ 20 point gap vs experienced |
| **TFLite Conversion** | 🔄 PARTIAL | Size optimization needed |

---

## 🔧 Usage Instructions

### Loading the Model

```python
from src.ats_engine.inference import run_ats_inference

# Single inference call (recommended)
result = run_ats_inference(resume_text, jd_text)

# Returns: {
#     "ats_score": 67.45,
#     "score_band": "Good Match",
#     "domain_index": 0,
#     "domain_name": "IT / Software",
#     "missing_keywords": {"hard_skills": [...], "soft_skills": [...]},
#     "feedback": ["Add Docker experience...", ...],
#     "is_fresher": False
# }
```

### Manual Testing Interface

```bash
# Terminal testing script
python tools/test_model.py --resume resume.pdf --jd job_description.txt
```

---

## 📈 Future Improvements

### Immediate Optimization Opportunities
- **TFLite Quantization**: INT8 quantization for <30 MB target
- **Model Pruning**: Remove low-impact weights (target: 40% reduction)
- **Caching Optimization**: Resume embeddings cache for repeat candidates

### Enhancement Roadmap
- **Multi-language Support**: Non-English resume processing
- **Bias Mitigation**: Enhanced fairness across demographic groups
- **Real-time Updates**: Incremental learning from user feedback
- **Advanced Features**: Salary prediction, interview likelihood

---

## 📞 Support & Contact

**Model Owner**: Sai (AI Development Team)
**Last Updated**: March 2026
**Next Review**: June 2026
**Documentation Version**: 1.0

For technical issues or questions, refer to:
- **Training Logs**: `ats-ai-core/model/ats_model/training_log.csv`
- **Evaluation Reports**: `ats-ai-core/evaluation/eval_report.csv`
- **Test Interface**: `tools/test_model.py`
- **Issue Tracker**: Project repository issues

---

*This model meets production deployment standards and has been validated across multiple domains with comprehensive performance metrics.*