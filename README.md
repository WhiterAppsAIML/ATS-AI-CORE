# ATS AI Core — Resume Scoring Engine

> **AI-powered resume-to-job-description alignment scoring engine built with TensorFlow/Keras and Universal Sentence Encoder.**

[![Model Status](https://img.shields.io/badge/Model-Production%20Ready-brightgreen)](.)
[![TF Version](https://img.shields.io/badge/TensorFlow-2.15-orange)](.)
[![Domain F1](https://img.shields.io/badge/Domain%20F1-0.8648-blue)](.)
[![MAE](https://img.shields.io/badge/MAE-5.09-blue)](.)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](.)

---

## What Is This?

The **ATS Scoring Engine** is the AI core of a mobile resume analysis application. It scores how well a resume aligns with a given job description (0–100 scale), identifies missing keywords, classifies the job domain, and generates actionable feedback — all in under 2 seconds.

It is designed as part of a larger system that includes a **Flutter mobile app** and a **Firebase deployment layer**. This repository contains only the AI/ML engine.

A key design principle: **fresher candidates (no work experience) are not penalized relative to experienced candidates.** Scoring is based on relative alignment to the job description, not absolute experience level.

---

## Model Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Mean Absolute Error (MAE) | 5.09 | < 8.0 | ✅ PASS |
| Band Accuracy | 82.42% | > 80% | ✅ PASS |
| Domain F1 Score | 0.8648 | > 0.85 | ✅ PASS |
| RMSE | 9.10 | — | ✅ |

### Per-Domain F1 Scores

| Domain | F1 Score |
|--------|----------|
| IT / Software | 0.8498 |
| Non-IT / Management | 0.8311 |
| Design / Creative | **0.9593** 🏆 |
| Healthcare | 0.8559 |
| Finance / Banking | 0.8454 |
| Legal | **0.9048** 🏆 |
| Education | 0.8077 |

---

## Architecture Overview

The model is a **dual-head neural network** built on TensorFlow/Keras with Universal Sentence Encoder Lite v4 as the shared encoder.

```
Input: (resume_text, jd_text)
          │
Universal Sentence Encoder Lite v4
├── Resume Embedding [512]
├── JD Embedding [512]
├── Cosine Similarity [1]
└── Dot Product [1]
          │
    ┌─────┴─────┐
    │           │
Score Head   Domain Head
    │           │
Dense(256)   Dense(256)
Dense(64)    Dense(128)
Sigmoid      Softmax(7)
    │           │
ATS Score    Domain Label
[0.0–1.0]   [0–6]
```

**Multi-task learning** — both heads train simultaneously on a weighted loss:
- ATS Score loss weight: `0.35` (MAE)
- Domain classification loss weight: `0.65` (Sparse Categorical Crossentropy)

---

## Supported Domains

| Index | Domain |
|-------|--------|
| 0 | IT / Software |
| 1 | Non-IT / Management |
| 2 | Design / Creative |
| 3 | Healthcare |
| 4 | Finance / Banking |
| 5 | Legal |
| 6 | Education |

---

## Project Structure

```
ats-ai-core/
├── src/
│   ├── ats_engine/         # Core model, trainer, inference, rubric layer
│   ├── preprocessing/      # Text cleaning, normalization, section segmentation
│   ├── keyword_gap/        # TF-IDF keyword extraction and skill classification
│   ├── feedback/           # Domain-specific feedback generation
│   ├── encoding/           # USE Lite encoder wrapper
│   └── conversion/         # TFLite conversion pipeline
├── model/
│   ├── ats_model/          # Keras weights + training logs (weights excluded from repo)
│   └── tflite/             # TFLite binary (excluded from repo)
├── data/
│   └── labeled/            # Training datasets (excluded from repo — see below)
├── rubrics/                # JSON configs: domain weights, feedback rules, keyword categories
├── scripts/                # Data pipeline and training utility scripts
├── evaluation/             # Evaluation script and reports
├── tests/                  # Unit tests for all modules
├── tools/                  # Manual testing interface
├── train.py                # Main training entry point
├── realtime_demo.py        # Live demo script
└── requirements.txt        # Python dependencies
```

> ⚠️ **Large files are excluded from this repo** (see below). You will need to obtain them separately.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/WhiterAppsAIML/ATS-AI-CORE.git
cd ATS-AI-CORE
```

### 2. Set up Python environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Obtain model weights

The trained Keras model weights (`final_model_weights.h5`, ~981 MB) and TFLite binary (`ats_core.tflite`, ~491 MB) are **not included in this repo** due to GitHub file size limits.

Contact the model owner (see Support section) or download from the shared team storage and place them at:

```
model/ats_model/final_model_weights.h5
model/ats_model/best_model_weights.h5
model/tflite/ats_core.tflite
```

### 4. Run inference

```python
from src.ats_engine.inference import run_ats_inference

result = run_ats_inference(resume_text, jd_text)

# Example output:
# {
#     "ats_score": 67.45,
#     "score_band": "Good Match",
#     "domain_index": 0,
#     "domain_name": "IT / Software",
#     "missing_keywords": {
#         "hard_skills": ["Docker", "Kubernetes"],
#         "soft_skills": ["leadership"]
#     },
#     "feedback": ["Add Docker experience to your skills section...", ...],
#     "is_fresher": False
# }
```

### 5. Manual test via CLI

```bash
python tools/test_model.py --resume resume.pdf --jd job_description.txt
```

---

## Score Band Reference

| Score Range | Band | Meaning |
|-------------|------|---------|
| 85 – 100 | Excellent Match | Strong candidate fit |
| 65 – 84 | Good Match | Reasonable fit |
| 45 – 64 | Moderate Match | Acceptable with gaps |
| 25 – 44 | Weak Match | Significant skill gaps |
| 0 – 24 | Poor Match | Poor alignment |

---

## Inference Pipeline

Each call to `run_ats_inference()` runs the following steps automatically:

1. **Text Preprocessing** — HTML removal, Unicode normalization, PII masking, whitespace cleanup
2. **Resume Analysis** — Section segmentation, fresher detection, text normalization
3. **Model Prediction** — Dual-head inference, score scaling (0–1 → 0–100)
4. **Keyword Gap Analysis** — TF-IDF extraction of top 15 keywords, hard/soft skill classification
5. **Feedback Generation** — Domain-specific, fresher-aware, actionable recommendations

### Performance Benchmarks

| Operation | Avg Time | Memory |
|-----------|----------|--------|
| Model loading (first call) | ~15 sec | ~2.5 GB |
| Inference | ~1.2 sec | ~500 MB |
| Full pipeline | ~2.0 sec | ~650 MB |
| Subsequent calls | ~0.4 sec | ~500 MB |

---

## Training

To retrain the model from scratch (requires training data):

```bash
python train.py
```

Training configuration is defined in `src/config.py`. Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Learning rate | 1e-4 (Adam) |
| Max epochs | 60 (early stopping, patience=8) |
| ATS loss weight | 0.35 |
| Domain loss weight | 0.65 |
| Data split | 75% train / 15% val / 10% test |

Training logs are saved to `model/ats_model/training_log.csv`.

---

## What's Excluded From This Repo

These files are gitignored and must be obtained separately:

| Path | Reason |
|------|--------|
| `data/labeled/*.csv` | Large datasets (100K+ rows) |
| `model/ats_model/*.h5` | Keras weights (~981 MB each) |
| `model/tflite/*.tflite` | TFLite binary (~491 MB) |
| `final_venv/` | Python virtual environment |
| `tfhub_cache/` | Downloaded TF Hub model cache |

---

## Running Tests

```bash
pytest tests/ -v
```

Individual module tests:
```bash
pytest tests/preprocessing/ -v
pytest tests/ats_engine/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow 2.15 / Keras |
| Base Encoder | Universal Sentence Encoder Lite v4 (TF-Hub) |
| Keyword Extraction | scikit-learn TF-IDF |
| Mobile Deployment | TensorFlow Lite |
| App Integration | Flutter + Firebase |
| Language | Python 3.10+ |

---

## Roadmap

- [ ] INT8 quantization for TFLite model (<30 MB target)
- [ ] Model pruning (target: 40% size reduction)
- [ ] Resume embedding cache for repeat candidates
- [ ] Multi-language resume support
- [ ] Unification with RSG (Resume Summary Generator) module

---

## Support

**Model Owner:** Sai — AI Engineering Intern, AIML Team
**Team:** WhiterApps AIML
**Last Updated:** March 2026

For issues, refer to:
- Training logs: `model/ats_model/training_log.csv`
- Evaluation report: `evaluation/eval_report.csv`
- Test interface: `tools/test_model.py`
- Open an issue in this repository

---

*This engine is part of the WhiterApps ATS mobile application. The Flutter UI and Firebase integration are maintained separately by the app development team.*
