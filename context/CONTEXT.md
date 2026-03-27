# CONTEXT.md — Project Context for IDE Agent (Claude Opus)

> This file is the primary context document loaded into the AntiGravity IDE agent.
> Read this file completely before acting on any task in this project.

---

## Who You Are Working With

You are assisting **Sai**, an AI Engineering Intern on a development team building a mobile resume analysis application. Sai's responsibility is the **AI core** of the application — specifically the **ATS Scoring Engine**.

The application is built by a larger team:
- **Sai (you are assisting)** → TensorFlow model, TFLite conversion, keyword gap module, feedback engine
- **Flutter team** → Mobile UI, Firebase integration, tflite_flutter runtime calls
- **Backend team** → Resume Summary Generator (separate model, separate scope)

---

## What This Project Does

The app has two AI-powered features:

**Feature 1 — ATS Scoring Engine (Sai's responsibility)**
A user uploads their resume and pastes a job description. The AI scores how well the resume matches the JD (0–100), identifies missing keywords, and generates actionable feedback. The score is designed to be fair to freshers — someone with no work experience but strong skill alignment can score as high as an experienced candidate.

**Feature 2 — Resume Summary Generator (NOT Sai's responsibility)**
The user fills in a resume builder form. The AI generates a 3–4 line professional summary. This is handled by another team member. Do not generate code for this feature.

---

## The Tech Stack (Sai's Portion)

| Layer | Technology |
|-------|-----------|
| Model framework | TensorFlow 2.x / Keras |
| Text encoder | Universal Sentence Encoder Lite (TF Hub) |
| Keyword extraction | scikit-learn TF-IDF (training/inference pre-processing only) |
| Feedback engine | Rule-based JSON mapper (no ML) |
| Deployment format | TensorFlow Lite (.tflite), Float16 quantized |
| Mobile runtime | Flutter via `tflite_flutter` package (Flutter team handles this) |
| Cloud | Firebase Storage (Flutter team handles deployment) |

---

## The Core Architecture (ATS Engine)

The ATS engine has two types of components:

**Type A — Neural Model Outputs (inside the TFLite file)**
1. ATS Score: float32 in [0.0, 1.0] — multiply by 100 for display
2. Domain Label: int32 in [0–6] — maps to IT, Non-IT, Design, Healthcare, Finance, Legal, Education

**Type B — Post-Processing (Python/Dart code, NOT inside TFLite)**
3. Missing Keywords: computed by TF-IDF keyword gap module after inference
4. Feedback List: computed by rule-based feedback mapper using score + domain

This separation is intentional and must not be changed. The TFLite model stays small by not embedding feedback logic.

---

## The Scoring Rubric — Key Facts

The score is built from 5 dimensions, each weighted differently per domain:

| Dimension | IT | Non-IT | Design | Healthcare | Finance | Legal | Education |
|-----------|-----|--------|--------|------------|---------|-------|-----------|
| Skill Alignment | 35% | 20% | 30% | 25% | 25% | 20% | 20% |
| Semantic Contextual Fit | 25% | 25% | 20% | 20% | 20% | 30% | 25% |
| Keyword Coverage | 20% | 20% | 15% | 20% | 20% | 25% | 20% |
| Structural Completeness | 10% | 15% | 15% | 15% | 15% | 15% | 20% |
| Achievement & Impact | 10% | 20% | 20% | 20% | 20% | 10% | 15% |

**Critical fresher rule:** Projects, internships, and academic work are treated equivalent to professional work experience. The model never hard-penalizes for zero years of experience.

---

## Score Band Meanings (for feedback mapper)

| Score | Band |
|-------|------|
| 85–100 | Excellent Match |
| 65–84 | Good Match |
| 45–64 | Moderate Match |
| 25–44 | Weak Match |
| 0–24 | Poor Match |

---

## Project Folder Structure (Quick Reference)

```
ats-ai-core/
├── ARCHITECTURE.md     ← Full design doc (read first)
├── PLAN.md             ← Sprint plan
├── RULES.md            ← Coding rules
├── CONTEXT.md          ← This file
│
├── data/
│   ├── raw/            ← Kaggle datasets (not committed to git)
│   ├── processed/      ← Cleaned data (JSON/CSV)
│   ├── labeled/        ← Weak + gold labels
│   └── synthetic/      ← Generated pairs
│
├── model/
│   ├── ats_model/      ← Saved Keras model
│   └── tflite/
│       ├── ats_core.tflite   ← Final export (<30MB)
│       └── IO_SCHEMA.md      ← Tensor contract for Flutter team
│
├── src/
│   ├── preprocessing/  ← text_cleaner.py, section_segmenter.py, normalizer.py
│   ├── encoding/       ← use_lite_encoder.py
│   ├── ats_engine/     ← model.py, trainer.py, rubric_layer.py
│   ├── keyword_gap/    ← extractor.py, classifier.py
│   ├── feedback/       ← feedback_mapper.py
│   └── conversion/     ← convert_to_tflite.py
│
├── rubrics/
│   ├── domain_weights.json
│   ├── feedback_rules.json
│   └── summary_templates.json  ← NOT Sai's responsibility
│
├── evaluation/
│   └── ats_eval.py
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_label_generation.ipynb
    ├── 03_ats_model_training.ipynb
    ├── 04_summary_model_training.ipynb  ← NOT Sai's responsibility
    └── 05_tflite_conversion.ipynb
```

---

## Datasets Being Used

All datasets are from Kaggle:

| Dataset | Use |
|---------|-----|
| Resume Dataset (2484 resumes, multi-domain) | Resume text corpus |
| LinkedIn Job Postings 2023 | JD text corpus |
| Monster.com Job Listings | JD diversity |
| UpdatedResumeDataSet | Additional resume diversity |
| Synthetic resume-JD pairs (generated) | Labeled training pairs |

Labels are generated in two phases:
1. **Weak labels**: TF-IDF cosine similarity + keyword overlap heuristic
2. **Gold labels**: 300–500 human-annotated pairs for fine-tuning

---

## Key Constraints to Always Remember

1. **No server inference** — All inference is on-device via TFLite. No API calls at runtime.
2. **No LLMs** — No GPT, no LLaMA, no generative models. Hardware constraints.
3. **30MB limit** — The full `.tflite` file must stay under 30MB after Float16 quantization.
4. **Single TFLite artifact** — Both ATS engine and summary generator are in one file (though summary generator is built by another team member).
5. **JSON-serializable outputs** — Everything the Flutter team receives must serialize cleanly to JSON.
6. **Feedback is deterministic** — Rule-based only. No generated text. Predictable, testable, safe.

---

## How to Approach a Task

When Sai asks you to implement something:

1. **Identify which sprint it belongs to** (see PLAN.md)
2. **Check RULES.md** for any relevant constraint before writing code
3. **Check ARCHITECTURE.md** Section 3.2 for pipeline position
4. **Write one complete file** with full docstrings and type hints
5. **Include unit tests** in a mirrored `tests/` path
6. **If the task is ambiguous**, state your assumption clearly before proceeding

---

## Common Tasks and Where to Start

| Task | Start Here |
|------|-----------|
| "Build the text cleaner" | `src/preprocessing/text_cleaner.py`, Sprint 2 |
| "Set up the encoder" | `src/encoding/use_lite_encoder.py`, Sprint 4 |
| "Build the ATS model" | `src/ats_engine/model.py`, Sprint 5 |
| "Build training script" | `src/ats_engine/trainer.py`, Sprint 6 |
| "Build keyword extractor" | `src/keyword_gap/extractor.py`, Sprint 7 |
| "Build feedback mapper" | `src/feedback/feedback_mapper.py`, Sprint 8 |
| "Convert to TFLite" | `src/conversion/convert_to_tflite.py`, Sprint 9 |
| "Write domain weights JSON" | `rubrics/domain_weights.json`, Sprint 3 |
| "Write feedback rules JSON" | `rubrics/feedback_rules.json`, Sprint 8 |

---

## What Success Looks Like

At the end of the internship, Sai will hand off:
- `model/tflite/ats_core.tflite` — the packaged TFLite model
- `model/tflite/IO_SCHEMA.md` — the tensor I/O contract
- `src/keyword_gap/` — standalone keyword gap module
- `src/feedback/` — standalone feedback mapper
- `evaluation/ats_eval.py` + eval report

The Flutter team should be able to pick these up and integrate without any follow-up questions from Sai.
