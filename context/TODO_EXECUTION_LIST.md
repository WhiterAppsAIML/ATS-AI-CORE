# TODO_EXECUTION_LIST.md
# ATS AI Core — Task Execution List
# Execute tasks in order. Send the TODO ID to the agent to trigger each task.
# Model: Claude Opus 4.6 | IDE: AntiGravity

---

## HOW TO USE THIS FILE

1. Load `AGENT_PROMPT.md` into the AntiGravity IDE as the system/context prompt
2. Send the TODO ID to execute that task (e.g. type: `Execute T-01`)
3. Agent writes the complete file + tests
4. You copy the output into your project
5. Mark the task `[x]` and move to the next

---

## DATASET REFERENCE (Quick Lookup)

| ID | Dataset | Path After Download |
|---|---|---|
| DS-1 | LiveCareer Resume CSV | `data/raw/resume_dataset/Resume.csv` |
| DS-2 | LinkedIn Job Postings CSV | `data/raw/linkedin_jobs/job_postings.csv` |
| DS-3 | HuggingFace Resume Score Details | `data/raw/resume_score_details/` |

---

## SPRINT 1 — Data Ingestion & Exploration

> **Goal:** Download datasets, understand structure, validate columns, check domain coverage.

- [ ] **T-01** · `src/preprocessing/data_loader.py`
  - Loaders for all 3 datasets (LiveCareer CSV, LinkedIn CSV, HuggingFace parquet)
  - Returns clean DataFrames with unified schema: `resume_text`, `jd_text`, `score`, `domain`
  - Handles missing values, encoding issues, and malformed rows
  - *Tests:* `tests/preprocessing/test_data_loader.py`

- [ ] **T-02** · `notebooks/01_data_exploration.ipynb`
  - Domain distribution bar chart (LiveCareer categories → 7 domain groups)
  - Score distribution histogram (from DS-3)
  - Text length stats for resumes and JDs
  - Null rate table for all three datasets
  - *No test file — notebook only*

---

## SPRINT 2 — Preprocessing Pipeline

> **Goal:** Build reusable, tested text cleaning and section segmentation modules.

- [ ] **T-03** · `src/preprocessing/text_cleaner.py`
  - Strip HTML tags (from `Resume_html` field of LiveCareer)
  - Remove special characters, normalize whitespace, lowercase
  - Remove boilerplate (page numbers, repeated headers)
  - Preserve section headers for segmenter
  - *Tests:* `tests/preprocessing/test_text_cleaner.py`

- [ ] **T-04** · `src/preprocessing/section_segmenter.py`
  - Segment resume text into: `skills`, `education`, `experience`, `projects`, `certifications`, `summary`
  - Regex + keyword-header heuristic approach (no ML)
  - Fresher rule: if `experience` section is empty, promote `projects` as equivalent
  - Returns `dict[str, str]` of section name → section text
  - *Tests:* `tests/preprocessing/test_section_segmenter.py`

- [ ] **T-05** · `src/preprocessing/normalizer.py`
  - Skill synonym normalization (JS → JavaScript, ML → Machine Learning, etc.)
  - Domain category mapper: maps LiveCareer's 25 raw categories to the 7 model domains
  - Seed synonym table included inline as a constant
  - *Tests:* `tests/preprocessing/test_normalizer.py`

- [ ] **T-06** · `src/preprocessing/domain_mapper.py`
  - Maps LiveCareer raw category strings → domain index (0–6)
  - Maps LinkedIn job titles → domain index (0–6) as a fallback
  - Uses keyword-based heuristic for unmapped categories
  - *Tests:* `tests/preprocessing/test_domain_mapper.py`

---

## SPRINT 3 — Label Generation

> **Goal:** Merge all datasets into labeled resume–JD pairs ready for training.

- [ ] **T-07** · `notebooks/02_label_generation.ipynb`
  - Load DS-3 (netsol/resume-score-details) as primary gold labels
  - Load DS-1 + DS-2 and generate weak labels via TF-IDF cosine similarity
  - Apply domain weights from `rubrics/domain_weights.json` to weak label scoring
  - Merge gold labels (DS-3) + weak labels (DS-1 × DS-2 pairs)
  - Save final labeled set to `data/labeled/training_pairs.csv`
  - Columns: `resume_text`, `jd_text`, `score` (0–100), `domain_index` (0–6), `label_source` (gold/weak)
  - *No test file — notebook only*

- [ ] **T-08** · `src/preprocessing/pair_builder.py`
  - Creates resume–JD pairs from DS-1 + DS-2 (same-domain pairing + random negative pairs)
  - Ratio: 70% same-domain, 30% cross-domain (for domain classifier training signal)
  - Caps pairs per domain to prevent imbalance (max 2000 pairs per domain)
  - *Tests:* `tests/preprocessing/test_pair_builder.py`

---

## SPRINT 4 — Encoder Setup

> **Goal:** Wrap USE Lite, validate embedding quality, confirm TFLite compatibility.

- [ ] **T-09** · `src/encoding/use_lite_encoder.py`
  - Load USE Lite from TFHub URL in `src/config.py`
  - `encode(text: str) -> np.ndarray` — single text → 512-dim vector
  - `encode_batch(texts: list[str]) -> np.ndarray` — batched for training efficiency
  - `similarity(text_a: str, text_b: str) -> float` — cosine similarity convenience method
  - Encoder weights frozen by default; `unfreeze()` method available
  - *Tests:* `tests/encoding/test_use_lite_encoder.py`

---

## SPRINT 5 — ATS Model Architecture

> **Goal:** Build the full Keras model with similarity head and domain classifier head.

- [ ] **T-10** · `src/ats_engine/model.py`
  - `build_ats_model() -> tf.keras.Model`
  - Input: two string tensors (`resume_text`, `jd_text`)
  - Shared encoder: USE Lite (frozen initially)
  - Similarity head: cosine similarity → dense → sigmoid → `ats_score` float32 [1]
  - Domain head: dense → softmax (7 classes) → argmax → `domain_label` int32 [1]
  - Total parameter count < 5M
  - Model summary printed on build
  - *Tests:* `tests/ats_engine/test_model.py`

- [ ] **T-11** · `src/ats_engine/rubric_layer.py`
  - Post-inference Python function (NOT a Keras layer)
  - `apply_rubric(raw_score: float, domain_index: int, dimension_scores: dict) -> float`
  - Loads domain weights from `rubrics/domain_weights.json`
  - Validates weights sum to 1.0 on load
  - *Tests:* `tests/ats_engine/test_rubric_layer.py`

---

## SPRINT 6 — Training & Evaluation

> **Goal:** Train the ATS model to target metrics. Validate. Save.

- [ ] **T-12** · `src/ats_engine/trainer.py`
  - `train(model, train_data, val_data) -> tf.keras.callbacks.History`
  - Multi-task loss: MAE (score head, weight 0.7) + CrossEntropy (domain head, weight 0.3)
  - All loss weights and hyperparameters from `src/config.py`
  - Early stopping on val MAE (patience=5)
  - Saves best model to `model/ats_model/`
  - *Tests:* `tests/ats_engine/test_trainer.py`

- [ ] **T-13** · `notebooks/03_ats_model_training.ipynb`
  - Full end-to-end training run using `trainer.py`
  - Load `data/labeled/training_pairs.csv`
  - Train/val/test split (75/15/10) — split BEFORE any feature extraction
  - Plot training curves (loss, MAE per epoch)
  - Print final metrics: MAE, RMSE, domain F1
  - *No test file — notebook only*

- [ ] **T-14** · `evaluation/ats_eval.py`
  - `evaluate_ats_model(model_path: Path, test_csv: Path) -> dict`
  - Computes: MAE, RMSE, score band accuracy, domain classification F1
  - Generates: predicted vs actual scatter plot saved to `evaluation/plots/`
  - Prints pass/fail against target thresholds from `src/config.py`
  - *Tests:* `tests/` → `tests/test_ats_eval.py`

---

## SPRINT 7 — Keyword Gap Module

> **Goal:** Build the TF-IDF-based missing keyword extractor and classifier.

- [ ] **T-15** · `src/keyword_gap/extractor.py`
  - `extract_missing_keywords(resume_text: str, jd_text: str, top_n: int = 20) -> list[dict]`
  - Fit TF-IDF on JD text, extract high-importance terms
  - Compare against resume vocabulary (token set diff)
  - Returns ranked list: `[{"keyword": str, "importance": float, "type": str}]`
  - *Tests:* `tests/keyword_gap/test_extractor.py`

- [ ] **T-16** · `src/keyword_gap/classifier.py`
  - `classify_keyword(keyword: str) -> Literal["hard_skill", "soft_skill", "domain_term", "other"]`
  - Uses seed lists from `rubrics/keyword_categories.json`
  - Fallback: POS-tag heuristic via spaCy (noun phrase → hard skill signal)
  - *Tests:* `tests/keyword_gap/test_classifier.py`

---

## SPRINT 8 — Feedback Engine

> **Goal:** Build the deterministic rule-based feedback mapper. No generative text.

- [ ] **T-17** · `rubrics/feedback_rules.json` *(expand to full 175 rules)*
  - Covers: 7 domains × 5 score bands × 5 dimensions = 175 rules minimum
  - Each rule: `{ "domain", "score_band", "dimension", "feedback", "fresher_variant" }`
  - `fresher_variant` is an alternate feedback string when experience signal is low
  - Agent writes the complete JSON (all 175 rules)
  - *No test file — JSON data file*

- [ ] **T-18** · `src/feedback/feedback_mapper.py`
  - `generate_feedback(domain_index: int, score: float, dimension_scores: dict, is_fresher: bool) -> list[str]`
  - Loads rules from `rubrics/feedback_rules.json`
  - Returns ordered list of 3–5 feedback strings
  - Uses `fresher_variant` when `is_fresher=True`
  - Score → band mapping from `src/config.py`
  - *Tests:* `tests/feedback/test_feedback_mapper.py`

---

## SPRINT 9 — TFLite Conversion

> **Goal:** Convert trained Keras model to TFLite. Validate parity. Check size.

- [ ] **T-19** · `src/conversion/convert_to_tflite.py`
  - `convert_and_validate(keras_model_path: Path, output_path: Path) -> dict`
  - Converts using `tf.lite.TFLiteConverter.from_saved_model()`
  - Applies Float16 dynamic quantization
  - Runs parity check: 10 sample inputs through Keras + TFLite, assert diff < 0.02
  - Checks output file size < 30MB
  - Returns `{"size_mb": float, "max_diff": float, "passed": bool}`
  - *Tests:* `tests/conversion/test_convert_to_tflite.py`

- [ ] **T-20** · `notebooks/05_tflite_conversion.ipynb`
  - Calls `convert_and_validate()` on the saved model
  - Prints size report and parity check results
  - Saves `model/tflite/ats_core.tflite`
  - *No test file — notebook only*

---

## SPRINT 10 — I/O Schema & Handoff Documentation

> **Goal:** Write the Flutter team's integration contract. No code — pure documentation.

- [ ] **T-21** · `model/tflite/IO_SCHEMA.md`
  - Complete tensor-level I/O contract for Flutter/tflite_flutter
  - Input tensor names, indices, shapes, dtypes, expected value ranges
  - Output tensor names, indices, shapes, dtypes, how to interpret values
  - How to call keyword gap + feedback as separate post-processing steps
  - Example pseudocode in Dart (Flutter team reference)
  - *Documentation file only*

- [ ] **T-22** · `HANDOFF.md`
  - How to load `ats_core.tflite` in Flutter
  - Full end-to-end call sequence
  - JSON serialization format for all outputs
  - Known limitations and edge cases
  - Contact: Sai (AI intern) for model questions; Flutter team owns integration
  - *Documentation file only*

---

## BONUS TASKS (Execute after Sprint 10 if time allows)

- [ ] **T-23** · `src/preprocessing/augmenter.py`
  - Synthetic pair augmentation for underrepresented domains (< 50 pairs)
  - Paraphrasing via word substitution from synonym table
  - *Tests:* `tests/preprocessing/test_augmenter.py`

- [ ] **T-24** · `src/ats_engine/dimension_scorer.py`
  - Breaks the composite ATS score into per-dimension subscores
  - Used by the feedback mapper to identify the weakest dimension
  - Returns `dict[str, float]` of dimension name → subscore
  - *Tests:* `tests/ats_engine/test_dimension_scorer.py`

- [ ] **T-25** · `evaluation/domain_eval.py`
  - Per-domain MAE and F1 breakdown
  - Flags any domain with MAE > 10.0 for targeted data augmentation
  - *Tests:* `tests/test_domain_eval.py`

- [ ] **T-26** · `src/conversion/int8_converter.py`
  - Fallback Int8 quantization if Float16 exceeds 30MB
  - Requires representative dataset for calibration
  - *Tests:* `tests/conversion/test_int8_converter.py`

- [ ] **T-27** · `src/keyword_gap/jd_analyzer.py`
  - Pre-analyzes a JD to extract required skills, experience signals, and domain-specific terms
  - Used upstream of the keyword gap module
  - *Tests:* `tests/keyword_gap/test_jd_analyzer.py`

- [ ] **T-28** · `evaluation/tflite_benchmark.py`
  - Measures TFLite inference time on CPU (simulates on-device performance)
  - Runs 100 inference passes and reports mean/p95 latency
  - *Tests:* `tests/test_tflite_benchmark.py`

---

## TASK SUMMARY TABLE

| ID | File | Sprint | Status |
|---|---|---|---|
| T-01 | src/preprocessing/data_loader.py | 1 | [ ] |
| T-02 | notebooks/01_data_exploration.ipynb | 1 | [ ] |
| T-03 | src/preprocessing/text_cleaner.py | 2 | [ ] |
| T-04 | src/preprocessing/section_segmenter.py | 2 | [ ] |
| T-05 | src/preprocessing/normalizer.py | 2 | [ ] |
| T-06 | src/preprocessing/domain_mapper.py | 2 | [ ] |
| T-07 | notebooks/02_label_generation.ipynb | 3 | [ ] |
| T-08 | src/preprocessing/pair_builder.py | 3 | [ ] |
| T-09 | src/encoding/use_lite_encoder.py | 4 | [ ] |
| T-10 | src/ats_engine/model.py | 5 | [ ] |
| T-11 | src/ats_engine/rubric_layer.py | 5 | [ ] |
| T-12 | src/ats_engine/trainer.py | 6 | [ ] |
| T-13 | notebooks/03_ats_model_training.ipynb | 6 | [ ] |
| T-14 | evaluation/ats_eval.py | 6 | [ ] |
| T-15 | src/keyword_gap/extractor.py | 7 | [ ] |
| T-16 | src/keyword_gap/classifier.py | 7 | [ ] |
| T-17 | rubrics/feedback_rules.json (full 175) | 8 | [ ] |
| T-18 | src/feedback/feedback_mapper.py | 8 | [ ] |
| T-19 | src/conversion/convert_to_tflite.py | 9 | [ ] |
| T-20 | notebooks/05_tflite_conversion.ipynb | 9 | [ ] |
| T-21 | model/tflite/IO_SCHEMA.md | 10 | [ ] |
| T-22 | HANDOFF.md | 10 | [ ] |
| T-23 | src/preprocessing/augmenter.py | Bonus | [ ] |
| T-24 | src/ats_engine/dimension_scorer.py | Bonus | [ ] |
| T-25 | evaluation/domain_eval.py | Bonus | [ ] |
| T-26 | src/conversion/int8_converter.py | Bonus | [ ] |
| T-27 | src/keyword_gap/jd_analyzer.py | Bonus | [ ] |
| T-28 | evaluation/tflite_benchmark.py | Bonus | [ ] |

---

## EXECUTION CHECKLIST (Run in Order)

```
Sprint 1  →  T-01  →  T-02
Sprint 2  →  T-03  →  T-04  →  T-05  →  T-06
Sprint 3  →  T-07  →  T-08
Sprint 4  →  T-09
Sprint 5  →  T-10  →  T-11
Sprint 6  →  T-12  →  T-13  →  T-14
Sprint 7  →  T-15  →  T-16
Sprint 8  →  T-17  →  T-18
Sprint 9  →  T-19  →  T-20
Sprint 10 →  T-21  →  T-22
Bonus     →  T-23  →  T-24  →  T-25  →  T-26  →  T-27  →  T-28
```

---

## TARGET METRICS (Pass/Fail Gate Before Sprint 10)

| Metric | Target | Check After |
|---|---|---|
| Score MAE | < 8.0 (0–100 scale) | T-14 |
| Domain F1 | > 0.85 | T-14 |
| TFLite file size | < 30 MB | T-19 |
| TFLite output parity | diff < 0.02 | T-19 |

If any metric fails, re-run the corresponding sprint before proceeding to Sprint 10.
