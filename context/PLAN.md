# PLAN.md — ATS Scoring Engine: Sprint-Wise Execution Plan

> **Scope:** AI Engineering Intern — ATS Scoring Engine only
> **Stack:** TensorFlow 2.x → TFLite → Firebase Storage
> **Out of Scope:** Flutter UI, Firebase integration code, Resume Summary Generator

---

## Sprint Overview

| Sprint | Focus | Deliverable |
|--------|-------|-------------|
| Sprint 1 | Data Collection & Exploration | Cleaned datasets + EDA notebook |
| Sprint 2 | Preprocessing Pipeline | `src/preprocessing/` modules |
| Sprint 3 | Label Generation | Weak-labeled resume-JD pairs |
| Sprint 4 | Encoder Setup | USE Lite wrapper + embedding validation |
| Sprint 5 | ATS Model Architecture | Keras model with all heads |
| Sprint 6 | Training & Evaluation | Trained model + eval report |
| Sprint 7 | Keyword Gap Module | TF-IDF keyword extractor |
| Sprint 8 | Feedback Engine | Rule-based feedback mapper |
| Sprint 9 | TFLite Conversion | `ats_core.tflite` under 30MB |
| Sprint 10 | Handoff & Documentation | I/O schema + handoff doc for Flutter team |

---

## Sprint 1 — Data Collection & Exploration

**Goal:** Collect all datasets, understand structure, identify cleaning needs.

**Tasks:**
- [ ] Download Resume Dataset (2484 resumes) from Kaggle
- [ ] Download LinkedIn Job Postings 2023 from Kaggle
- [ ] Download Monster.com Job Listings from Kaggle
- [ ] Download UpdatedResumeDataSet from Kaggle
- [ ] Place all raw files in `data/raw/`
- [ ] Run `notebooks/01_data_exploration.ipynb`
- [ ] Document column names, data types, null rates, domain distribution

**Exit Criteria:** EDA notebook complete; domain distribution chart confirms coverage of IT, Non-IT, Design, Healthcare, Finance, Legal, Education.

---

## Sprint 2 — Preprocessing Pipeline

**Goal:** Build reusable text cleaning and section segmentation modules.

**Tasks:**
- [ ] Build `src/preprocessing/text_cleaner.py`
  - Strip HTML tags, special characters
  - Normalize whitespace and casing
  - Remove boilerplate (page numbers, headers/footers)
- [ ] Build `src/preprocessing/section_segmenter.py`
  - Segment resumes into: Skills, Education, Experience, Projects, Certifications, Summary
  - Regex + heuristic approach (no ML needed here)
- [ ] Build `src/preprocessing/normalizer.py`
  - Standardize skill synonyms (e.g. "JS" → "JavaScript")
  - Lower-case domain-specific term normalization
- [ ] Write unit tests for each module
- [ ] Run on full dataset; save to `data/processed/`

**Exit Criteria:** All resume and JD text is clean, structured, and saved as JSON/CSV in `data/processed/`.

---

## Sprint 3 — Label Generation

**Goal:** Generate ATS score labels for all resume-JD pairs using the two-phase strategy.

**Tasks:**
- [ ] Build `notebooks/02_label_generation.ipynb`
- [ ] Phase 1 — Weak Labeling:
  - Compute TF-IDF cosine similarity between resume and JD text
  - Compute keyword overlap ratio (intersection / JD keywords)
  - Combine into a weighted heuristic score (0–100)
  - Apply domain-weight table from `rubrics/domain_weights.json`
- [ ] Pair each resume with 3–5 JDs (random + same-domain) to create diverse pairs
- [ ] Save weak-labeled pairs to `data/labeled/weak_labels.csv`
- [ ] Phase 2 — Human Annotation scaffolding:
  - Export 300-sample annotation sheet
  - Document annotation guidelines in `data/labeled/ANNOTATION_GUIDE.md`
- [ ] Save final gold labels to `data/labeled/gold_labels.csv`

**Exit Criteria:** At minimum, weak labels generated for all pairs. Gold labels added as they become available.

---

## Sprint 4 — Encoder Setup

**Goal:** Load and validate USE Lite as the shared text encoder.

**Tasks:**
- [ ] Build `src/encoding/use_lite_encoder.py`
  - Load USE Lite from TFHub (`https://tfhub.dev/google/universal-sentence-encoder-lite/2`)
  - Expose `encode(text: str) -> np.ndarray` interface
  - Batch encoding support for training efficiency
- [ ] Validate embedding quality:
  - Test that similar texts produce high cosine similarity
  - Test that dissimilar texts produce low similarity
  - Log sample pairs in `notebooks/` for review
- [ ] Confirm the encoder is TFLite-convertible (no unsupported ops)

**Exit Criteria:** Encoder returns stable 512-dim vectors; cosine similarity sanity checks pass.

---

## Sprint 5 — ATS Model Architecture

**Goal:** Build the full Keras ATS model with all heads.

**Tasks:**
- [ ] Build `src/ats_engine/model.py`
  - Input layers: resume text + JD text (string tensors)
  - Shared encoder (USE Lite, frozen initially)
  - Similarity head: cosine similarity + dense layer → scalar score
  - Domain classifier head: 7-class softmax (IT, Non-IT, Design, Healthcare, Finance, Legal, Education)
  - Score output: sigmoid → 0.0–1.0 float
- [ ] Build `src/ats_engine/rubric_layer.py`
  - Apply domain-specific weights from `rubrics/domain_weights.json` post-inference
  - This is NOT a neural layer — it's a post-processing step
- [ ] Confirm model summary and tensor shapes
- [ ] Save model config to `model/ats_model/config.json`

**Architecture Notes:**
- Do NOT build a generative head — scores and domain label only
- Feedback and keyword gap are post-processing, not model outputs
- Keep total parameter count under 5M to stay within TFLite size target

**Exit Criteria:** Model compiles, forward pass runs on sample input, output tensors match I/O contract.

---

## Sprint 6 — Training & Evaluation

**Goal:** Train the ATS model and validate performance.

**Tasks:**
- [ ] Build `src/ats_engine/trainer.py`
  - Multi-task loss: MAE (score) + CrossEntropy (domain)
  - Loss weights: 0.7 score + 0.3 domain (tunable)
  - Optimizer: Adam, lr=1e-4
  - Early stopping on validation MAE
- [ ] Run training in `notebooks/03_ats_model_training.ipynb`
- [ ] Fine-tune encoder layers if MAE plateaus above 8.0
- [ ] Build `evaluation/ats_eval.py`
  - Report: MAE, RMSE, score band accuracy, domain F1
  - Plot: predicted vs actual score scatter
- [ ] Save trained model to `model/ats_model/`

**Target Metrics:**
- Score MAE < 8.0 (on 0–100 scale)
- Domain classification F1 > 0.85

**Exit Criteria:** Model saved, eval report generated, targets met or documented with explanation.

---

## Sprint 7 — Keyword Gap Module

**Goal:** Build the TF-IDF-based missing keyword extractor.

**Tasks:**
- [ ] Build `src/keyword_gap/extractor.py`
  - Extract high-importance terms from JD using TF-IDF
  - Filter to nouns and noun phrases (spaCy or regex)
  - Compare against resume vocabulary
  - Return ranked list of missing keywords
- [ ] Build `src/keyword_gap/classifier.py`
  - Classify missing keywords as: hard skill, soft skill, or domain term
  - Use `rubrics/keyword_categories.json` lookup + fallback heuristic
- [ ] Output format: `{ "hard_skills": [...], "soft_skills": [...] }` (JSON-serializable)
- [ ] Write unit tests with sample resume-JD pairs

**Exit Criteria:** Module returns ranked, classified keyword gap list. Tested on 10+ sample pairs.

---

## Sprint 8 — Feedback Engine

**Goal:** Build the rule-based feedback mapper.

**Tasks:**
- [ ] Design `rubrics/feedback_rules.json` schema:
  ```json
  {
    "domain": "IT",
    "score_band": "Moderate Match",
    "dimension": "Skill Alignment",
    "feedback": "Add specific programming languages and frameworks mentioned in the JD."
  }
  ```
- [ ] Populate feedback rules for all 7 domains × 5 score bands × 5 dimensions = 175 rules minimum
- [ ] Build `src/feedback/feedback_mapper.py`
  - Input: domain index, score (0–100), dimension scores
  - Output: ordered list of 3–5 actionable feedback strings
  - Fresher-friendly: if experience signal is low, substitute "Add projects or academic work demonstrating..." variants
- [ ] Test on sample inputs across all domains and score bands

**Exit Criteria:** Feedback mapper returns 3–5 relevant, non-generic, domain-correct feedback items for every input combination.

---

## Sprint 9 — TFLite Conversion

**Goal:** Convert the trained Keras model to TFLite and validate.

**Tasks:**
- [ ] Build `src/conversion/convert_to_tflite.py`
  - Load saved Keras model
  - Apply Float16 dynamic quantization
  - Export to `model/tflite/ats_core.tflite`
- [ ] Validate outputs:
  - Run same 50 sample inputs through Keras and TFLite
  - Assert output difference < 0.02 (on 0.0–1.0 scale)
- [ ] Check model file size — must be under 30MB
- [ ] Document all input tensor names, shapes, dtypes, and output tensor indices in `model/tflite/IO_SCHEMA.md`

**Exit Criteria:** `.tflite` file exists under 30MB; output parity confirmed; I/O schema documented.

---

## Sprint 10 — Handoff & Documentation

**Goal:** Package everything for handoff to Flutter team.

**Tasks:**
- [ ] Write `model/tflite/IO_SCHEMA.md` (tensor-level contract for Flutter)
- [ ] Write `HANDOFF.md`:
  - How to load the model in Flutter (tflite_flutter)
  - How to call the ATS engine
  - How to interpret outputs
  - How to call the keyword gap and feedback modules (as separate JSON/API)
- [ ] Final review of all `src/` modules — docstrings, type hints, no hardcoded paths
- [ ] Tag the final commit as `v1.0-ats-core`
- [ ] Demo run: end-to-end test with 3 sample resume-JD pairs, outputs logged

**Exit Criteria:** Flutter team can pick up `ats_core.tflite` + `IO_SCHEMA.md` and integrate without clarification.

---

## Notes for the AI Agent (IDE Agent / Claude Opus)

- Always refer to `ARCHITECTURE.md` for design decisions before writing any code.
- Always refer to `RULES.md` before generating any file.
- The ATS score and domain label are model outputs. Keyword gap and feedback are POST-processing — never put them inside the neural model.
- Do not import libraries outside the approved stack without flagging it first.
- Do not modify `rubrics/*.json` files directly — always propose changes as a diff and confirm.
