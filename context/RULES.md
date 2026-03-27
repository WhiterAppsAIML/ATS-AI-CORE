# RULES.md — Coding Rules & Agent Guidelines

> These rules apply to ALL code, notebooks, and configuration files in this project.
> The AI agent (Claude Opus via AntiGravity IDE) must follow every rule below without exception.

---

## 1. Scope Rules

### 1.1 — This Intern's Scope
The AI intern is responsible ONLY for:
- `src/preprocessing/`
- `src/encoding/`
- `src/ats_engine/`
- `src/keyword_gap/`
- `src/feedback/`
- `src/conversion/`
- `rubrics/*.json`
- `evaluation/ats_eval.py`
- `notebooks/` (01 through 05)
- `model/tflite/IO_SCHEMA.md`

**Do NOT touch:**
- Flutter code (`.dart` files)
- Firebase config files
- `src/summary_generator/` (belongs to another team member)
- Any file not listed above

### 1.2 — No Scope Creep
If a task seems to require modifying something outside the above scope, STOP and flag it to the team lead. Do not guess or improvise.

---

## 2. Technology Rules

### 2.1 — Approved Libraries Only
| Purpose | Approved Library | Forbidden Alternatives |
|---------|-----------------|----------------------|
| Deep learning | TensorFlow 2.x / Keras | PyTorch, JAX, ONNX |
| Text encoding | TF Hub USE Lite / MobileBERT (TFLite-compatible only) | OpenAI embeddings, HuggingFace models with non-TFLite ops |
| Keyword extraction | scikit-learn TF-IDF | Rake-NLTK (training phase only — not at inference) |
| NLP utilities | spaCy (training phase only) | NLTK, Stanza |
| Data handling | pandas, numpy | Dask, PySpark |
| Model conversion | `tf.lite.TFLiteConverter` | ONNX converter, coremltools |

### 2.2 — No Large Generative Models
Do NOT use or suggest:
- GPT-3/4/4o or any OpenAI model
- LLaMA, Mistral, Gemma, or any LLM
- Any model that cannot be converted to TFLite
- Any model requiring a server call at inference time

### 2.3 — TFLite Compatibility Check
Before using any TensorFlow operation, verify it is supported by TFLite. Unsupported ops include:
- `tf.py_function`
- Dynamic shapes where avoidable
- Certain string ops at inference time (pre-tokenize before passing to model)

---

## 3. Code Style Rules

### 3.1 — Python Standards
- Python 3.10+
- Type hints required on all function signatures
- Docstrings required on all public functions and classes (Google style)
- Max line length: 100 characters
- No wildcard imports (`from x import *`)

### 3.2 — File and Function Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- No abbreviations unless widely standard (e.g. `tfidf`, `ats`, `use`)

### 3.3 — No Hardcoded Paths
All file paths must use `pathlib.Path` and be configurable via a central `config.py`. Example:
```python
# CORRECT
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / "data"

# WRONG
path = "/home/sai/ats-ai-core/data/raw/resumes.csv"
```

### 3.4 — No Hardcoded Hyperparameters in Model Files
All training hyperparameters (learning rate, batch size, epochs, loss weights) must be in a `config.py` or passed as arguments. Never embed them directly in model or trainer code.

---

## 4. Model Design Rules

### 4.1 — I/O Contract is Sacred
The ATS engine must always output exactly:
- `Output 0`: ATS score — float32, shape `[1]`, range `[0.0, 1.0]`
- `Output 1`: Domain label — int32, shape `[1]`, range `[0, 6]`

Do NOT change output tensor order or shape without updating `IO_SCHEMA.md` and notifying the Flutter team.

### 4.2 — Score Is Relative, Not Absolute
The score must reflect resume-JD alignment, NOT years of experience. The model must never use experience years as a direct scoring signal. Experience level affects weight distribution via the rubric, not the raw similarity score.

### 4.3 — Feedback and Keyword Gap Are NOT Model Outputs
Feedback strings and keyword lists must NEVER be output tensors of the TFLite model. They are computed in post-processing Python/Dart code using the domain label and score as inputs.

### 4.4 — Model Size Constraint
The final `ats_core.tflite` file must be under 30MB. Validate size after every conversion attempt. If over 30MB, apply:
1. Float16 quantization (default)
2. int8 quantization (if Float16 is insufficient)
3. Reduce model depth (before reaching this step, consult team lead)

### 4.5 — Frozen Encoder First
Always start training with the USE Lite encoder frozen. Only unfreeze encoder layers if MAE is above 10.0 after full training. Fine-tuning the encoder adds size and risk — justify it explicitly.

---

## 5. Data Rules

### 5.1 — Never Commit Raw Data
Do not commit Kaggle datasets or any raw data files to the repository. Add `data/raw/` and `data/processed/` to `.gitignore`.

### 5.2 — Label Integrity
- Weak labels go to `data/labeled/weak_labels.csv`
- Gold labels go to `data/labeled/gold_labels.csv`
- Never overwrite gold labels with weak labels
- Always version label files with a date suffix if regenerating

### 5.3 — No Data Leakage
Perform train/validation/test split BEFORE any feature extraction or TF-IDF fitting. The TF-IDF vocabulary must be fit only on training data.

### 5.4 — Domain Balance
Ensure the training set has representation from all 7 domains. If any domain has fewer than 50 pairs, flag it and use synthetic augmentation.

---

## 6. Rubric Rules

### 6.1 — Weights Must Sum to 100%
Every domain row in `rubrics/domain_weights.json` must sum exactly to 1.0 (100%). Validate this programmatically before training.

### 6.2 — Fresher Treatment
No scoring dimension may directly penalize a candidate for having zero years of experience. The `Structural Completeness` dimension treats Projects and Certifications as equivalent to Work Experience sections.

### 6.3 — Feedback Must Be Actionable
Every feedback string in `rubrics/feedback_rules.json` must:
- Start with a verb (Add, Include, Highlight, Quantify, etc.)
- Reference the specific gap (not just "improve your resume")
- Offer an alternative for freshers where experience is typically required

---

## 7. Notebook Rules

### 7.1 — Notebooks Are for Exploration Only
Production code lives in `src/`. Notebooks import from `src/` — they do not contain reusable logic themselves.

### 7.2 — Clear Outputs Before Committing
Never commit notebooks with embedded model outputs, large dataframes, or binary outputs. Clear all cell outputs before committing.

### 7.3 — Numbered Prefix
Notebooks must follow the naming convention: `NN_short_description.ipynb` where NN is a two-digit number (01, 02, etc.). No renaming existing notebooks without team lead approval.

---

## 8. Agent-Specific Rules (for Claude Opus / AntiGravity IDE)

### 8.1 — Read Before Writing
Before generating any new file or function, read the relevant existing files in `src/` and `ARCHITECTURE.md`. Do not generate code in isolation.

### 8.2 — No Silent Assumptions
If a task requires a decision not covered by this document or `ARCHITECTURE.md`, state the assumption explicitly in a comment and in your response before proceeding.

### 8.3 — Propose, Don't Auto-Apply to Rubrics
Changes to `rubrics/*.json` must be proposed as a JSON diff, not applied directly.

### 8.4 — One File Per Response
When writing new source files, generate one complete file per response. Do not split a module across multiple responses.

### 8.5 — Test Coverage Expectation
Every module in `src/` must have at least 3 unit test cases. Write tests in the same response as the module, placed in a `tests/` directory mirroring the `src/` structure.

### 8.6 — Always Validate TFLite Output
After any conversion script, include a validation block that runs 5 sample inputs through both the Keras model and TFLite model and asserts output parity within tolerance 0.02.
