# AGENT PROMPT INJECTION — ATS Pipeline Fix
**Priority: CRITICAL | Sprint: 5–8 Remediation**
**Author: Sai | Delivered via: Claude Assistant**

---

## Context for the Agent

A GUI test was run where a resume and job description were provided as inputs.
**The model produced NO output** — no ATS score, no missing keywords, no feedback.

This document identifies every broken link in the pipeline and gives you exact,
prioritized instructions to fix them. Work through Priority 1 → 2 → 3 in order.
Do not skip ahead.

---

## What the Model Must Produce (Non-Negotiable)

The three outputs below are the entire purpose of the ATS Scoring Engine.
All three must work end-to-end before the project is considered functional.

| Priority | Output | How It's Produced |
|----------|--------|-------------------|
| 1 — MUST WORK FIRST | ATS Score (0–100) | TFLite model tensor → multiply float by 100 |
| 2 — DEPENDS ON #1 | Missing Keywords | Post-processing: TF-IDF gap between JD and resume |
| 3 — DEPENDS ON #1 | Feedback List | Post-processing: rule lookup via score band + domain index |

> Outputs 2 and 3 are **NOT** model tensors. They run in Python after the model
> returns score + domain label. If the model isn't returning those two values,
> outputs 2 and 3 will never work.

---

## Priority 1 — Fix the Inference Path (Score Must Come Out)

### 1.1 Check `inference.py` exists and is wired

**File:** `src/ats_engine/inference.py`

This file must exist and must do exactly this:

```python
def run_inference(resume_text: str, jd_text: str) -> dict:
    """
    Load the TFLite model and run inference on a resume-JD pair.

    Args:
        resume_text: Full resume as plain text string.
        jd_text: Full job description as plain text string.

    Returns:
        dict with keys: 'ats_score' (float 0–100), 'domain_label' (int 0–6)
    """
```

- Load model from `model/tflite/ats_core.tflite`
- Input tensor 0 = resume_text (shape [1], string)
- Input tensor 1 = jd_text (shape [1], string)
- Output tensor 0 = ats_score (float32, multiply by 100)
- Output tensor 1 = domain_label (int32, index 0–6)
- Return a plain Python dict — no tensors in the return value

**If this file does not exist → create it now using the pattern above.**
**If it exists but is incomplete → patch only the broken sections.**

---

### 1.2 Verify the TFLite model file is present and valid

Run this check:

```bash
ls -lh model/tflite/ats_core.tflite
python -c "
import tensorflow as tf
interp = tf.lite.Interpreter('model/tflite/ats_core.tflite')
interp.allocate_tensors()
print('Input details:', interp.get_input_details())
print('Output details:', interp.get_output_details())
"
```

**Expected:** 2 inputs (resume_text, jd_text), 2 outputs (ats_score float32, domain_label int32).

**If the file is missing or the tensor count is wrong:**
→ The Keras model must be re-exported. Run `src/conversion/convert_to_tflite.py`.
→ After conversion, re-run the check above before proceeding.

---

### 1.3 Confirm the GUI is calling `inference.py`

The GUI test Sai ran attached a resume and pasted a JD — but produced no output.
The most likely cause is that the GUI button/action is **not actually calling** `run_inference()`.

Check the GUI entry point (the script or widget that handles the "Submit / Score" action):
- It must read resume text and JD text from the input fields.
- It must call `run_inference(resume_text, jd_text)`.
- It must display the returned `ats_score` value.

If the call is missing → wire it now. This is the minimum fix to unblock output #1.

---

## Priority 2 — Fix the Missing Keywords Module

**File:** `src/keyword_gap/extractor.py`

This runs AFTER `run_inference()` returns. It does NOT touch the TFLite model.

```python
def extract_missing_keywords(resume_text: str, jd_text: str) -> dict:
    """
    Identify keywords present in the JD but absent from the resume.

    Args:
        resume_text: Full resume as plain text.
        jd_text: Full job description as plain text.

    Returns:
        dict with keys:
            'hard_skills': list of str — technical/tool keywords missing
            'soft_skills': list of str — soft skill keywords missing
        Keywords are ranked by TF-IDF importance to the JD (highest first).
    """
```

### Checklist for this module:

- [ ] TF-IDF is fitted on `jd_text` only — never on the resume
- [ ] Keyword gap = JD high-importance terms not found in resume vocabulary
- [ ] Hard skills and soft skills are reported **separately**
- [ ] TF-IDF fitting happens at call time (not at model load time)
- [ ] No sklearn objects are serialized inside the TFLite model

**If this file does not exist → create it.**
**If it exists → run a quick smoke test with a sample resume and JD pair to confirm output.**

---

## Priority 3 — Fix the Feedback Mapper

**File:** `src/feedback/feedback_mapper.py`

This also runs AFTER `run_inference()` returns. It needs the score AND the domain label.

```python
def get_feedback(ats_score: float, domain_label: int) -> list[str]:
    """
    Map score band + domain to a list of actionable feedback items.

    Args:
        ats_score: Float in [0.0, 100.0] (already multiplied).
        domain_label: Integer 0–6 (from model output tensor 1).

    Returns:
        List of feedback strings. Each string is specific, actionable,
        and fresher-friendly (no penalty for missing experience).
    """
```

Score band mapping (use this exactly):

| Score Range | Band |
|-------------|------|
| 85–100 | Excellent Match |
| 65–84 | Good Match |
| 45–64 | Moderate Match |
| 25–44 | Weak Match |
| 0–24 | Poor Match |

Domain index mapping (0=IT, 1=NonIT, 2=Design, 3=Healthcare, 4=Finance, 5=Legal, 6=Education).

Rules are loaded from `rubrics/feedback_rules.json`.

**If `feedback_rules.json` has fewer than 30 rules → flag it. Target is 175 rules (Sprint 8).**
**For now, a minimum of 5 rules per domain × 5 bands × 7 domains = 35 rules is needed to unblock testing.**

---

## End-to-End Wiring Check

After fixing all three priorities, verify this call chain works top to bottom:

```python
# Smoke test — run this manually to confirm all three outputs produce values
from src.ats_engine.inference import run_inference
from src.keyword_gap.extractor import extract_missing_keywords
from src.feedback.feedback_mapper import get_feedback

resume = "Python developer with Django, REST APIs, PostgreSQL, 2 projects on GitHub"
jd     = "Looking for a Python backend engineer with FastAPI, Docker, Kubernetes, CI/CD"

result    = run_inference(resume, jd)
keywords  = extract_missing_keywords(resume, jd)
feedback  = get_feedback(result['ats_score'], result['domain_label'])

print("Score:    ", result['ats_score'])
print("Domain:   ", result['domain_label'])
print("Keywords: ", keywords)
print("Feedback: ", feedback)
```

**All four print statements must produce non-empty values.** If any line throws or returns empty,
that module still has a bug — fix it before marking this injection complete.

---

## What NOT to Touch

- `src/summary_generator/` — out of scope, do not open these files
- Any `.dart` Flutter files
- `rubrics/summary_templates.json`
- Gold labels in `data/labeled/gold_labels.csv`
- IO_SCHEMA.md tensor order — any change here requires explicit sign-off from Sai first

---

## Definition of Done for This Injection

- [ ] `run_inference()` returns a dict with `ats_score` and `domain_label`
- [ ] GUI wired to call `run_inference()` and display the score
- [ ] `extract_missing_keywords()` returns separate hard/soft skill lists
- [ ] `get_feedback()` returns at least 1 non-empty feedback string per call
- [ ] Smoke test above runs without errors and all outputs are non-empty
- [ ] No new files created outside the paths listed in scope above

---

*Injection prepared by: Claude Assistant on behalf of Sai*
*Reference documents: ATS_AI_Core_Documentation.pdf, ARCHITECTURE.pdf*
