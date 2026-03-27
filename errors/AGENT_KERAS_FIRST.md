# AGENT PROMPT INJECTION — Keras Model First, Accuracy Above All
**Priority: CRITICAL | Sprint: 5–6 Focus**
**Author: Sai | Direction confirmed**

---

## Directive

**Do NOT touch TFLite conversion at this stage.**
`src/conversion/convert_to_tflite.py` is frozen. Do not run it. Do not optimize it.
TFLite conversion is a future task — it happens only after the Keras model is
producing accurate, validated results on all three outputs.

The goal right now is one thing:

> **The Keras model must produce a correct ATS Score, correct Missing Keywords,
> and correct Feedback for any resume + JD pair fed into it.**

All three must work. All three must be accurate. Nothing else matters until then.

---

## The Three Outputs — Acceptance Criteria

### Output 1 — ATS Score
- Range: 0–100 (float, displayed as integer)
- Must reflect actual resume-to-JD alignment, not a random or constant value
- Evaluation target: **Score MAE < 8.0** on the held-out test set
- A fresher resume with strong skill match must score comparably to an
  experienced resume with the same skill match — experience alone must not
  inflate or deflate the score
- Verify: run `evaluation/ats_eval.py` and confirm MAE is under 8.0

### Output 2 — Missing Keywords
- Must return keywords that are genuinely present in the JD but absent from the resume
- Must separate **hard skills** (tools, technologies, certifications) from
  **soft skills** (communication, leadership, collaboration)
- Must be ranked by TF-IDF importance to the JD — highest importance first
- Must NOT include keywords already present in the resume
- Verify: test on 3 sample pairs manually and confirm the returned keywords
  are real gaps, not noise

### Output 3 — Feedback
- Must return domain-specific, actionable feedback — not generic strings
- Must use the correct domain (from model's domain_label output) to select rules
- Must use the correct score band to select feedback tier
- Must include fresher_variant text for candidates with no work experience
- Must load from `rubrics/feedback_rules.json` (175 rules, 7 domains × 5 bands)
- Verify: call `get_feedback()` with at least 5 different score/domain combinations
  and confirm the output is specific and non-repetitive

---

## Current Known Fixes Already Applied (Do Not Redo These)

The following fixes were completed in the previous session. Do not revisit unless
a new bug surfaces in that area:

| Fix | File | Status |
|-----|------|--------|
| Encoder changed to USE v4 (raw string input) | src/config.py | Done |
| Rubric JSON files copied to correct directory | ats-ai-core/rubrics/ | Done |
| feedback_rules.json expanded to 175 rules | rubrics/feedback_rules.json | Done |
| 3-way unpack bug fixed | evaluation/ats_eval.py | Done |
| 3-way unpack bug fixed | src/conversion/convert_to_tflite.py | Done (frozen) |

---

## Task 1 — Run Full Training and Capture Metrics

**File:** `src/ats_engine/trainer.py`

Run a full training cycle. After training completes, immediately run evaluation.

```bash
python src/ats_engine/trainer.py
python evaluation/ats_eval.py
```

Capture and report ALL of the following metrics — do not summarize, paste them raw:

- Score MAE (target: < 8.0)
- Domain Classification F1 (target: > 0.85)
- Score Band Accuracy (% of samples assigned to the correct band)
- Per-domain F1 breakdown (one score per domain, 7 values)
- Training loss curve (final 5 epoch values)
- Validation loss curve (final 5 epoch values)

**If MAE > 8.0 or Domain F1 < 0.85 → do not proceed to Task 2. Report metrics
and wait for Sai's instructions.**

---

## Task 2 — Validate All Three Outputs End-to-End

Only run this after Task 1 metrics pass both targets.

Run the following smoke test with all 5 sample pairs. Paste the full output — do not truncate:

```python
from src.ats_engine.inference import run_inference
from src.keyword_gap.extractor import extract_missing_keywords
from src.feedback.feedback_mapper import get_feedback

pairs = [
    {
        "label": "IT Fresher — Strong Match",
        "resume": "Final year CS student. Skills: Python, Django, REST APIs, PostgreSQL, Git. "
                  "Projects: E-commerce backend (Django + PostgreSQL), REST API for a food delivery app.",
        "jd": "Backend Engineer — Python, Django, REST APIs, PostgreSQL, Docker, CI/CD pipelines."
    },
    {
        "label": "IT Experienced — Weak Match",
        "resume": "5 years Java developer. Spring Boot, Oracle DB, Maven, Jenkins.",
        "jd": "Frontend React developer. JavaScript, TypeScript, Next.js, Tailwind, GraphQL."
    },
    {
        "label": "Finance Fresher — Moderate Match",
        "resume": "BBA Finance graduate. Excel, financial modelling basics, internship at NBFC. "
                  "Coursework: equity valuation, risk management.",
        "jd": "Equity Research Analyst. DCF modelling, Bloomberg Terminal, Python for finance, CFA preferred."
    },
    {
        "label": "Healthcare — Good Match",
        "resume": "MBBS graduate. Clinical rotations in general medicine, pediatrics. "
                  "BLS certified. EHR experience with MedTech software.",
        "jd": "Junior Resident Doctor. MBBS required, BLS/ACLS certified, EHR systems, patient care."
    },
    {
        "label": "Design — Poor Match",
        "resume": "Mechanical engineer. AutoCAD, SolidWorks, FEA analysis, GD&T.",
        "jd": "UX Designer. Figma, user research, wireframing, prototyping, usability testing."
    },
]

for p in pairs:
    result   = run_inference(p["resume"], p["jd"])
    keywords = extract_missing_keywords(p["resume"], p["jd"])
    feedback = get_feedback(result['ats_score'], result['domain_label'])

    print(f"\n{'='*60}")
    print(f"Test Case : {p['label']}")
    print(f"ATS Score : {result['ats_score']:.1f} / 100")
    print(f"Domain    : {result['domain_label']} (0=IT,1=NonIT,2=Design,3=Health,4=Finance,5=Legal,6=Edu)")
    print(f"Hard KWs  : {keywords.get('hard_skills', [])[:5]}")
    print(f"Soft KWs  : {keywords.get('soft_skills', [])[:3]}")
    print(f"Feedback  : {feedback[:2]}")
```

**Sanity checks to apply to the output:**

| Test Case | Expected Score Range | Expected Domain |
|-----------|---------------------|-----------------|
| IT Fresher — Strong Match | 65–85 | 0 (IT) |
| IT Experienced — Weak Match | 10–30 | 0 (IT) |
| Finance Fresher — Moderate Match | 40–65 | 4 (Finance) |
| Healthcare — Good Match | 65–85 | 3 (Healthcare) |
| Design — Poor Match | 5–25 | 2 (Design) |

If any score falls outside its expected range by more than 15 points, or if any
domain label is wrong, report it as a misclassification — do not silently accept it.

---

## Task 3 — Fresher Fairness Check

This is a core design rule: freshers must not be penalized.

Run this specific comparison and report both scores:

```python
experienced_resume = (
    "8 years software engineer. Python, FastAPI, Docker, Kubernetes, "
    "PostgreSQL, Redis, CI/CD, led team of 5 engineers."
)
fresher_resume = (
    "Final year CS student. Python, FastAPI, Docker basics learned via "
    "online course. Built a containerized REST API project on GitHub."
)
jd = (
    "Backend Engineer. Python, FastAPI, Docker, PostgreSQL. "
    "Team player, problem solver, able to work independently."
)

r_exp     = run_inference(experienced_resume, jd)
r_fresher = run_inference(fresher_resume, jd)

print(f"Experienced Score : {r_exp['ats_score']:.1f}")
print(f"Fresher Score     : {r_fresher['ats_score']:.1f}")
print(f"Gap               : {r_exp['ats_score'] - r_fresher['ats_score']:.1f} points")
```

**Acceptance rule: the gap must be ≤ 20 points.**

A fresher with matching skills should not score more than 20 points below an
experienced candidate on the same JD. If the gap is > 20, the model is
penalizing experience level — report this as a fairness failure and do not
proceed until it is investigated.

---

## Task 4 — Report Summary

After all tasks complete, provide a report in this exact format:

```
=== KERAS MODEL VALIDATION REPORT ===

METRICS
-------
Score MAE         : [value]  → [PASS / FAIL — target <8.0]
Domain F1         : [value]  → [PASS / FAIL — target >0.85]
Band Accuracy     : [value]%

PER-DOMAIN F1
-------------
IT / Software     : [value]
Non-IT / Mgmt     : [value]
Design / Creative : [value]
Healthcare        : [value]
Finance / Banking : [value]
Legal             : [value]
Education         : [value]

SMOKE TEST
----------
All 5 pairs within expected score range : [YES / NO — list failures if NO]
All 5 domain labels correct             : [YES / NO — list failures if NO]

FRESHER FAIRNESS
----------------
Score gap (experienced vs fresher)      : [value] points → [PASS / FAIL — target ≤20]

OVERALL STATUS
--------------
[ ] READY — all checks pass. Sai to review before proceeding to TFLite.
[ ] NOT READY — failures listed above. Waiting for Sai's instructions.
```

---

## What NOT to Do

- Do NOT run `src/conversion/convert_to_tflite.py` — TFLite is deferred
- Do NOT modify `rubrics/feedback_rules.json` — it is complete at 175 rules
- Do NOT change the encoder back to USE Lite v2
- Do NOT touch `src/summary_generator/` — out of scope
- Do NOT mark any task complete without pasting the actual output

---

## Definition of Done for This Injection

- [ ] Training completed and raw metrics reported
- [ ] Score MAE < 8.0 confirmed
- [ ] Domain F1 > 0.85 confirmed
- [ ] All 5 smoke test pairs produce output within expected ranges
- [ ] Fresher fairness gap ≤ 20 points confirmed
- [ ] Full validation report pasted in the format above
- [ ] TFLite conversion NOT touched

---

*Injection prepared by: Claude Assistant on behalf of Sai*
*Next step after this injection passes: Sai reviews report → TFLite decision.*
