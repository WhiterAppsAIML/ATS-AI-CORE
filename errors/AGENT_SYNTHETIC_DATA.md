# AGENT PROMPT INJECTION — Synthetic Data: Legal & Education Domains
**Priority: CRITICAL | Sprint: 6 — Domain F1 Fix**
**Author: Sai | Approved approach: Synthetic data generation**

---

## Context

Training results showed Domain F1 = 0.7791 (target > 0.85).
Root cause confirmed: Legal (228 samples, F1=0.68) and Education (331 samples, F1=0.62)
are severely data-starved compared to other domains (Non-IT has 1,973 samples).

All other metrics are production-grade:
- Score MAE: 2.33 (target <8.0) ✓
- Band Accuracy: 95.31% ✓
- Fresher Fairness gap: 10.1 pts (target ≤20) ✓

**The only fix needed: generate synthetic resume-JD pairs for Legal and Education,
retrain, and re-evaluate. Nothing else changes.**

---

## Task 1 — Generate Synthetic Legal Pairs

**Target:** 600 new Legal domain resume-JD pairs
**Output file:** `data/synthetic/legal_synthetic.csv`
**Columns:** `resume_text`, `jd_text`, `ats_score`, `domain_label`
**Domain label:** 5 (Legal)

### Legal JD Archetypes to Cover (build ~120 pairs per archetype):

**Archetype 1 — Corporate Lawyer / In-House Counsel**
JD keywords to include: contract drafting, corporate law, M&A, due diligence,
regulatory compliance, LLB/LLM, legal research, client advisory, negotiation

**Archetype 2 — Litigation Associate**
JD keywords: civil litigation, court filings, pleadings, legal briefs,
case management, discovery, trial preparation, oral arguments, Bar admission

**Archetype 3 — Legal Compliance Officer**
JD keywords: regulatory compliance, risk assessment, internal audit, policy drafting,
GDPR, legal frameworks, compliance training, reporting

**Archetype 4 — Paralegal / Legal Assistant**
JD keywords: legal documentation, case files, client communication, research support,
drafting correspondence, scheduling hearings, LLB or paralegal diploma

**Archetype 5 — Intellectual Property (IP) Specialist**
JD keywords: patent filing, trademark registration, IP litigation, licensing agreements,
IP law, prior art search, WIPO, copyright law

### Score Distribution for Legal Pairs (must follow this spread):
| ATS Score Range | Band | % of 600 pairs |
|----------------|------|----------------|
| 85–100 | Excellent | 15% (90 pairs) |
| 65–84 | Good | 25% (150 pairs) |
| 45–64 | Moderate | 30% (180 pairs) |
| 25–44 | Weak | 20% (120 pairs) |
| 0–24 | Poor | 10% (60 pairs) |

### Fresher representation:
At least 30% of Legal pairs must be fresher profiles (LLB students, fresh graduates,
internship experience only). These must score based on skill/keyword alignment,
not experience level.

---

## Task 2 — Generate Synthetic Education Pairs

**Target:** 500 new Education domain resume-JD pairs
**Output file:** `data/synthetic/education_synthetic.csv`
**Columns:** `resume_text`, `jd_text`, `ats_score`, `domain_label`
**Domain label:** 6 (Education)

### Education JD Archetypes to Cover (build ~100 pairs per archetype):

**Archetype 1 — School Teacher (K–12)**
JD keywords: lesson planning, classroom management, curriculum development,
student assessment, subject expertise, B.Ed, teaching certification,
parent communication, differentiated instruction

**Archetype 2 — College / University Lecturer**
JD keywords: course design, lecture delivery, research publications, PhD or Master's,
academic writing, student mentoring, syllabus planning, peer review

**Archetype 3 — EdTech / Instructional Designer**
JD keywords: e-learning, LMS platforms (Moodle, Canvas), content authoring tools
(Articulate, Captivate), instructional design, ADDIE model, learning objectives,
multimedia content

**Archetype 4 — Academic Coordinator / Administrator**
JD keywords: academic planning, timetable management, faculty coordination,
accreditation compliance, student records, program administration, reporting

**Archetype 5 — Special Education / Counseling**
JD keywords: special needs education, IEP planning, student counseling, behavioral
support, inclusive education, RCI certification, therapeutic communication

### Score Distribution for Education Pairs (must follow this spread):
| ATS Score Range | Band | % of 500 pairs |
|----------------|------|----------------|
| 85–100 | Excellent | 15% (75 pairs) |
| 65–84 | Good | 25% (125 pairs) |
| 45–64 | Moderate | 30% (150 pairs) |
| 25–44 | Weak | 20% (100 pairs) |
| 0–24 | Poor | 10% (50 pairs) |

### Fresher representation:
At least 35% of Education pairs must be fresher profiles (B.Ed students, fresh graduates,
teaching internship only). Fresher scores must reflect skill alignment, not experience.

---

## Task 3 — Merge Into Training Set

After both CSV files are generated:

1. Load existing training data from `data/labeled/`
2. Append `legal_synthetic.csv` and `education_synthetic.csv`
3. Verify new domain counts:
   - Legal: original 228 + 600 new = ~828 total
   - Education: original 331 + 500 new = ~831 total
   - All other domains: unchanged
4. Re-run train/val/test split (80/15/5) — **split AFTER merging, not before**
5. Save merged dataset to `data/labeled/merged_with_synthetic.csv`
6. Confirm class distribution is printed before training starts

**Critical:** Do NOT refit TF-IDF on the full merged set before the split.
Split first, then fit TF-IDF on training portion only. No data leakage.

---

## Task 4 — Retrain and Evaluate

Run full training on the merged dataset:

```bash
python src/ats_engine/trainer.py --data data/labeled/merged_with_synthetic.csv
python evaluation/ats_eval.py
```

Capture and report ALL metrics in the same format as the previous report:

```
=== RETRAIN VALIDATION REPORT (Post Synthetic Data) ===

METRICS
-------
Score MAE         : [value]  → [PASS / FAIL — target <8.0]
Score RMSE        : [value]
Domain F1         : [value]  → [PASS / FAIL — target >0.85]
Band Accuracy     : [value]%

PER-DOMAIN F1
-------------
IT / Software     : [value]
Non-IT / Mgmt     : [value]
Design / Creative : [value]
Healthcare        : [value]
Finance / Banking : [value]
Legal             : [value]   ← must improve from 0.68
Education         : [value]   ← must improve from 0.62

DELTA FROM PREVIOUS RUN
-----------------------
Domain F1 change  : [previous 0.7791] → [new value]  (+/- X)
Legal F1 change   : [previous 0.68]   → [new value]  (+/- X)
Education F1 change:[previous 0.62]  → [new value]  (+/- X)
Score MAE change  : [previous 2.33]   → [new value]  (must not regress above 5.0)

FRESHER FAIRNESS (re-run the same test)
----------------------------------------
Experienced Score : [value]
Fresher Score     : [value]
Gap               : [value] pts → [PASS / FAIL — target ≤20]

OVERALL STATUS
--------------
[ ] READY — Domain F1 > 0.85 and MAE < 8.0. Awaiting Sai's review.
[ ] NOT READY — [list specific failures]. Awaiting Sai's instructions.
```

**If Domain F1 is still below 0.85 after retraining:**
- Report the new per-domain breakdown
- Do NOT attempt another fix independently
- State clearly which domains are still underperforming and by how much
- Wait for Sai's instructions

**If Score MAE regresses above 5.0:**
- This is a regression — report it immediately
- Do not mark the run as passing even if Domain F1 improves

---

## What NOT to Do

- Do NOT run TFLite conversion — still deferred
- Do NOT modify `rubrics/feedback_rules.json`
- Do NOT touch any domain other than Legal and Education in the synthetic data
- Do NOT oversample existing Legal/Education data — generate NEW pairs only
- Do NOT use real people's names or real case references in synthetic text
- Do NOT touch `src/summary_generator/`
- Do NOT self-approve if either target metric fails — always report to Sai

---

## Definition of Done for This Injection

- [ ] `data/synthetic/legal_synthetic.csv` generated (600 pairs, domain=5)
- [ ] `data/synthetic/education_synthetic.csv` generated (500 pairs, domain=6)
- [ ] Score distribution verified for both files (matches tables above)
- [ ] Fresher representation ≥ 30% in Legal, ≥ 35% in Education
- [ ] Merged dataset saved to `data/labeled/merged_with_synthetic.csv`
- [ ] Retrain completed on merged dataset
- [ ] Full retrain report pasted in the format above
- [ ] Domain F1 > 0.85 confirmed (or failure escalated to Sai)
- [ ] Score MAE still < 8.0 confirmed (no regression)
- [ ] TFLite conversion NOT touched

---

*Injection prepared by: Claude Assistant on behalf of Sai*
*Previous report baseline: MAE=2.33, Domain F1=0.7791, Band Acc=95.31%*
*Next step after this injection passes: Sai reviews report → TFLite conversion decision.*
