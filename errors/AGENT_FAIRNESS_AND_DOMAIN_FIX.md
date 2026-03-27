# AGENT PROMPT INJECTION — Fresher Fairness Fix + Domain F1 Recovery
**Priority: CRITICAL | Sprint: 6 — Two-Phase Fix**
**Author: Sai | Approved approach: Fairness first, then combined data + loss tuning**

---

## Situation Summary

Post-synthetic-data retrain produced two regressions:

| Issue | Previous | Current | Status |
|-------|----------|---------|--------|
| Fresher Fairness Gap | 10.1 pts ✓ | 43.3 pts (inverted) | CRITICAL FAIL |
| Domain F1 (IT) | 0.7625 | 0.7500 | Regressed |
| Domain F1 (Finance) | 0.7672 | 0.7019 | Regressed |
| Domain F1 (macro) | 0.7791 | 0.8218 | Still below 0.85 |

Legal (0.88) and Education (0.83) are now strong — do not touch their data.

**Execute Phase 1 completely before starting Phase 2.**
Do not combine phases or skip ahead.

---

# PHASE 1 — Fix Fresher Fairness (Do This First)

## What Happened

The synthetic Legal and Education pairs overloaded the model with high-scoring
fresher profiles. The model learned: fresher resume = high ATS score, regardless
of actual alignment. This inverted the fairness relationship.

The fix is a score audit of the synthetic CSVs — not a model change.

---

## Phase 1 — Task 1: Audit Synthetic Fresher Score Labels

**Files to audit:**
- `data/synthetic/legal_synthetic.csv`
- `data/synthetic/education_synthetic.csv`

**Rule: A fresher profile must only receive a high score if its skills and
keywords genuinely align with the JD. Experience level alone must never
determine score direction.**

Apply these correction rules to every row where the resume is a fresher profile
(student, fresh graduate, internship-only, 0 years experience):

| Condition | Correct Score Range |
|-----------|-------------------|
| Fresher + strong skill/keyword match to JD | 60–85 |
| Fresher + moderate skill/keyword match | 35–60 |
| Fresher + weak skill/keyword match | 15–35 |
| Fresher + mismatched domain entirely | 5–20 |

**The same rules apply to experienced profiles:**

| Condition | Correct Score Range |
|-----------|-------------------|
| Experienced + strong skill/keyword match | 70–95 |
| Experienced + moderate skill/keyword match | 45–70 |
| Experienced + weak skill/keyword match | 20–45 |
| Experienced + mismatched domain | 5–25 |

The gap between a fresher and an experienced candidate with identical skill
alignment must not exceed 15 points. If it does, adjust the fresher score up
(do not reduce the experienced score).

**After audit, verify:**
```python
import pandas as pd

legal = pd.read_csv('data/synthetic/legal_synthetic.csv')
edu   = pd.read_csv('data/synthetic/education_synthetic.csv')

# Flag any fresher rows with score > 85 (likely mislabeled)
# Flag any experienced rows with score < 20 (likely mislabeled)
# Print mean score for fresher vs experienced in each file
```

Report the mean score for fresher vs experienced profiles in both files before
proceeding. They must be within 10–15 points of each other on average.

---

## Phase 1 — Task 2: Retrain on Corrected Synthetic Data

After score audit and correction:

1. Re-merge corrected synthetic files with original labeled data
2. Save to `data/labeled/merged_corrected.csv`
3. Retrain:

```bash
python src/ats_engine/trainer.py --data data/labeled/merged_corrected.csv
```

4. Run ONLY the fresher fairness check first — do not run full eval yet:

```python
from src.ats_engine.inference import run_inference

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
print(f"Gap               : {r_exp['ats_score'] - r_fresher['ats_score']:.1f} pts")
print(f"Direction correct : {'YES' if r_exp['ats_score'] > r_fresher['ats_score'] else 'NO — still inverted'}")
```

**Gate:** Experienced score must be HIGHER than fresher score, and gap must be
≤ 20 points. If the direction is still inverted (fresher > experienced), stop
and report — do not proceed to Phase 2.

---

# PHASE 2 — Fix Domain F1: IT + Finance Data + Loss Weight Tuning

Only begin Phase 2 after Phase 1 fresher fairness check passes.

## Phase 2 — Task 1: Generate Synthetic IT + Finance Pairs

### IT / Software — 400 new pairs
**Output:** `data/synthetic/it_supplemental.csv` | Domain label: 0

Archetypes (80 pairs each):
- **Backend Engineer:** Python/Java/Node, REST APIs, databases, Docker, microservices
- **Frontend Developer:** React/Vue/Angular, TypeScript, CSS, responsive design, testing
- **Data Engineer / ML:** Python, TensorFlow/PyTorch, pandas, SQL, pipeline design
- **DevOps / Cloud:** AWS/GCP/Azure, Kubernetes, Terraform, CI/CD, Linux
- **Mobile Developer:** Flutter/React Native/Android/iOS, APIs, app deployment

Score distribution: 15% Excellent / 25% Good / 30% Moderate / 20% Weak / 10% Poor
Fresher representation: ≥ 25% (CS students, bootcamp grads, project-only experience)

### Finance / Banking — 400 new pairs
**Output:** `data/synthetic/finance_supplemental.csv` | Domain label: 4

Archetypes (80 pairs each):
- **Investment Banking Analyst:** DCF, financial modelling, Excel, Bloomberg, M&A
- **Risk Analyst:** credit risk, VaR, Basel norms, SQL, risk reporting
- **Financial Analyst / FP&A:** budgeting, forecasting, Excel, ERP systems, variance analysis
- **Chartered Accountant / Audit:** IFRS, audit procedures, tax compliance, Tally, Big 4
- **FinTech / Quant:** Python, algorithmic trading, data analysis, financial APIs, statistics

Score distribution: 15% Excellent / 25% Good / 30% Moderate / 20% Weak / 10% Poor
Fresher representation: ≥ 25% (finance graduates, CA articleship, internship-only)

---

## Phase 2 — Task 2: Tune Loss Weights in `config.py`

**File:** `src/config.py`

The previous training used domain loss weight = 0.80. IT and Finance degraded,
suggesting domain gradients are not being applied evenly across all 7 classes.

Apply these changes:

```python
# Previous values (for reference — do not restore these)
# SCORE_LOSS_WEIGHT = 0.20
# DOMAIN_LOSS_WEIGHT = 0.80

# Updated values
SCORE_LOSS_WEIGHT = 0.35      # Increased — prevents MAE regression
DOMAIN_LOSS_WEIGHT = 0.65     # Reduced slightly — was over-rotating on domain
DOMAIN_CLASS_WEIGHTS = {       # Per-class weighting to protect IT and Finance
    0: 1.4,   # IT — needs more gradient signal
    1: 0.8,   # Non-IT — well represented, reduce weight
    2: 0.9,   # Design — strong, slight reduction
    3: 1.0,   # Healthcare — stable
    4: 1.5,   # Finance — needs more gradient signal
    5: 0.9,   # Legal — now well represented
    6: 1.0,   # Education — now well represented
}
```

Also update in `trainer.py` — confirm per-sample class weights are computed
from `DOMAIN_CLASS_WEIGHTS` and applied correctly to the training loop.
(This was fixed in the previous session — verify it is still intact.)

---

## Phase 2 — Task 3: Retrain on Full Merged Dataset

Merge all data sources:
1. `data/labeled/merged_corrected.csv` (from Phase 1)
2. `data/synthetic/it_supplemental.csv`
3. `data/synthetic/finance_supplemental.csv`

Save to: `data/labeled/merged_final.csv`

Verify domain counts before training — print this table:

```
Domain distribution in merged_final.csv:
  IT / Software         : [count]
  Non-IT / Management   : [count]
  Design / Creative     : [count]
  Healthcare            : [count]
  Finance / Banking     : [count]
  Legal                 : [count]
  Education             : [count]
  TOTAL                 : [count]
```

No domain should have fewer than 600 samples or more than 2,500 samples.
If any domain is outside this range, report it before training.

Then retrain:
```bash
python src/ats_engine/trainer.py --data data/labeled/merged_final.csv
python evaluation/ats_eval.py
```

---

## Phase 2 — Task 4: Full Validation Report

```
=== RETRAIN VALIDATION REPORT (Post Fairness + Domain Fix) ===

METRICS
-------
Score MAE         : [value]  → [PASS / FAIL — target <8.0]
Score RMSE        : [value]
Domain F1         : [value]  → [PASS / FAIL — target >0.85]
Band Accuracy     : [value]%

PER-DOMAIN F1
-------------
IT / Software     : [value]   ← must recover from 0.75
Non-IT / Mgmt     : [value]
Design / Creative : [value]
Healthcare        : [value]
Finance / Banking : [value]   ← must recover from 0.70
Legal             : [value]   ← must hold above 0.85
Education         : [value]   ← must hold above 0.80

DELTA FROM PREVIOUS RUN
-----------------------
Domain F1 change  : 0.8218 → [new]  (+/- X)
IT F1 change      : 0.7500 → [new]  (+/- X)
Finance F1 change : 0.7019 → [new]  (+/- X)
Legal F1 held     : 0.8837 → [new]  (must not drop below 0.82)
Education F1 held : 0.8252 → [new]  (must not drop below 0.79)
Score MAE change  : 3.87   → [new]  (must not regress above 6.0)

FRESHER FAIRNESS
----------------
Experienced Score : [value]
Fresher Score     : [value]
Gap               : [value] pts → [PASS / FAIL — target ≤20, experienced > fresher]
Direction         : [Experienced higher / Fresher higher — must be Experienced higher]

OVERALL STATUS
--------------
[ ] READY — Domain F1 > 0.85, MAE < 8.0, Fairness gap ≤ 20 and direction correct.
            Awaiting Sai's review before TFLite conversion.
[ ] NOT READY — [list specific failures]. Awaiting Sai's instructions.
```

---

## Regression Guards — Hard Stops

If any of these occur, stop immediately and report. Do not attempt further fixes:

| Metric | Hard Stop Condition |
|--------|-------------------|
| Score MAE | Regresses above 6.0 |
| Legal F1 | Drops below 0.82 |
| Education F1 | Drops below 0.79 |
| Fresher direction | Fresher score still > Experienced score |
| Any domain F1 | Drops below 0.65 |

---

## What NOT to Do

- Do NOT start Phase 2 if Phase 1 fairness check fails
- Do NOT run TFLite conversion — still deferred
- Do NOT modify `rubrics/feedback_rules.json`
- Do NOT touch `src/summary_generator/`
- Do NOT self-approve if any target metric fails

---

## Definition of Done

**Phase 1:**
- [ ] Synthetic CSV score labels audited and corrected
- [ ] Mean fresher vs experienced score within 10–15 pts in both files
- [ ] Retrain on corrected data complete
- [ ] Fresher fairness: gap ≤ 20, experienced score > fresher score

**Phase 2:**
- [ ] IT supplemental (400 pairs) and Finance supplemental (400 pairs) generated
- [ ] Loss weights updated in `config.py`
- [ ] Domain counts in `merged_final.csv` all between 600–2,500
- [ ] Retrain on `merged_final.csv` complete
- [ ] Domain F1 > 0.85
- [ ] Score MAE < 8.0
- [ ] Full validation report pasted in format above

---

*Injection prepared by: Claude Assistant on behalf of Sai*
*Baseline: MAE=3.87, Domain F1=0.8218, Fairness gap=43.3pts (inverted)*
*Next step after all checks pass: Sai reviews → TFLite conversion decision*
