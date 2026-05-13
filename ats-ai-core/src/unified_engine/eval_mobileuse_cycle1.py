"""
M-3: Full validation gate evaluation — MobileUSE Cycle 1
Evaluates best_model_mobileuse_cycle1.h5 against all gates.
Read-only — does NOT modify model weights.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import re
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, ".")
from unified_model import build_unified_model

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ATS_CSV      = PROJECT_ROOT / "data" / "labeled" / "merged_final.csv"
WEIGHTS_PATH = PROJECT_ROOT / "model" / "ats_model" / "best_model_mobileuse_cycle1.h5"
EVAL_DIR     = PROJECT_ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV   = EVAL_DIR / "eval_report_mobileuse_cycle1.csv"

BATCH = 64

DOMAIN_NAMES = {
    0: "IT/Software",
    1: "Non-IT/Management",
    2: "Design/Creative",
    3: "Healthcare",
    4: "Finance/Banking",
    5: "Legal",
    6: "Education",
}

# ── Fresher detection regex (from existing ats_eval.py logic) ──────────
_fresher_re = re.compile(
    r"fresher|fresh\s*graduate|entry[- ]level|intern(?:ship)?|"
    r"0\s*years?\s*(?:of\s*)?(?:experience|professional)|"
    r"final[- ]year\s*student|looking for entry|recent(?:ly)?\s*graduat|"
    r"no\s*(?:prior\s*)?experience|trainee|articleship|pupillage",
    re.IGNORECASE,
)

# ── Score band function (from config.py) ───────────────────────────────
def get_score_band(score):
    s = int(round(score))
    if s >= 85: return "Excellent"
    if s >= 65: return "Good"
    if s >= 45: return "Moderate"
    if s >= 25: return "Weak"
    return "Poor"

# ── UTF-8 stdout ───────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ======================================================================
# TASK 1: Load model + weights
# ======================================================================
print("=" * 65)
print("  M-3: FULL VALIDATION GATE EVALUATION — MobileUSE Cycle 1")
print("=" * 65)

print(f"\n[1/5] Building model and loading weights...")
model = build_unified_model()
print(f"  Loading: {WEIGHTS_PATH}")
model.load_weights(str(WEIGHTS_PATH))
print("  Weights loaded.")

# Smoke check
sample_out = model(
    [tf.constant(["test resume"]), tf.constant(["test jd"])],
    training=False
)
assert len(sample_out) == 3, f"Expected 3 output heads, got {len(sample_out)}"
print(f"  Output heads: 3 (ATS, Domain, RSG) — OK")

# ======================================================================
# TASK 2: Load data with SAME 75/15/10 split from M-2
# ======================================================================
print(f"\n[2/5] Loading test data (same split as training)...")
df = pd.read_csv(str(ATS_CSV)).dropna()
print(f"  Total rows: {len(df)}")

resume_texts  = df["resume_text"].astype(str).values
jd_texts      = df["jd_text"].astype(str).values
ats_scores    = (df["score"].astype(float) / 100.0).values.astype("float32")
domain_labels = df["domain_index"].astype(int).values.astype("int32")

# Reproduce exact same split as M-2 training
idx = np.arange(len(df))
train_idx, temp_idx = train_test_split(idx, test_size=0.25, random_state=42)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.40, random_state=42)

print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
print(f"  Evaluating on TEST set ({len(test_idx)} pairs)")

# ======================================================================
# TASK 3: Run inference on test set
# ======================================================================
print(f"\n[3/5] Running inference on {len(test_idx)} test pairs...")

ats_preds = []
dom_preds = []

for i in range(0, len(test_idx), BATCH):
    batch_idx = test_idx[i:i+BATCH]
    r_batch  = tf.constant(resume_texts[batch_idx])
    jd_batch = tf.constant(jd_texts[batch_idx])
    out = model([r_batch, jd_batch], training=False)
    ats_preds.append(out[0].numpy().flatten())
    dom_preds.append(np.argmax(out[1].numpy(), axis=1))

ats_pred_100 = np.concatenate(ats_preds) * 100     # 0-100 scale
ats_true_100 = ats_scores[test_idx] * 100           # 0-100 scale
dom_pred_flat = np.concatenate(dom_preds)
dom_true_flat = domain_labels[test_idx]

# ======================================================================
# GATE A: Overall MAE (0-100 scale) — gate < 8.0
# ======================================================================
mae  = float(np.mean(np.abs(ats_pred_100 - ats_true_100)))
rmse = float(np.sqrt(np.mean((ats_pred_100 - ats_true_100) ** 2)))
mae_pass = mae < 8.0

# ======================================================================
# GATE B: Band Accuracy — gate > 80%
# ======================================================================
true_bands = np.array([get_score_band(s) for s in ats_true_100])
pred_bands = np.array([get_score_band(s) for s in ats_pred_100])
band_acc = float(np.mean(true_bands == pred_bands))
band_pass = band_acc > 0.80

# ======================================================================
# GATE C: Domain F1 macro-average — gate > 0.85
#         Per-domain F1 — each > 0.80
# ======================================================================
macro_f1 = float(f1_score(dom_true_flat, dom_pred_flat, average="macro", zero_division=0))
per_domain_f1 = f1_score(dom_true_flat, dom_pred_flat, average=None,
                         labels=list(range(7)), zero_division=0)
f1_pass = macro_f1 > 0.85
per_domain_all_pass = all(f > 0.80 for f in per_domain_f1[:7])

# ======================================================================
# GATE D: Fresher Fairness gap
# ======================================================================
test_resumes = resume_texts[test_idx]
fresher_mask = np.array([bool(_fresher_re.search(t)) for t in test_resumes])
experienced_mask = ~fresher_mask

fresher_mean = float(ats_pred_100[fresher_mask].mean()) if fresher_mask.any() else None
experienced_mean = float(ats_pred_100[experienced_mask].mean()) if experienced_mask.any() else None

if fresher_mean is not None and experienced_mean is not None:
    fresher_gap = abs(experienced_mean - fresher_mean)
    fresher_direction = "freshers lower" if fresher_mean < experienced_mean else "freshers higher"
    fresher_lower = fresher_mean < experienced_mean
else:
    fresher_gap = 0.0
    fresher_direction = "N/A"
    fresher_lower = False

fresher_pass = fresher_gap <= 20 and not fresher_lower

# ======================================================================
# PRINT RESULTS — Task 4
# ======================================================================
print(f"\n[4/5] Gate evaluation results...")

print(f"\n{'=' * 65}")
print(f"  GATE A: ATS MAE")
print(f"{'=' * 65}")
print(f"  MAE  : {mae:.2f}  (gate < 8.0)  {'[PASS]' if mae_pass else '[FAIL]'}")
print(f"  RMSE : {rmse:.2f}")

print(f"\n{'=' * 65}")
print(f"  GATE B: Band Accuracy")
print(f"{'=' * 65}")
print(f"  Band Accuracy: {band_acc*100:.1f}%  (gate > 80%)  {'[PASS]' if band_pass else '[FAIL]'}")

print(f"\n{'=' * 65}")
print(f"  GATE C: Domain F1")
print(f"{'=' * 65}")
print(f"  Macro F1: {macro_f1:.4f}  (gate > 0.85)  {'[PASS]' if f1_pass else '[FAIL]'}")
print(f"\n  Per-domain F1 table:")
print(f"  {'Domain':<22s}  {'F1':>8s}  {'Gate':>6s}  {'Status':>6s}")
print(f"  {'-'*22}  {'-'*8}  {'-'*6}  {'-'*6}")
for idx in range(7):
    name = DOMAIN_NAMES.get(idx, f"Domain-{idx}")
    f1_val = float(per_domain_f1[idx])
    status = "[PASS]" if f1_val > 0.80 else "[FAIL]"
    print(f"  {name:<22s}  {f1_val:>8.4f}  {'>0.80':>6s}  {status:>6s}")

# Full classification report
print(f"\n  Full classification report:")
present_labels = sorted(set(dom_true_flat) | set(dom_pred_flat))
present_names = [DOMAIN_NAMES.get(l, f"Domain-{l}") for l in present_labels]
print(classification_report(dom_true_flat, dom_pred_flat,
                            labels=present_labels, target_names=present_names,
                            zero_division=0))

print(f"\n{'=' * 65}")
print(f"  GATE D: Fresher Fairness")
print(f"{'=' * 65}")
n_fresher = int(fresher_mask.sum())
n_experienced = int(experienced_mask.sum())
print(f"  Freshers detected:    {n_fresher}  (of {len(test_idx)} test pairs)")
print(f"  Experienced detected: {n_experienced}")
if fresher_mean is not None and experienced_mean is not None:
    print(f"  Fresher mean ATS:     {fresher_mean:.2f}")
    print(f"  Experienced mean ATS: {experienced_mean:.2f}")
    print(f"  Gap:                  {fresher_gap:.1f} pts  ({fresher_direction})")
    print(f"  Gate (gap <= 20, freshers not systematically lower): {'[PASS]' if fresher_pass else '[FAIL]'}")
else:
    print(f"  Insufficient data for fresher/experienced split.")

# ── Single-line gate summary ──────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  GATE SUMMARY")
print(f"{'=' * 65}")
summary_line = (
    f"MAE: {mae:.2f} {'[PASS]' if mae_pass else '[FAIL]'}  |  "
    f"F1: {macro_f1:.4f} {'[PASS]' if f1_pass else '[FAIL]'}  |  "
    f"Band: {band_acc*100:.1f}% {'[PASS]' if band_pass else '[FAIL]'}  |  "
    f"Fresher: {fresher_gap:.1f} pts {'[PASS]' if fresher_pass else '[FAIL]'}"
)
print(f"  {summary_line}")

all_pass = mae_pass and f1_pass and band_pass and fresher_pass and per_domain_all_pass
print(f"\n  ALL GATES: {'PASS — proceed to M-4' if all_pass else 'FAIL — report to Sai'}")
print(f"{'=' * 65}")

# ======================================================================
# TASK 5: Save eval report CSV
# ======================================================================
print(f"\n[5/5] Saving evaluation report...")

report_rows = []
# Main metrics row
row = {
    "metric": "overall",
    "mae": round(mae, 4),
    "rmse": round(rmse, 4),
    "band_accuracy": round(band_acc, 4),
    "domain_macro_f1": round(macro_f1, 4),
    "fresher_gap_pts": round(fresher_gap, 1),
    "fresher_direction": fresher_direction,
    "fresher_mean": round(fresher_mean, 2) if fresher_mean else None,
    "experienced_mean": round(experienced_mean, 2) if experienced_mean else None,
    "mae_pass": mae_pass,
    "f1_pass": f1_pass,
    "band_pass": band_pass,
    "fresher_pass": fresher_pass,
    "all_gates_pass": all_pass,
    "test_samples": len(test_idx),
}
report_rows.append(row)

# Per-domain F1 rows
for idx in range(7):
    name = DOMAIN_NAMES.get(idx, f"Domain-{idx}")
    f1_val = float(per_domain_f1[idx])
    report_rows.append({
        "metric": f"domain_f1_{name}",
        "domain_macro_f1": round(f1_val, 4),
        "f1_pass": f1_val > 0.80,
    })

report_df = pd.DataFrame(report_rows)
report_df.to_csv(str(REPORT_CSV), index=False)
print(f"  Saved: {REPORT_CSV}")
print(f"  Rows:  {len(report_df)}")

print(f"\nM-3 COMPLETE — evaluation finished.")
