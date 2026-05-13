"""
INJECTION-3-EVAL — Unified Model Full Evaluation
Evaluates all 3 heads: ATS scoring, Domain classification, RSG template classification.
Read-only — does NOT modify model weights.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from src.unified_engine.unified_model import build_unified_model
from src.unified_engine.data_loader import load_ats_data, load_rsg_data
from src.config import RSG_CSV_PATH

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "unified_model")
WEIGHTS_PATH      = os.path.join(UNIFIED_MODEL_DIR, "best_unified_weights.h5")
ATS_CSV           = os.path.join(PROJECT_ROOT, "data", "labeled", "merged_final.csv")
RSG_CSV           = str(RSG_CSV_PATH)
MAPPING_JSON      = os.path.join(UNIFIED_MODEL_DIR, "rsg_label_mapping.json")
REPORT_PATH       = os.path.join(UNIFIED_MODEL_DIR, "eval_report.json")

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
DOMAIN_SHORT = {
    0: "IT", 1: "NonIT", 2: "Design", 3: "Healthcare",
    4: "Finance", 5: "Legal", 6: "Education",
}

# ======================================================================
# TASK 1 — Load model and verify 3 heads
# ======================================================================
print("=" * 60)
print("TASK 1: Load unified model and verify output heads")
print("=" * 60)

model = build_unified_model()
print(f"Loading weights: {WEIGHTS_PATH}")
model.load_weights(WEIGHTS_PATH)
print("Weights loaded.\n")

# Smoke test
sample_r  = tf.constant(["Software engineer with 3 years Python experience"])
sample_jd = tf.constant(["Looking for Python developer with Django experience"])
outputs = model([sample_r, sample_jd], training=False)

assert len(outputs) == 3, f"HARD STOP: Expected 3 output heads, got {len(outputs)}"
print(f"✓ Output heads: {len(outputs)}")
print(f"  ATS shape  : {outputs[0].shape}")
print(f"  Domain shape: {outputs[1].shape}")
print(f"  RSG shape  : {outputs[2].shape}")
print()

# ======================================================================
# TASK 2 — ATS Head Evaluation
# ======================================================================
print("=" * 60)
print("TASK 2: ATS Head Evaluation")
print("=" * 60)

r_texts, jd_texts, ats_scores, domain_labels = load_ats_data(ATS_CSV)
ats_n = len(r_texts)
ats_idx = np.arange(ats_n)
_, val_ats = train_test_split(ats_idx, test_size=0.2, random_state=42)

print(f"ATS validation samples: {len(val_ats)}")

# Predict in batches
ats_preds = []
dom_preds = []
dom_true  = []

for i in range(0, len(val_ats), BATCH):
    batch_idx = val_ats[i:i+BATCH]
    r_batch = tf.constant(r_texts[batch_idx])
    jd_batch = tf.constant(jd_texts[batch_idx])
    out = model([r_batch, jd_batch], training=False)
    ats_preds.append(out[0].numpy().flatten())
    dom_preds.append(np.argmax(out[1].numpy(), axis=1))
    dom_true.append(domain_labels[batch_idx])

ats_pred_flat = np.concatenate(ats_preds) * 100     # back to 0-100
ats_true_flat = ats_scores[val_ats] * 100            # back to 0-100 (scores stored 0-1)
dom_pred_flat = np.concatenate(dom_preds)
dom_true_flat = np.concatenate(dom_true)

# MAE and RMSE
mae  = np.mean(np.abs(ats_pred_flat - ats_true_flat))
rmse = np.sqrt(np.mean((ats_pred_flat - ats_true_flat) ** 2))

# Band accuracy (bands: 0-24, 25-44, 45-64, 65-84, 85-100)
def to_band(score):
    if score < 25:   return 0
    if score < 45:   return 1
    if score < 65:   return 2
    if score < 85:   return 3
    return 4

true_bands = np.array([to_band(s) for s in ats_true_flat])
pred_bands = np.array([to_band(s) for s in ats_pred_flat])
band_acc = np.mean(true_bands == pred_bands)

# Fresher fairness gap
# Heuristic: fresher = score < 40 (0-100 scale), experienced = score >= 60
fresher_mask = ats_true_flat < 40
experienced_mask = ats_true_flat >= 60

if fresher_mask.any() and experienced_mask.any():
    mean_pred_fresher = np.mean(ats_pred_flat[fresher_mask])
    mean_pred_exp     = np.mean(ats_pred_flat[experienced_mask])
    # Also compute based on error gap
    fresher_error = np.mean(np.abs(ats_pred_flat[fresher_mask] - ats_true_flat[fresher_mask]))
    exp_error     = np.mean(np.abs(ats_pred_flat[experienced_mask] - ats_true_flat[experienced_mask]))
    fairness_gap  = abs(fresher_error - exp_error)
else:
    fairness_gap = 0.0

print(f"\n  ATS MAE          : {mae:.2f}  (target < 8.0)  {'✓' if mae < 8.0 else '✗ FAIL'}")
print(f"  ATS RMSE         : {rmse:.2f}")
print(f"  Band Accuracy    : {band_acc*100:.1f}%  (target > 80%)  {'✓' if band_acc > 0.80 else '✗'}")
print(f"  Fresher gap      : {fairness_gap:.2f} pts  (target < 20)  {'✓' if fairness_gap < 20 else '✗'}")

if mae > 8.0:
    print("\n⛔ HARD STOP: ATS MAE > 8.0 — report to Sai")
    sys.exit(1)

# ======================================================================
# TASK 3 — Domain Head Evaluation
# ======================================================================
print("\n" + "=" * 60)
print("TASK 3: Domain Head Evaluation")
print("=" * 60)

macro_f1 = f1_score(dom_true_flat, dom_pred_flat, average="macro")
per_domain_f1 = f1_score(dom_true_flat, dom_pred_flat, average=None,
                         labels=list(range(7)))

print(f"\n  Domain Macro F1  : {macro_f1:.4f}  (target > 0.85)  {'✓' if macro_f1 > 0.85 else '✗'}")
print(f"\n  Per-domain F1:")
per_domain_f1_dict = {}
any_critical_fail = False
for idx in range(7):
    name = DOMAIN_NAMES.get(idx, f"Domain-{idx}")
    short = DOMAIN_SHORT.get(idx, f"D{idx}")
    f1_val = per_domain_f1[idx] if idx < len(per_domain_f1) else 0.0
    status = "✓" if f1_val > 0.80 else ("⛔" if f1_val < 0.75 else "✗")
    if f1_val < 0.75:
        any_critical_fail = True
    print(f"    {name:20s}: {f1_val:.4f}  {status}")
    per_domain_f1_dict[short] = round(float(f1_val), 4)

if any_critical_fail:
    print("\n⛔ HARD STOP: Per-domain F1 < 0.75 detected — report to Sai")
    sys.exit(1)

# Detailed classification report
print("\n  Full classification report:")
target_names = [DOMAIN_NAMES[i] for i in range(7)]
# Only include classes that actually appear in the data
present_labels = sorted(set(dom_true_flat) | set(dom_pred_flat))
present_names = [DOMAIN_NAMES.get(l, f"Domain-{l}") for l in present_labels]
print(classification_report(dom_true_flat, dom_pred_flat,
                            labels=present_labels, target_names=present_names))

# ======================================================================
# TASK 4 — RSG Head Evaluation
# ======================================================================
print("=" * 60)
print("TASK 4: RSG Head Evaluation")
print("=" * 60)

with open(MAPPING_JSON) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}

profile_texts, template_ids = load_rsg_data(RSG_CSV)

# Filter to mapped IDs
valid = np.array([int(tid) in id_to_idx for tid in template_ids])
prof_f = profile_texts[valid]
tmpl_f = np.array([id_to_idx[int(tid)] for tid in template_ids[valid]])

# Same split as training (80/20)
split_r = int(0.8 * len(prof_f))
rsg_val_prof = prof_f[split_r:]
rsg_val_tmpl = tmpl_f[split_r:]

print(f"RSG validation samples: {len(rsg_val_prof)}")

# Predict in batches
rsg_preds = []
for i in range(0, len(rsg_val_prof), BATCH):
    batch_prof = tf.constant(rsg_val_prof[i:i+BATCH])
    out = model([batch_prof, batch_prof], training=False)
    rsg_preds.append(np.argmax(out[2].numpy(), axis=1))

rsg_pred_flat = np.concatenate(rsg_preds)
rsg_overall_acc = np.mean(rsg_pred_flat == rsg_val_tmpl)

print(f"\n  RSG Overall Accuracy: {rsg_overall_acc*100:.1f}%  (target > 85%)  {'✓' if rsg_overall_acc > 0.85 else '✗'}")

if rsg_overall_acc < 0.60:
    print("\n⛔ HARD STOP: RSG overall accuracy < 60% — report to Sai")
    sys.exit(1)

# Per-domain accuracy for RSG (group by template index ranges)
# Since we don't have a domain column in RSG data, group by template index
unique_templates = sorted(set(rsg_val_tmpl))
print(f"\n  Per-template-group accuracy (sample):")
per_domain_rsg = {}

# Group templates into rough domains based on index ranges
# (This is approximate — based on template distribution)
correct_per_tmpl = {}
total_per_tmpl = {}
for pred, true in zip(rsg_pred_flat, rsg_val_tmpl):
    total_per_tmpl[true] = total_per_tmpl.get(true, 0) + 1
    if pred == true:
        correct_per_tmpl[true] = correct_per_tmpl.get(true, 0) + 1

# Show accuracy for templates with enough samples
template_accs = {}
for tmpl in sorted(total_per_tmpl.keys()):
    n = total_per_tmpl[tmpl]
    c = correct_per_tmpl.get(tmpl, 0)
    acc = c / n if n > 0 else 0
    template_accs[str(tmpl)] = round(acc, 4)
    if n >= 3:  # Only print templates with enough samples
        print(f"    Template {tmpl:3d}: {acc*100:5.1f}%  (n={n})")

per_domain_rsg = template_accs

# ======================================================================
# TASK 5 — Generate eval report
# ======================================================================
print("\n" + "=" * 60)
print("TASK 5: Generate eval_report.json")
print("=" * 60)

all_targets_met = bool(
    mae < 8.0 and
    band_acc > 0.80 and
    fairness_gap < 20 and
    macro_f1 > 0.85 and
    all(f1 > 0.80 for f1 in per_domain_f1[:7]) and
    rsg_overall_acc > 0.85
)

report = {
    "ats_head": {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "band_accuracy": round(float(band_acc), 4),
        "fresher_fairness_gap": round(float(fairness_gap), 4),
    },
    "domain_head": {
        "macro_f1": round(float(macro_f1), 4),
        "per_domain_f1": per_domain_f1_dict,
    },
    "rsg_head": {
        "overall_accuracy": round(float(rsg_overall_acc), 4),
        "per_domain_accuracy": per_domain_rsg,
    },
    "all_targets_met": all_targets_met,
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved: {REPORT_PATH}")
print(f"All targets met: {all_targets_met}")

# ── Final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("UNIFIED MODEL EVALUATION SUMMARY")
print("=" * 60)
print(f"  ATS MAE            : {mae:.2f}          {'✅' if mae < 8.0 else '❌'}")
print(f"  ATS RMSE           : {rmse:.2f}")
print(f"  Band Accuracy      : {band_acc*100:.1f}%        {'✅' if band_acc > 0.80 else '❌'}")
print(f"  Fresher Gap        : {fairness_gap:.2f} pts   {'✅' if fairness_gap < 20 else '❌'}")
print(f"  Domain Macro F1    : {macro_f1:.4f}       {'✅' if macro_f1 > 0.85 else '❌'}")
for idx in range(7):
    short = DOMAIN_SHORT.get(idx, f"D{idx}")
    f1_val = per_domain_f1[idx] if idx < len(per_domain_f1) else 0.0
    print(f"    {short:12s} F1  : {f1_val:.4f}       {'✅' if f1_val > 0.80 else '❌'}")
print(f"  RSG Accuracy       : {rsg_overall_acc*100:.1f}%        {'✅' if rsg_overall_acc > 0.85 else '❌'}")
print()
print("Send this output to Sai before running INJECTION-4-TFLITE.")
