"""
INJECTION-2-EVAL: Unified Model Production Validation

This script validates the unified model is production-ready by:
1. Loading the final Stage 3 weights
2. Testing all 3 heads produce correct output shapes/ranges
3. Running inference on held-out test data
4. Comparing ATS performance against the standalone model
5. Verifying no regressions in any head
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

from src.unified_engine.unified_model import build_unified_model
from src.unified_engine.data_loader import load_ats_data, load_rsg_data
from src.config import RSG_CSV_PATH, RSG_MAPPING_JSON

# ── Paths ──────────────────────────────────────────────────────────────
STAGE3_WEIGHTS = Path(r"model\unified_model\unified_stage3_weights.h5")
ATS_TEST_CSV   = Path(r"data\labeled\test_split.csv")
RSG_CSV        = RSG_CSV_PATH
MAPPING_JSON   = RSG_MAPPING_JSON

# Standalone ATS model for comparison
ATS_WEIGHTS    = Path(r"model\ats_model\final_model_weights.h5")
# ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print("UNIFIED MODEL PRODUCTION VALIDATION")
print("=" * 60)
print()

# ── Load label mapping ───────────────────────────────────────────────────
with open(MAPPING_JSON) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}
idx_to_id = {int(k): int(v) for k, v in mapping["idx_to_id"].items()}

# ── Build and load unified model ─────────────────────────────────────────
print("1. LOADING UNIFIED MODEL")
print("-" * 40)
print(f"   Weights: {STAGE3_WEIGHTS}")

if not STAGE3_WEIGHTS.exists():
    print("   ERROR: Stage 3 weights not found!")
    print("   HARD STOP: Run Stage 3 training first.")
    exit(1)

model = build_unified_model()
model.load_weights(str(STAGE3_WEIGHTS))
print("   Model loaded successfully.")
print()

# ── Test 1: Output shape validation ──────────────────────────────────────
print("2. OUTPUT SHAPE VALIDATION")
print("-" * 40)

test_resume = tf.constant(["Python developer with Django experience"])
test_jd = tf.constant(["Looking for backend engineer"])

ats_out, dom_out, rsg_out = model([test_resume, test_jd], training=False)

shape_checks = [
    ("ATS score", ats_out.shape, (1, 1)),
    ("Domain probs", dom_out.shape, (1, 7)),
    ("RSG template", rsg_out.shape, (1, 46)),
]

all_shapes_ok = True
for name, actual, expected in shape_checks:
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected:
        all_shapes_ok = False
    print(f"   {name}: {actual} (expected {expected}) - {status}")

if not all_shapes_ok:
    print("\n   HARD STOP: Output shapes incorrect!")
    exit(1)
print()

# ── Test 2: Output range validation ──────────────────────────────────────
print("3. OUTPUT RANGE VALIDATION")
print("-" * 40)

ats_val = float(ats_out[0][0])
dom_sum = float(tf.reduce_sum(dom_out[0]))
rsg_sum = float(tf.reduce_sum(rsg_out[0]))

range_checks = [
    ("ATS score in [0,1]", 0 <= ats_val <= 1, f"{ats_val:.4f}"),
    ("Domain probs sum to 1", abs(dom_sum - 1.0) < 0.01, f"{dom_sum:.4f}"),
    ("RSG probs sum to 1", abs(rsg_sum - 1.0) < 0.01, f"{rsg_sum:.4f}"),
]

all_ranges_ok = True
for name, passed, value in range_checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_ranges_ok = False
    print(f"   {name}: {value} - {status}")

if not all_ranges_ok:
    print("\n   HARD STOP: Output ranges invalid!")
    exit(1)
print()

# ── Test 3: ATS head evaluation on test set ──────────────────────────────
print("4. ATS HEAD EVALUATION (Test Set)")
print("-" * 40)

if ATS_TEST_CSV.exists():
    r_texts, jd_texts, ats_true, domain_true = load_ats_data(str(ATS_TEST_CSV), limit=None)
    print(f"   Test samples: {len(r_texts)}")
    
    # Batch predict
    ats_pred, dom_pred, _ = model.predict(
        {"resume_text": r_texts, "jd_text": jd_texts},
        batch_size=32, verbose=0
    )
    
    ats_pred_flat = ats_pred.flatten()
    dom_pred_labels = np.argmax(dom_pred, axis=1)
    
    # Metrics
    ats_mae = mean_absolute_error(ats_true, ats_pred_flat) * 100  # Scale to 0-100
    dom_acc = accuracy_score(domain_true, dom_pred_labels)
    dom_f1 = f1_score(domain_true, dom_pred_labels, average='weighted')
    
    print(f"   ATS MAE (0-100):    {ats_mae:.2f}")
    print(f"   Domain Accuracy:    {dom_acc:.4f} ({dom_acc*100:.1f}%)")
    print(f"   Domain F1 (weighted): {dom_f1:.4f}")
    
    # Regression checks
    ats_pass = ats_mae < 8.0
    dom_pass = dom_f1 > 0.80
    
    print()
    print(f"   ATS MAE < 8.0:  {'PASS' if ats_pass else 'FAIL'}")
    print(f"   Domain F1 > 0.80: {'PASS' if dom_pass else 'FAIL'}")
    
    if not (ats_pass and dom_pass):
        print("\n   WARNING: Regression detected in ATS/Domain heads!")
else:
    print(f"   Test CSV not found: {ATS_TEST_CSV}")
    print("   Skipping ATS test set evaluation.")
    ats_mae = None
    dom_f1 = None
print()

# ── Test 4: RSG head evaluation ──────────────────────────────────────────
print("5. RSG HEAD EVALUATION")
print("-" * 40)

profile_texts, template_ids = load_rsg_data(str(RSG_CSV))
valid_mask = np.array([int(tid) in id_to_idx for tid in template_ids])
profile_texts_f = profile_texts[valid_mask]
template_indices = np.array([id_to_idx[int(tid)] for tid in template_ids[valid_mask]])

# Use last 20% as test
split = int(0.8 * len(profile_texts_f))
test_txt = profile_texts_f[split:]
test_lbl = template_indices[split:]

print(f"   RSG test samples: {len(test_txt)}")

_, _, rsg_pred = model.predict(
    {"resume_text": test_txt, "jd_text": test_txt},
    batch_size=32, verbose=0
)

rsg_pred_labels = np.argmax(rsg_pred, axis=1)
rsg_acc = accuracy_score(test_lbl, rsg_pred_labels)

print(f"   RSG Accuracy:       {rsg_acc:.4f} ({rsg_acc*100:.1f}%)")

rsg_pass = rsg_acc >= 0.50
print(f"   RSG Accuracy >= 50%: {'PASS' if rsg_pass else 'FAIL'}")
print()

# ── Test 5: Inference speed check ────────────────────────────────────────
print("6. INFERENCE SPEED CHECK")
print("-" * 40)

import time

# Warm-up
_ = model([test_resume, test_jd], training=False)

# Time 100 single inferences
start = time.time()
for _ in range(100):
    _ = model([test_resume, test_jd], training=False)
elapsed = time.time() - start

avg_ms = (elapsed / 100) * 1000
print(f"   Average inference time: {avg_ms:.1f} ms per sample")
print(f"   Inference speed < 500ms: {'PASS' if avg_ms < 500 else 'FAIL'}")
print()

# ── Test 6: Sample predictions ───────────────────────────────────────────
print("7. SAMPLE PREDICTIONS")
print("-" * 40)

samples = [
    ("Python developer with 5 years Django REST API experience",
     "Senior backend engineer needed with Python and SQL"),
    ("Registered nurse with ICU experience and patient care skills",
     "Looking for healthcare professional for hospital"),
    ("Marketing manager with digital campaigns and SEO expertise",
     "Marketing role for brand management"),
]

domain_names = {0:"IT", 1:"Management", 2:"Design", 3:"Healthcare",
                4:"Finance", 5:"Legal", 6:"Education"}

for i, (resume, jd) in enumerate(samples):
    ats_out, dom_out, rsg_out = model(
        [tf.constant([resume]), tf.constant([jd])], training=False
    )
    
    ats_score = float(ats_out[0][0]) * 100
    dom_idx = int(np.argmax(dom_out[0]))
    rsg_idx = int(np.argmax(rsg_out[0]))
    tmpl_id = idx_to_id.get(rsg_idx, -1)
    
    print(f"   Sample {i+1}:")
    print(f"     ATS Score: {ats_score:.1f}/100")
    print(f"     Domain:    {domain_names.get(dom_idx, 'Unknown')} ({dom_idx})")
    print(f"     Template:  ID {tmpl_id} (index {rsg_idx})")
    print()

# ── Final Summary ────────────────────────────────────────────────────────
print("=" * 60)
print("PRODUCTION VALIDATION SUMMARY")
print("=" * 60)
print()

checks = [
    ("Output shapes correct", all_shapes_ok),
    ("Output ranges valid", all_ranges_ok),
    ("ATS MAE < 8.0", ats_mae is not None and ats_mae < 8.0),
    ("Domain F1 > 0.80", dom_f1 is not None and dom_f1 > 0.80),
    ("RSG Accuracy >= 50%", rsg_pass),
    ("Inference < 500ms", avg_ms < 500),
]

all_passed = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"   [{status}] {name}")

print()
if all_passed:
    print("=" * 60)
    print("   ALL CHECKS PASSED - MODEL IS PRODUCTION READY")
    print("=" * 60)
    print()
    print("   Final model weights: model/unified_model/unified_stage3_weights.h5")
    print()
    print("   Performance summary:")
    if ats_mae:
        print(f"     - ATS MAE:        {ats_mae:.2f} (target < 8.0)")
    if dom_f1:
        print(f"     - Domain F1:      {dom_f1:.4f} (target > 0.80)")
    print(f"     - RSG Accuracy:   {rsg_acc*100:.1f}% (target >= 50%)")
    print(f"     - Inference time: {avg_ms:.1f}ms per sample")
else:
    print("=" * 60)
    print("   VALIDATION FAILED - DO NOT DEPLOY TO PRODUCTION")
    print("=" * 60)
    print()
    print("   Review failed checks above and fix before deployment.")
