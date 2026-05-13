import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # must be before any TF import

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.unified_engine.unified_model import build_unified_model
from src.config import RSG_KERAS_PATH, RSG_MAPPING_JSON, UNIFIED_MODEL_DIR

# ── Paths — confirmed from your system ─────────────────────────────
RSG_KERAS    = RSG_KERAS_PATH
MAPPING_JSON = RSG_MAPPING_JSON
SAVE_PATH    = UNIFIED_MODEL_DIR / "unified_with_rsg_weights.h5"
# ────────────────────────────────────────────────────────────────────

print("=== INJECTION-1B: RSG WEIGHT TRANSFER ===\n")

# ── Step 1: Load label mapping ──────────────────────────────────────
print("Loading label mapping...")
with open(MAPPING_JSON) as f:
    mapping = json.load(f)
idx_to_id = {int(k): int(v) for k, v in mapping["idx_to_id"].items()}
print(f"  Classes: {len(idx_to_id)}")
print(f"  Sample : idx 0 -> tmpl {idx_to_id[0]}, "
      f"idx 10 -> tmpl {idx_to_id[10]}, "
      f"idx 45 -> tmpl {idx_to_id[45]}")

# ── Step 2: Build the unified model ────────────────────────────────
print("\nBuilding unified model skeleton...")
unified = build_unified_model()
print("  Built.")

# ── Step 3: Load the RSG source model ──────────────────────────────
print(f"\nLoading RSG model from:")
print(f"  {RSG_KERAS}")
if not RSG_KERAS.exists():
    print(f"\nERROR: File not found — check the path above.")
    print("HARD STOP: Fix the path and re-run.")
    exit(1)
rsg_source = tf.keras.models.load_model(str(RSG_KERAS), compile=False)
print("  RSG model loaded.")

# ── Step 4: Identify layers with weights ───────────────────────────
# RSG source layers (from retrained summary_model.keras):
#   dense, batch_normalization, dense_1, batch_normalization_1,
#   dense_2, dense_3   (Dropout layers have no weights — skipped)
# Unified RSG head layers (prefixed rsg_):
#   rsg_dense1, rsg_bn1, rsg_dense2, rsg_bn2, rsg_dense3, rsg_template

src_layers = [l for l in rsg_source.layers if l.get_weights()]
dst_layers = [l for l in unified.layers
              if l.name.startswith("rsg_") and l.get_weights()]

print(f"\nSource RSG layers  ({len(src_layers)}): {[l.name for l in src_layers]}")
print(f"Unified RSG layers ({len(dst_layers)}): {[l.name for l in dst_layers]}")

if len(src_layers) != len(dst_layers):
    print(f"\nMISMATCH: {len(src_layers)} source layers vs "
          f"{len(dst_layers)} unified layers.")
    print("HARD STOP — do not continue. Report to Sai.")
    exit(1)

# ── Step 5: Transfer weights by shape matching ─────────────────────
print("\nTransferring weights:")
transferred = 0
failed = []

for src_l, dst_l in zip(src_layers, dst_layers):
    src_weights = src_l.get_weights()
    dst_weights = dst_l.get_weights()
    src_shapes = [w.shape for w in src_weights]
    dst_shapes = [w.shape for w in dst_weights]

    if src_shapes == dst_shapes:
        dst_l.set_weights(src_weights)
        transferred += 1
        print(f"  OK   {src_l.name:<28} -> {dst_l.name}   shapes={src_shapes}")
    else:
        failed.append(
            f"  FAIL {src_l.name} {src_shapes} -> {dst_l.name} {dst_shapes}")

if failed:
    print("\nSHAPE MISMATCHES FOUND:")
    for line in failed:
        print(line)
    print("\nHARD STOP — report the mismatches above to Sai.")
    exit(1)

print(f"\nAll {transferred}/{len(src_layers)} RSG layers transferred — SUCCESS")

# ── Step 6: Save unified model weights ─────────────────────────────
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
unified.save_weights(str(SAVE_PATH))
print(f"\nSaved: {SAVE_PATH}")

# ── Step 7: Sanity check — run one inference ────────────────────────
print("\n=== SANITY CHECK ===")
test_resume = tf.constant(
    ["Python developer with 5 years of Django REST APIs SQL experience"])
test_jd     = tf.constant(
    ["Looking for senior backend software engineer with Python Django"])

ats_out, dom_out, rsg_out = unified(
    [test_resume, test_jd], training=False)

ats_score = float(ats_out[0][0]) * 100
dom_idx   = int(np.argmax(dom_out[0]))
rsg_idx   = int(np.argmax(rsg_out[0]))
tmpl_id   = idx_to_id[rsg_idx]

domain_names = {0:"IT", 1:"Management", 2:"Design", 3:"Healthcare",
                4:"Finance", 5:"Legal", 6:"Education"}

print(f"  ATS score   : {ats_score:.1f} / 100")
print(f"  Domain      : {dom_idx} ({domain_names.get(dom_idx, 'Unknown')})")
print(f"  RSG index   : {rsg_idx} -> template ID {tmpl_id}")
print()

# Assertions
assert 0 <= ats_score <= 100, f"ATS score out of range: {ats_score}"
assert 0 <= dom_idx   <= 6,   f"Domain out of range: {dom_idx}"
assert 0 <= rsg_idx   <= 45,  f"RSG index out of range: {rsg_idx}"
assert tmpl_id in idx_to_id.values(), f"Template ID {tmpl_id} not in mapping"

# For the test resume (Python/IT), domain should be 0 (IT)
if dom_idx == 0:
    print("  Domain prediction: CORRECT (IT for Python developer)")
else:
    print(f"  Domain prediction: {dom_idx} — note this for Sai "
          "(ATS weights not loaded yet, will correct in Stage 1 training)")

print()
print("=== INJECTION-1B: COMPLETE ===")
print()
print("Files ready for training:")
print(f"  {SAVE_PATH}          <- unified model with RSG weights pre-loaded")
print(f"  model\\unified_model\\rsg_label_mapping.json  <- template ID lookup")
print()
print("Next step: INJECTION-2-STAGE1 (RSG warmup training)")
print("Show this full output to Sai before starting training.")
