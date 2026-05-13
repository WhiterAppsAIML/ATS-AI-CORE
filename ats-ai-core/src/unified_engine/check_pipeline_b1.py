"""
B-1: Data Pipeline Smoke Test — USE Lite v2
Pass 1 batch of 32 pairs through the model in inference mode.
Verify output shapes for all three heads.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from src.unified_engine.unified_model import build_unified_model

# ── Locate data ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
ATS_CSV = PROJECT_ROOT / "data" / "labeled" / "merged_final.csv"

if not ATS_CSV.exists():
    print(f"FAIL: Data file not found at {ATS_CSV}")
    sys.exit(1)

print("=" * 65)
print("  B-1: DATA PIPELINE SMOKE TEST — USE Lite v2")
print("=" * 65)

# ── Step 1: Load 1 batch of 32 pairs ──────────────────────────────────
print("\n[1/4] Loading data...")
df = pd.read_csv(str(ATS_CSV)).dropna()
print(f"  Total rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# Sample exactly 32 rows
batch_df = df.head(32)
resume_texts  = batch_df["resume_text"].astype(str).values
jd_texts      = batch_df["jd_text"].astype(str).values
ats_scores    = (batch_df["score"].astype(float) / 100.0).values.astype("float32")
domain_labels = batch_df["domain_index"].astype(int).values.astype("int32")

print(f"  Batch size: {len(resume_texts)}")
print(f"  Sample resume text[:80]: {resume_texts[0][:80]}...")
print(f"  Sample JD text[:80]:     {jd_texts[0][:80]}...")

# ── Step 2: Build model ───────────────────────────────────────────────
print("\n[2/4] Building model...")
model = build_unified_model()
print("  Model built successfully.")

# Confirm encoder layer
enc = model.get_layer("mobile_use_encoder")
print(f"  Encoder layer: {enc.name}")
print(f"  Encoder trainable: {enc.trainable}")

# ── Step 3: Forward pass — inference mode ─────────────────────────────
print("\n[3/4] Running inference on batch of 32...")
inputs = {
    "resume_text": tf.constant(resume_texts),
    "jd_text":     tf.constant(jd_texts),
}

# model() returns a list of 3 outputs: [ats_score, domain_probs, rsg_template]
outputs = model(inputs, training=False)
ats_out, dom_out, rsg_out = outputs

print(f"  ATS output shape:    {ats_out.shape}   — Expected: (32, 1)")
print(f"  Domain output shape: {dom_out.shape}   — Expected: (32, 7)")
print(f"  RSG output shape:    {rsg_out.shape}  — Expected: (32, 46)")

# ── Step 4: Validate shapes ───────────────────────────────────────────
print("\n[4/4] Validating shapes...")
gates = []

if ats_out.shape == (32, 1):
    print("  [PASS] ATS output shape: (32, 1)")
    gates.append(True)
else:
    print(f"  [FAIL] ATS output shape: {ats_out.shape} != (32, 1)")
    gates.append(False)

if dom_out.shape == (32, 7):
    print("  [PASS] Domain output shape: (32, 7)")
    gates.append(True)
else:
    print(f"  [FAIL] Domain output shape: {dom_out.shape} != (32, 7)")
    gates.append(False)

if rsg_out.shape == (32, 46):
    print("  [PASS] RSG output shape: (32, 46)")
    gates.append(True)
else:
    print(f"  [FAIL] RSG output shape: {rsg_out.shape} != (32, 46)")
    gates.append(False)

# Value sanity checks
ats_vals = ats_out.numpy()
dom_vals = dom_out.numpy()
rsg_vals = rsg_out.numpy()

print(f"\n  ATS score range: [{ats_vals.min():.4f}, {ats_vals.max():.4f}]")
print(f"  Domain probs sum (row 0): {dom_vals[0].sum():.4f} — Expected: ~1.0")
print(f"  RSG probs sum (row 0):    {rsg_vals[0].sum():.4f} — Expected: ~1.0")

if abs(dom_vals[0].sum() - 1.0) < 0.01:
    print("  [PASS] Domain softmax sums to ~1.0")
    gates.append(True)
else:
    print(f"  [FAIL] Domain softmax sum: {dom_vals[0].sum():.4f}")
    gates.append(False)

if abs(rsg_vals[0].sum() - 1.0) < 0.01:
    print("  [PASS] RSG softmax sums to ~1.0")
    gates.append(True)
else:
    print(f"  [FAIL] RSG softmax sum: {rsg_vals[0].sum():.4f}")
    gates.append(False)

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if all(gates):
    print("  B-1 PIPELINE SMOKE TEST: ALL GATES PASSED")
else:
    print(f"  B-1 PIPELINE SMOKE TEST: {gates.count(False)} GATE(S) FAILED")
print("=" * 65)
