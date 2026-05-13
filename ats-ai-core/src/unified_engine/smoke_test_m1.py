"""
M-1 Smoke Test — Data pipeline → model inference (1 batch, 32 pairs)
No training. No modifications to data loader or model.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TFHUB_CACHE_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tfhub_cache"
)

import numpy as np
import pandas as pd
import tensorflow as tf

# ── Build model ────────────────────────────────────────────────────────
from unified_model import build_unified_model

print("Building model...")
model = build_unified_model()
print("Model built.\n")

# ── Load exactly 1 batch (32 pairs) from the ATS CSV ──────────────────
ATS_CSV = r"C:\Users\saini\Desktop\ats\ats-ai-core\data\labeled\merged_final.csv"
BATCH = 32

print(f"Loading {BATCH} rows from: {ATS_CSV}")
df = pd.read_csv(ATS_CSV).dropna().head(BATCH)
print(f"Loaded {len(df)} rows.\n")

resume_texts = df["resume_text"].astype(str).values
jd_texts     = df["jd_text"].astype(str).values

# ── Run inference (no training) ────────────────────────────────────────
print("Running model([resume_texts, jd_texts], training=False)...")
resume_tf = tf.constant(resume_texts)
jd_tf     = tf.constant(jd_texts)

ats_out, dom_out, rsg_out = model([resume_tf, jd_tf], training=False)

print()
print("=" * 60)
print("M-1 SMOKE TEST — OUTPUT SHAPES")
print("=" * 60)
print(f"ATS output shape:    {ats_out.shape}")    # expected (32, 1)
print(f"Domain output shape: {dom_out.shape}")     # expected (32, 7)
print(f"RSG output shape:    {rsg_out.shape}")     # expected (32, 46)
print("=" * 60)

# ── Sanity checks ──────────────────────────────────────────────────────
assert ats_out.shape == (BATCH, 1),  f"ATS shape mismatch: {ats_out.shape}"
assert dom_out.shape == (BATCH, 7),  f"Domain shape mismatch: {dom_out.shape}"
assert rsg_out.shape == (BATCH, 46), f"RSG shape mismatch: {rsg_out.shape}"

# Value range checks
print(f"\nATS score range:    [{float(tf.reduce_min(ats_out)):.4f}, {float(tf.reduce_max(ats_out)):.4f}]  (expected 0-1)")
print(f"Domain softmax sum: {float(tf.reduce_mean(tf.reduce_sum(dom_out, axis=1))):.4f}  (expected ~1.0)")
print(f"RSG softmax sum:    {float(tf.reduce_mean(tf.reduce_sum(rsg_out, axis=1))):.4f}  (expected ~1.0)")

print("\nSMOKE TEST PASSED — M-1 complete")
