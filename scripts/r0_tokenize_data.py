"""
INJECTION-R0: Tokenization Pipeline

Pre-tokenizes ATS and RSG training data into .npz files for R1-R4 training.
Run from the project root: python scripts/r0_tokenize_data.py
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "ats-ai-core"))

from transformers import AutoTokenizer

# ── Paths ────────────────────────────────────────────────────────────────────

ATS_CSV         = ROOT / "data" / "labeled" / "merged_final.csv"
RSG_CSV_PRIMARY = ROOT / "data" / "labeled" / "rsg_data.csv"
RSG_CSV_FALLBACK = ROOT / "data" / "labeled" / "rsg_balanced.csv"
SPLITS_JSON     = ROOT / "model" / "unified_model" / "data_splits.json"
RSG_MAPPING_JSON = ROOT / "model" / "unified_model" / "rsg_label_mapping.json"

OUT_DIR         = ROOT / "data" / "tokenized"
EVAL_DIR        = ROOT / "evaluation"
ATS_NPZ         = OUT_DIR / "ats_tokenized.npz"
RSG_NPZ         = OUT_DIR / "rsg_tokenized.npz"
REPORT_JSON     = EVAL_DIR / "r0_tokenization_report.json"

MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN         = 128

OUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def seq_len_stats(attention_masks: np.ndarray, name: str) -> dict:
    lengths = attention_masks.sum(axis=1)
    stats = {
        "mean":  float(np.mean(lengths)),
        "p50":   float(np.percentile(lengths, 50)),
        "p95":   float(np.percentile(lengths, 95)),
        "p99":   float(np.percentile(lengths, 99)),
        "max":   float(np.max(lengths)),
        "truncated_pct": float(100.0 * (lengths == MAX_LEN).mean()),
    }
    print(f"  {name}: mean={stats['mean']:.1f}  p50={stats['p50']:.0f}"
          f"  p95={stats['p95']:.0f}  p99={stats['p99']:.0f}"
          f"  truncated={stats['truncated_pct']:.1f}%")
    return stats


def tokenize_batch(tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids  = enc["input_ids"].astype(np.int32)
    mask = enc["attention_mask"].astype(np.int32)
    return ids, mask


# ── Load tokenizer ────────────────────────────────────────────────────────────

print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"  Vocab size: {tokenizer.vocab_size}")

report: dict = {"model": MODEL_NAME, "max_length": MAX_LEN}


# ── Stage 1: ATS dataset ──────────────────────────────────────────────────────

print("\n=== Stage 1: ATS Tokenization ===")
import pandas as pd

df_ats = pd.read_csv(ATS_CSV).dropna()
print(f"  Loaded {len(df_ats):,} rows from {ATS_CSV.name}")

score_col  = "score" if "score" in df_ats.columns else "ats_score"
domain_col = "domain_index" if "domain_index" in df_ats.columns else "domain_label"

resume_texts = df_ats["resume_text"].astype(str).tolist()
jd_texts     = df_ats["jd_text"].astype(str).tolist()
raw_scores   = df_ats[score_col].astype(float).values
domain_arr   = df_ats[domain_col].astype(int).values

# Normalise 0-100 → 0-1
ats_scores = (raw_scores / 100.0).astype(np.float32)
assert ats_scores.min() >= 0.0 and ats_scores.max() <= 1.0, (
    f"ATS scores out of [0,1]: min={ats_scores.min():.4f} max={ats_scores.max():.4f}"
)
assert domain_arr.min() >= 0 and domain_arr.max() <= 6, (
    f"Domain labels out of [0,6]: {domain_arr.min()} … {domain_arr.max()}"
)

print("  Tokenizing resume texts …")
resume_ids, resume_mask = tokenize_batch(tokenizer, resume_texts)

print("  Tokenizing JD texts …")
jd_ids, jd_mask = tokenize_batch(tokenizer, jd_texts)

print("  Sequence length stats:")
ats_stats = {
    "resume": seq_len_stats(resume_mask, "resume"),
    "jd":     seq_len_stats(jd_mask, "jd"),
}

np.savez_compressed(
    ATS_NPZ,
    resume_input_ids=resume_ids,
    resume_attention_mask=resume_mask,
    jd_input_ids=jd_ids,
    jd_attention_mask=jd_mask,
    ats_scores=ats_scores,
    domain_labels=domain_arr.astype(np.int32),
)
print(f"  Saved → {ATS_NPZ}  (N={len(resume_ids):,})")

report["ats"] = {
    "n_total": int(len(resume_ids)),
    "score_min": float(ats_scores.min()),
    "score_max": float(ats_scores.max()),
    "domain_counts": {str(d): int((domain_arr == d).sum()) for d in range(7)},
    "seq_stats": ats_stats,
}


# ── Stage 2: RSG dataset ──────────────────────────────────────────────────────

print("\n=== Stage 2: RSG Tokenization ===")

rsg_csv = RSG_CSV_PRIMARY if RSG_CSV_PRIMARY.exists() else RSG_CSV_FALLBACK
if rsg_csv == RSG_CSV_FALLBACK:
    print(f"  WARNING: {RSG_CSV_PRIMARY.name} not found, using {RSG_CSV_FALLBACK.name}")
else:
    print(f"  Using {rsg_csv.name}")

df_rsg = pd.read_csv(rsg_csv).dropna()
print(f"  Loaded {len(df_rsg):,} rows")

profile_texts_all = df_rsg["profile_text"].astype(str).values
raw_labels        = df_rsg["template_index"].astype(int).values

# Load label mapping
with open(RSG_MAPPING_JSON) as f:
    mapping = json.load(f)
id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}

# Determine if template_index values are original IDs (need mapping) or already 0-45
needs_mapping = any(t not in range(46) for t in raw_labels[:min(1000, len(raw_labels))])

if needs_mapping:
    valid_mask = np.array([int(t) in id_to_idx for t in raw_labels])
    print(f"  Mapping original IDs: {valid_mask.sum():,} / {len(raw_labels):,} valid")
    assert valid_mask.sum() >= 1000, f"Too few valid RSG samples: {valid_mask.sum()}"
    profile_texts = profile_texts_all[valid_mask].tolist()
    rsg_labels    = np.array([id_to_idx[int(t)] for t in raw_labels[valid_mask]], dtype=np.int32)
else:
    # Already 0-45 (rsg_balanced.csv with pre-indexed labels)
    valid_mask = (raw_labels >= 0) & (raw_labels <= 45)
    print(f"  Direct indices: {valid_mask.sum():,} / {len(raw_labels):,} in [0,45]")
    assert valid_mask.sum() >= 1000, f"Too few valid RSG samples: {valid_mask.sum()}"
    profile_texts = profile_texts_all[valid_mask].tolist()
    rsg_labels    = raw_labels[valid_mask].astype(np.int32)

assert rsg_labels.min() >= 0 and rsg_labels.max() <= 45, (
    f"RSG labels out of [0,45]: {rsg_labels.min()} … {rsg_labels.max()}"
)

print("  Tokenizing profile texts …")
profile_ids, profile_mask = tokenize_batch(tokenizer, profile_texts)

print("  Sequence length stats:")
rsg_stats = {"profile": seq_len_stats(profile_mask, "profile")}

np.savez_compressed(
    RSG_NPZ,
    profile_input_ids=profile_ids,
    profile_attention_mask=profile_mask,
    rsg_labels=rsg_labels,
)
print(f"  Saved → {RSG_NPZ}  (M={len(profile_ids):,})")

report["rsg"] = {
    "n_total": int(len(profile_ids)),
    "source_file": rsg_csv.name,
    "label_range": [int(rsg_labels.min()), int(rsg_labels.max())],
    "label_counts": {str(l): int((rsg_labels == l).sum()) for l in range(46)},
    "seq_stats": rsg_stats,
}


# ── Stage 3: Verify canonical splits ─────────────────────────────────────────

print("\n=== Stage 3: Split Verification ===")
with open(SPLITS_JSON) as f:
    splits = json.load(f)

n_ats   = len(resume_ids)
n_rsg   = len(profile_ids)
checks  = {}

for split_name, expected_range, label in [
    ("ats_train", n_ats, "ATS train"),
    ("ats_val",   n_ats, "ATS val"),
    ("ats_test",  n_ats, "ATS test"),
    ("rsg_train", n_rsg, "RSG train"),
    ("rsg_val",   n_rsg, "RSG val"),
]:
    idx = np.array(splits[split_name])
    oob = (idx >= expected_range).sum()
    checks[split_name] = {
        "n": int(len(idx)),
        "max_idx": int(idx.max()) if len(idx) else -1,
        "dataset_size": int(expected_range),
        "oob_count": int(oob),
        "ok": bool(oob == 0),
    }
    status = "OK" if oob == 0 else f"OOB={oob}"
    print(f"  {label:12s}: n={len(idx):6,}  max_idx={idx.max() if len(idx) else -1:6,}"
          f"  dataset={expected_range:6,}  [{status}]")

ats_total = sum(checks[k]["n"] for k in ("ats_train", "ats_val", "ats_test"))
rsg_total = sum(checks[k]["n"] for k in ("rsg_train", "rsg_val"))
print(f"\n  ATS total covered: {ats_total:,} / {n_ats:,}")
print(f"  RSG total covered: {rsg_total:,} / {n_rsg:,}")

ats_train_pct = 100.0 * checks["ats_train"]["n"] / n_ats
ats_val_pct   = 100.0 * checks["ats_val"]["n"]   / n_ats
ats_test_pct  = 100.0 * checks["ats_test"]["n"]  / n_ats
rsg_train_pct = 100.0 * checks["rsg_train"]["n"] / n_rsg
rsg_val_pct   = 100.0 * checks["rsg_val"]["n"]   / n_rsg
print(f"  ATS  split: {ats_train_pct:.1f}% / {ats_val_pct:.1f}% / {ats_test_pct:.1f}%"
      f"  (target 75/15/10)")
print(f"  RSG  split: {rsg_train_pct:.1f}% / {rsg_val_pct:.1f}%  (target 80/20)")

all_oob_ok = all(c["ok"] for c in checks.values())
if not all_oob_ok:
    print("  WARNING: Some split indices are out-of-bounds for the current dataset size.")
    print("           Run data_prep_r0.py to regenerate splits.")

report["splits"] = {
    "splits_json": str(SPLITS_JSON),
    "checks": checks,
    "all_indices_valid": all_oob_ok,
}


# ── Save report ───────────────────────────────────────────────────────────────

with open(REPORT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved → {REPORT_JSON}")


# ── Final gate ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
assert not np.any(np.isnan(ats_scores)),   "NaN detected in ats_scores"
assert len(resume_ids) == len(jd_ids) == len(ats_scores) == len(domain_arr)
assert len(profile_ids) == len(rsg_labels)

n_ats_final = len(resume_ids)
n_rsg_final = len(profile_ids)
print(f"ATS npz  — N={n_ats_final:,}  shapes: [{n_ats_final},{MAX_LEN}]")
print(f"RSG npz  — M={n_rsg_final:,}  shapes: [{n_rsg_final},{MAX_LEN}]")
print("=" * 60)
print("R0 PASSED — proceed to R1")
