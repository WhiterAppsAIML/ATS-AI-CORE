"""scripts/r0_tokenize_data.py — INJECTION-R0 Tokenization Pipeline

Pre-tokenizes all training data for MiniLM (all-MiniLM-L6-v2) and saves
to .npz files so that R1–R4 training scripts load tokens directly.

Stage 1 — ATS tokenization
  Loads data/labeled/merged_final.csv, drops NaN rows
  Normalizes score (0-100) → ats_scores (0-1 float32)
  Tokenizes resume_text and jd_text independently: max_length=128,
      padding='max_length', truncation=True
  Asserts scores in [0,1] and domain labels in [0,6]
  Saves data/tokenized/ats_tokenized.npz with 6 arrays

Stage 2 — RSG tokenization
  Tries rsg_data.csv first, falls back to rsg_balanced.csv with a warning
  Auto-detects whether template_index values are original IDs
      (need id_to_idx mapping) or are already 0-45
  Filters to valid samples only, maps to [0,45] label space
  Saves data/tokenized/rsg_tokenized.npz with 3 arrays

Stage 3 — Split verification
  Loads model/unified_model/data_splits.json and checks every index
      is in-bounds for the current dataset sizes
  Prints split % ratios vs 75/15/10 and 80/20 targets
  Warns (but doesn't hard-fail) if splits are OOB, so you can regenerate
      via data_prep_r0.py if needed

Usage:
    python scripts/r0_tokenize_data.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.config import (
    DATA_DIR,
    EVALUATION_DIR,
    LABELED_DIR,
    MINILM_MAX_SEQ_LEN,
    MINILM_MODEL_NAME,
    NUM_DOMAINS,
    RSG_BALANCED_CSV_PATH,
    RSG_CSV_PATH,
    RSG_MAPPING_JSON,
    RSG_NUM_CLASSES,
    UNIFIED_MODEL_DIR,
)

# ── Constants ──────────────────────────────────────────────────────────────────
SEQ_LEN       = MINILM_MAX_SEQ_LEN   # 128
TOKENIZED_DIR = DATA_DIR / "tokenized"
ATS_NPZ       = TOKENIZED_DIR / "ats_tokenized.npz"
RSG_NPZ       = TOKENIZED_DIR / "rsg_tokenized.npz"
REPORT_PATH   = EVALUATION_DIR / "r0_tokenization_report.json"
SPLITS_JSON   = PROJECT_ROOT / "model" / "unified_model" / "data_splits.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Helper: batch tokenize ────────────────────────────────────────────────────

def batch_tokenize(tokenizer, texts: list[str], desc: str = "texts") -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a list of texts and return (input_ids, attention_mask) as int32 arrays."""
    log.info("Tokenizing %d %s (seq_len=%d)...", len(texts), desc, SEQ_LEN)
    t0 = time.time()
    enc = tokenizer(
        texts,
        max_length=SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids  = enc["input_ids"].astype(np.int32)
    mask = enc["attention_mask"].astype(np.int32)
    elapsed = time.time() - t0
    log.info("  -> shape %s, %.1f sec (%.0f texts/sec)", ids.shape, elapsed, len(texts) / max(elapsed, 0.01))
    return ids, mask


def seq_length_stats(attention_mask: np.ndarray) -> dict:
    """Compute sequence length distribution from attention masks."""
    lengths = attention_mask.sum(axis=1)
    return {
        "mean":  round(float(np.mean(lengths)), 1),
        "p50":   int(np.percentile(lengths, 50)),
        "p95":   int(np.percentile(lengths, 95)),
        "p99":   int(np.percentile(lengths, 99)),
        "min":   int(np.min(lengths)),
        "max":   int(np.max(lengths)),
    }


# ── Stage 1: ATS tokenization ────────────────────────────────────────────────

def stage1_ats(tokenizer) -> dict:
    """Tokenize ATS dataset from merged_final.csv."""
    csv_path = LABELED_DIR / "merged_final.csv"
    log.info("Loading ATS data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Drop NaN rows in critical columns
    before = len(df)
    df = df.dropna(subset=["resume_text", "jd_text", "score", "domain_index"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        log.warning("Dropped %d NaN rows (%.2f%%)", dropped, 100 * dropped / before)

    # Normalize score: 0-100 -> 0-1
    raw_scores = df["score"].values.astype(np.float32)
    ats_scores = np.clip(raw_scores / 100.0, 0.0, 1.0).astype(np.float32)

    # Domain labels
    domain_labels = df["domain_index"].values.astype(np.int32)

    # Assertions
    assert ats_scores.min() >= 0.0, f"ATS score min {ats_scores.min()} < 0"
    assert ats_scores.max() <= 1.0, f"ATS score max {ats_scores.max()} > 1"
    assert domain_labels.min() >= 0, f"Domain label min {domain_labels.min()} < 0"
    assert domain_labels.max() <= NUM_DOMAINS - 1, (
        f"Domain label max {domain_labels.max()} > {NUM_DOMAINS - 1}"
    )

    # Tokenize
    resume_ids, resume_mask = batch_tokenize(
        tokenizer, df["resume_text"].tolist(), "resume texts"
    )
    jd_ids, jd_mask = batch_tokenize(
        tokenizer, df["jd_text"].tolist(), "JD texts"
    )

    # Sequence length stats
    resume_stats = seq_length_stats(resume_mask)
    jd_stats     = seq_length_stats(jd_mask)

    log.info("Resume seq-len stats: %s", resume_stats)
    log.info("JD seq-len stats:     %s", jd_stats)

    # Save .npz
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ATS_NPZ,
        resume_input_ids=resume_ids,
        resume_attention_mask=resume_mask,
        jd_input_ids=jd_ids,
        jd_attention_mask=jd_mask,
        ats_scores=ats_scores,
        domain_labels=domain_labels,
    )
    size_mb = ATS_NPZ.stat().st_size / (1024 * 1024)
    log.info("Saved %s (%.2f MB)", ATS_NPZ, size_mb)

    # Domain distribution
    unique, counts = np.unique(domain_labels, return_counts=True)
    domain_dist = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "csv_path":          str(csv_path),
        "total_rows":        before,
        "rows_after_dropna": len(df),
        "dropped_nan":       dropped,
        "score_range_raw":   [float(raw_scores.min()), float(raw_scores.max())],
        "score_range_norm":  [float(ats_scores.min()), float(ats_scores.max())],
        "domain_range":      [int(domain_labels.min()), int(domain_labels.max())],
        "domain_distribution": domain_dist,
        "resume_seq_stats":  resume_stats,
        "jd_seq_stats":      jd_stats,
        "npz_path":          str(ATS_NPZ),
        "npz_size_mb":       round(size_mb, 2),
        "arrays": {
            "resume_input_ids":     list(resume_ids.shape),
            "resume_attention_mask": list(resume_mask.shape),
            "jd_input_ids":         list(jd_ids.shape),
            "jd_attention_mask":    list(jd_mask.shape),
            "ats_scores":           list(ats_scores.shape),
            "domain_labels":        list(domain_labels.shape),
        },
    }


# ── Stage 2: RSG tokenization ────────────────────────────────────────────────

def stage2_rsg(tokenizer) -> dict:
    """Tokenize RSG dataset, auto-mapping template IDs to 0-45 label space."""

    # Try rsg_data.csv first, then rsg_balanced.csv
    if RSG_CSV_PATH.exists():
        csv_path = RSG_CSV_PATH
        log.info("Loading RSG data from %s", csv_path)
    elif RSG_BALANCED_CSV_PATH.exists():
        csv_path = RSG_BALANCED_CSV_PATH
        log.warning("rsg_data.csv not found, falling back to %s", RSG_BALANCED_CSV_PATH)
    else:
        raise FileNotFoundError(
            f"Neither {RSG_CSV_PATH} nor {RSG_BALANCED_CSV_PATH} found"
        )

    df = pd.read_csv(csv_path)

    # Drop NaN
    before = len(df)
    df = df.dropna(subset=["profile_text", "template_index"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        log.warning("Dropped %d NaN rows", dropped)

    raw_labels = df["template_index"].values.astype(np.int64)

    # Auto-detect: are labels already 0-45, or do they need id_to_idx mapping?
    max_label = int(raw_labels.max())
    needs_mapping = max_label > (RSG_NUM_CLASSES - 1)  # > 45

    if needs_mapping:
        log.info("template_index range [%d, %d] exceeds [0, %d] -- applying id_to_idx mapping",
                 raw_labels.min(), max_label, RSG_NUM_CLASSES - 1)

        if RSG_MAPPING_JSON.exists():
            with open(RSG_MAPPING_JSON, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}
        else:
            # Build mapping from sorted unique IDs
            log.warning("rsg_label_mapping.json not found -- building from data")
            unique_ids = sorted(set(raw_labels))
            id_to_idx = {int(uid): idx for idx, uid in enumerate(unique_ids)}

        # Map labels, filter unmappable
        mapped = []
        valid_mask = []
        for lbl in raw_labels:
            idx = id_to_idx.get(int(lbl), -1)
            mapped.append(idx)
            valid_mask.append(idx >= 0 and idx < RSG_NUM_CLASSES)

        valid_mask = np.array(valid_mask)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            log.warning("Filtering %d samples with unmappable template_index values", invalid_count)

        df = df[valid_mask].reset_index(drop=True)
        rsg_labels = np.array([m for m, v in zip(mapped, valid_mask) if v], dtype=np.int32)
    else:
        log.info("template_index range [%d, %d] already in [0, %d] -- no mapping needed",
                 raw_labels.min(), max_label, RSG_NUM_CLASSES - 1)
        # Still filter out any out-of-range
        valid_mask = (raw_labels >= 0) & (raw_labels < RSG_NUM_CLASSES)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            log.warning("Filtering %d out-of-range samples", invalid_count)
            df = df[valid_mask].reset_index(drop=True)
        rsg_labels = df["template_index"].values.astype(np.int32)

    # Assertions
    assert rsg_labels.min() >= 0, f"RSG label min {rsg_labels.min()} < 0"
    assert rsg_labels.max() <= RSG_NUM_CLASSES - 1, (
        f"RSG label max {rsg_labels.max()} > {RSG_NUM_CLASSES - 1}"
    )

    # Tokenize
    profile_ids, profile_mask = batch_tokenize(
        tokenizer, df["profile_text"].tolist(), "profile texts"
    )

    # Sequence length stats
    profile_stats = seq_length_stats(profile_mask)
    log.info("Profile seq-len stats: %s", profile_stats)

    # Save .npz
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RSG_NPZ,
        profile_input_ids=profile_ids,
        profile_attention_mask=profile_mask,
        rsg_labels=rsg_labels,
    )
    size_mb = RSG_NPZ.stat().st_size / (1024 * 1024)
    log.info("Saved %s (%.2f MB)", RSG_NPZ, size_mb)

    # Label distribution
    unique, counts = np.unique(rsg_labels, return_counts=True)
    label_dist = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "csv_path":          str(csv_path),
        "total_rows":        before,
        "rows_after_filter": len(df),
        "dropped_or_filtered": before - len(df),
        "needs_mapping":     needs_mapping,
        "label_range":       [int(rsg_labels.min()), int(rsg_labels.max())],
        "num_unique_labels": len(unique),
        "label_distribution": label_dist,
        "profile_seq_stats": profile_stats,
        "npz_path":          str(RSG_NPZ),
        "npz_size_mb":       round(size_mb, 2),
        "arrays": {
            "profile_input_ids":     list(profile_ids.shape),
            "profile_attention_mask": list(profile_mask.shape),
            "rsg_labels":            list(rsg_labels.shape),
        },
    }


# ── Stage 3: Split verification ──────────────────────────────────────────────

def stage3_splits(ats_n: int, rsg_n: int) -> dict:
    """Verify data_splits.json indices are in-bounds for current datasets."""
    if not SPLITS_JSON.exists():
        log.warning("data_splits.json not found at %s -- skipping split verification", SPLITS_JSON)
        return {
            "splits_found": False,
            "status": "SKIP_NO_SPLITS_FILE",
            "note": f"Expected at {SPLITS_JSON}. Run data_prep_r0.py to generate.",
        }

    with open(SPLITS_JSON, "r", encoding="utf-8") as f:
        splits = json.load(f)

    result: dict = {"splits_found": True}
    warnings: list[str] = []

    # data_splits.json uses flat keys: ats_train, ats_val, ats_test, rsg_train, rsg_val
    for split_key, dataset_n, label, expected_pct in [
        ("ats_train", ats_n, "ATS train", 0.75),
        ("ats_val",   ats_n, "ATS val",   0.15),
        ("ats_test",  ats_n, "ATS test",  0.10),
        ("rsg_train", rsg_n, "RSG train", 0.80),
        ("rsg_val",   rsg_n, "RSG val",   0.20),
    ]:
        indices = splits.get(split_key, [])
        n = len(indices)
        pct = n / dataset_n if dataset_n > 0 else 0

        oob = [i for i in indices if i < 0 or i >= dataset_n]
        if oob:
            msg = f"{split_key}: {len(oob)} indices OOB (dataset size={dataset_n})"
            log.warning(msg)
            warnings.append(msg)

        result[f"{split_key}_n"] = n
        result[f"{split_key}_pct"] = round(pct * 100, 1)
        result[f"{split_key}_oob"] = len(oob)

        log.info("%-10s : %5d samples (%.1f%%, target %.0f%%)", label, n, pct * 100, expected_pct * 100)

    result["warnings"] = warnings
    result["status"] = "WARN_OOB" if warnings else "PASS"
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        log.error("transformers library not installed: %s", exc)
        sys.exit(1)

    log.info("Loading tokenizer: %s", MINILM_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)

    report: dict = {}
    failures: list[str] = []

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\n-- STAGE 1: ATS Tokenization --")
    try:
        s1 = stage1_ats(tokenizer)
    except Exception as exc:
        log.error("ATS tokenization failed: %s", exc, exc_info=True)
        failures.append(f"ATS tokenization error: {exc}")
        s1 = {"error": str(exc)}
    report["ats"] = s1

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\n-- STAGE 2: RSG Tokenization --")
    try:
        s2 = stage2_rsg(tokenizer)
    except Exception as exc:
        log.error("RSG tokenization failed: %s", exc, exc_info=True)
        failures.append(f"RSG tokenization error: {exc}")
        s2 = {"error": str(exc)}
    report["rsg"] = s2

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print("\n-- STAGE 3: Split Verification --")
    ats_n = s1.get("rows_after_dropna", 0)
    rsg_n = s2.get("rows_after_filter", 0)
    try:
        s3 = stage3_splits(ats_n, rsg_n)
    except Exception as exc:
        log.error("Split verification failed: %s", exc, exc_info=True)
        failures.append(f"Split verification error: {exc}")
        s3 = {"error": str(exc)}
    report["splits"] = s3

    # ── Write report ─────────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report written -> %s", REPORT_PATH)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  INJECTION-R0 TOKENIZATION REPORT")
    print("=" * 62)

    if "error" not in s1:
        print(f"  ATS samples     : {s1['rows_after_dropna']}")
        print(f"  ATS score range : {s1['score_range_norm'][0]:.3f} - {s1['score_range_norm'][1]:.3f}")
        print(f"  ATS domains     : {s1['domain_range']}  ({len(s1['domain_distribution'])} classes)")
        print(f"  Resume seq-len  : mean={s1['resume_seq_stats']['mean']}, "
              f"p95={s1['resume_seq_stats']['p95']}, "
              f"p99={s1['resume_seq_stats']['p99']}")
        print(f"  JD seq-len      : mean={s1['jd_seq_stats']['mean']}, "
              f"p95={s1['jd_seq_stats']['p95']}, "
              f"p99={s1['jd_seq_stats']['p99']}")
        print(f"  ATS npz         : {s1['npz_path']}  ({s1['npz_size_mb']} MB)")
    else:
        print(f"  ATS             : FAILED - {s1['error']}")

    if "error" not in s2:
        print(f"  RSG samples     : {s2['rows_after_filter']}")
        print(f"  RSG labels      : {s2['label_range']}  ({s2['num_unique_labels']} classes)")
        print(f"  Profile seq-len : mean={s2['profile_seq_stats']['mean']}, "
              f"p95={s2['profile_seq_stats']['p95']}, "
              f"p99={s2['profile_seq_stats']['p99']}")
        print(f"  RSG npz         : {s2['npz_path']}  ({s2['npz_size_mb']} MB)")
    else:
        print(f"  RSG             : FAILED - {s2['error']}")

    split_status = s3.get("status", "ERROR")
    print(f"  Splits          : {split_status}")
    if s3.get("warnings"):
        for w in s3["warnings"]:
            print(f"    WARNING: {w}")

    print(f"  Report          : {REPORT_PATH}")
    print("=" * 62)

    if failures:
        print("\n  R0 FAILED -- Hard Stop:")
        for msg in failures:
            print(f"    - {msg}")
        print()
        sys.exit(1)
    else:
        print("\n  R0 PASSED -- proceed to R1\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
