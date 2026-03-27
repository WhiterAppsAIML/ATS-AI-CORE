"""
train.py — ATS AI Core Master Training Pipeline
================================================
Runs the full end-to-end training pipeline in sequence:
  1. Load and clean all three datasets
  2. Build training pairs (weak labels from DS-1 × DS-2)
  3. Merge with gold labels from DS-3
  4. Build and compile the ATS model
  5. Train with multi-task loss
  6. Evaluate on the held-out test split
  7. Convert to TFLite and validate

Usage:
    # Activate your venv first, then:
    python train.py

    # Skip conversion (train only):
    python train.py --skip-conversion

    # Use a smaller dataset for a quick smoke test:
    python train.py --smoke-test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# TF Hub KerasLayer requires Keras 2 (tf_keras) — not compatible with Keras 3.
# Must be set BEFORE importing tensorflow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ATS AI Core training pipeline")
    parser.add_argument(
        "--skip-conversion", action="store_true",
        help="Skip TFLite conversion after training",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Use 200 samples per dataset for a quick end-to-end check",
    )
    parser.add_argument(
        "--unfreeze-encoder", action="store_true",
        help="Unfreeze USE Lite encoder weights (only if MAE > 10.0)",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Force regeneration of training pairs and data splits",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from src.config import (
        ATS_MODEL_DIR, GOLD_LABELS_CSV, LABELED_DIR,
        RANDOM_SEED, TRAINING_PAIRS_CSV, WEAK_LABELS_CSV,
    )
    from src.preprocessing.data_loader import load_all_datasets
    from src.preprocessing.domain_mapper import map_text
    from src.preprocessing.normalizer import normalize_text
    from src.preprocessing.pair_builder import build_pairs, merge_with_gold, save_training_pairs
    from src.preprocessing.text_cleaner import clean_text
    from src.ats_engine.model import build_ats_model
    from src.ats_engine.trainer import load_training_data, train
    from evaluation.ats_eval import evaluate_ats_model

    smoke = args.smoke_test

    # ── STEP 1: Load datasets ────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1 - Loading datasets")
    logger.info("=" * 55)
    datasets = load_all_datasets()
    resumes_df = datasets["resumes"]
    jobs_df    = datasets["jobs"]
    gold_df    = datasets["scored_pairs"]

    # Cap LinkedIn JDs to avoid processing millions of rows when only
    # MAX_PAIRS_PER_DOMAIN * NUM_DOMAINS pairs are needed.
    MAX_JDS = 50_000
    if len(jobs_df) > MAX_JDS:
        logger.info("Sampling %d JDs from %d to speed up pair building", MAX_JDS, len(jobs_df))
        jobs_df = jobs_df.sample(n=MAX_JDS, random_state=RANDOM_SEED)

    if smoke:
        resumes_df = resumes_df.sample(n=min(200, len(resumes_df)), random_state=RANDOM_SEED)
        jobs_df    = jobs_df.sample(n=min(200, len(jobs_df)), random_state=RANDOM_SEED)
        gold_df    = gold_df.sample(n=min(200, len(gold_df)), random_state=RANDOM_SEED)
        logger.info("SMOKE TEST MODE: using %d resumes / %d jobs / %d gold pairs",
                    len(resumes_df), len(jobs_df), len(gold_df))

    # ── STEP 2: Build weak-labeled pairs ─────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 2 - Building training pairs")
    logger.info("=" * 55)

    if TRAINING_PAIRS_CSV.exists() and not args.force_rebuild:
        logger.info("training_pairs.csv already exists - skipping pair building.")
        logger.info("Use --force-rebuild to rebuild.")
    else:
        weak_df = build_pairs(resumes_df, jobs_df)
        weak_df.to_csv(WEAK_LABELS_CSV, index=False)
        logger.info("Weak labels saved to %s", WEAK_LABELS_CSV)

        # Tag gold labels with domain
        if "domain_index" not in gold_df.columns or (gold_df["domain_index"] == -1).all():
            gold_df["domain_index"] = gold_df["jd_text"].apply(map_text)
        gold_df.to_csv(GOLD_LABELS_CSV, index=False)

        combined = merge_with_gold(weak_df, gold_df)

        # Save train/val/test splits separately for clean evaluation
        combined_shuffled = combined.sample(frac=1, random_state=RANDOM_SEED)
        n = len(combined_shuffled)
        n_test = int(n * 0.10)
        n_val  = int(n * 0.15)
        combined_shuffled.iloc[:n_test].to_csv(LABELED_DIR / "test_split.csv", index=False)
        combined_shuffled.iloc[n_test: n_test + n_val].to_csv(
            LABELED_DIR / "val_split.csv", index=False
        )
        combined_shuffled.iloc[n_test + n_val:].to_csv(TRAINING_PAIRS_CSV, index=False)
        logger.info("All splits saved to %s", LABELED_DIR)

        # Domain audit
        from src.config import DOMAIN_LABELS
        domain_counts = combined_shuffled["domain_index"].value_counts()
        logger.info("Domain distribution:")
        for dom, count in domain_counts.items():
            name = DOMAIN_LABELS.get(dom, f"Unknown({dom})")
            logger.info("  Domain %s (%s): %d samples", dom, name, count)
            if count < 100:
                logger.warning("  -> UNDERREPRESENTED DOMAIN: %s has only %d samples", name, count)

    # ── STEP 3: Build model ───────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3 - Building ATS model")
    logger.info("=" * 55)
    model = build_ats_model(frozen_encoder=not args.unfreeze_encoder)
    model.summary(line_length=100)

    # ── STEP 4: Train ─────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 4 - Training")
    logger.info("=" * 55)

    # Use pre-saved splits from Step 2 instead of re-splitting.
    # load_training_data re-splits training_pairs.csv which creates a tiny
    # test set and overwrites the balanced test_split.csv from Step 2.
    val_csv = LABELED_DIR / "val_split.csv"
    test_csv = LABELED_DIR / "test_split.csv"
    if val_csv.exists() and test_csv.exists():
        logger.info("Using pre-saved train/val/test splits from Step 2.")
        import numpy as np
        train_df = pd.read_csv(TRAINING_PAIRS_CSV)
        train_df = train_df.dropna(subset=["resume_text", "jd_text", "score"])
        train_df = train_df[train_df["domain_index"] >= 0]
        train_df["score_norm"] = train_df["score"].clip(0, 100) / 100.0

        val_df = pd.read_csv(val_csv)
        val_df = val_df.dropna(subset=["resume_text", "jd_text", "score"])
        val_df = val_df[val_df["domain_index"] >= 0]
        val_df["score_norm"] = val_df["score"].clip(0, 100) / 100.0

        logger.info("Split: train=%d  val=%d  test=%d",
                     len(train_df), len(val_df), len(pd.read_csv(test_csv)))
    else:
        train_df, val_df, test_df = load_training_data(TRAINING_PAIRS_CSV)
        test_df.to_csv(LABELED_DIR / "test_split.csv", index=False)

    history = train(model, train_df, val_df)

    # ── STEP 5: Evaluate ──────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 5 - Evaluation")
    logger.info("=" * 55)
    results = evaluate_ats_model(
        model_path=ATS_MODEL_DIR / "final_model_weights.h5",
        test_csv=LABELED_DIR / "test_split.csv",
    )

    # ── STEP 6: TFLite conversion ─────────────────────────────────────────────
    if not args.skip_conversion:
        logger.info("=" * 55)
        logger.info("STEP 6 - TFLite Conversion")
        logger.info("=" * 55)
        from src.conversion.convert_to_tflite import convert_and_validate
        report = convert_and_validate()
        if not report["passed"]:
            logger.error("TFLite conversion failed. Check logs above.")
            sys.exit(1)

    logger.info("=" * 55)
    logger.info("Pipeline complete.")
    logger.info("MAE=%.2f  Domain F1=%.4f", results["mae"], results["domain_f1"])
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
