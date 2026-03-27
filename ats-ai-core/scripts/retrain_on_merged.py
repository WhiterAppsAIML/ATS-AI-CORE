"""
Retrain script — loads merged CSV, splits, trains, evaluates.

Usage:
    python scripts/retrain_on_merged.py --data data/labeled/merged_corrected.csv
    python scripts/retrain_on_merged.py --data data/labeled/merged_final.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retrain ATS model on merged data")
    parser.add_argument("--data", type=str, required=True, help="Path to merged CSV")
    args = parser.parse_args()

    from src.config import (
        ATS_MODEL_DIR, LABELED_DIR, RANDOM_SEED,
        DOMAIN_LABELS, BATCH_SIZE, EPOCHS,
    )
    from src.ats_engine.model import build_ats_model
    from src.ats_engine.trainer import load_training_data, train
    from evaluation.ats_eval import evaluate_ats_model

    csv_path = Path(args.data)
    if not csv_path.is_absolute():
        csv_path = Path(_PROJECT_ROOT) / csv_path

    logger.info("=" * 55)
    logger.info("RETRAIN — Loading data from %s", csv_path)
    logger.info("=" * 55)

    # Load and split
    train_df, val_df, test_df = load_training_data(csv_path)

    # Save test split for evaluation
    test_csv_path = LABELED_DIR / "test_split.csv"
    test_df.to_csv(test_csv_path, index=False)
    logger.info("Test split saved to %s (%d rows)", test_csv_path, len(test_df))

    # Print domain distribution
    logger.info("Domain distribution in training set:")
    counts = train_df["domain_index"].astype(int).value_counts()
    for idx in sorted(DOMAIN_LABELS.keys()):
        name = DOMAIN_LABELS[idx]
        c = counts.get(idx, 0)
        logger.info("  %s: %d", name, c)
    logger.info("  TOTAL: %d", len(train_df))

    # Build model
    logger.info("=" * 55)
    logger.info("Building ATS model")
    logger.info("=" * 55)
    model = build_ats_model(frozen_encoder=True)
    model.summary(line_length=100)

    # Train
    logger.info("=" * 55)
    logger.info("Training")
    logger.info("=" * 55)
    history = train(model, train_df, val_df)

    # Evaluate
    logger.info("=" * 55)
    logger.info("Evaluation")
    logger.info("=" * 55)
    results = evaluate_ats_model(
        model_path=ATS_MODEL_DIR / "final_model_weights.h5",
        test_csv=test_csv_path,
    )

    logger.info("=" * 55)
    logger.info("Retrain complete.")
    logger.info("MAE=%.2f  Domain F1=%.4f", results["mae"], results["domain_f1"])
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
