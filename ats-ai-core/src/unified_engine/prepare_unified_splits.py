"""
R-0: Data preparation and canonical splits for unified retraining.

Copies external RSG data (optional), validates ATS and RSG datasets,
and saves train/val/test indices to data_splits.json.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Allow `import src.*` when running from ats-ai-core project root.
sys.path.insert(0, ".")

from src.config import (  # noqa: E402
    LABELED_DIR,
    MIN_PAIRS_PER_DOMAIN,
    NUM_DOMAINS,
    RANDOM_SEED,
    RSG_CSV_PATH,
    RSG_MAPPING_JSON,
    UNIFIED_MODEL_DIR,
)
from src.unified_engine.data_loader import load_ats_data, load_rsg_data  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate datasets, then create canonical splits."
    )
    parser.add_argument(
        "--rsg-source",
        type=str,
        default=None,
        help="Path to external weak_labels.csv to copy into RSG_CSV_PATH",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local RSG_CSV_PATH if it already exists",
    )
    parser.add_argument(
        "--ats-csv",
        type=str,
        default=str(LABELED_DIR / "merged_final.csv"),
        help="Path to ATS merged_final.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for train/test split",
    )
    return parser.parse_args()


def _copy_rsg_data(source: Path, destination: Path, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"RSG source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"RSG CSV already exists at {destination}")
        print("Skip copy. Use --overwrite to replace it.")
        return
    shutil.copy2(str(source), str(destination))
    print(f"Copied RSG CSV to {destination}")


def _validate_ats(ats_csv: Path) -> np.ndarray:
    if not ats_csv.exists():
        raise FileNotFoundError(f"ATS CSV not found: {ats_csv}")

    resume_texts, jd_texts, scores, domains = load_ats_data(str(ats_csv))
    count = len(resume_texts)
    print(f"ATS rows: {count}")
    if count < 60000:
        raise AssertionError(f"Too few ATS pairs: {count}")

    score_min = float(scores.min())
    score_max = float(scores.max())
    print(f"ATS score range: {score_min:.6f} to {score_max:.6f}")
    if score_min < 0.0 or score_max > 1.0:
        raise AssertionError("ATS scores must be in [0.0, 1.0]")

    domain_min = int(domains.min())
    domain_max = int(domains.max())
    print(f"ATS domain range: {domain_min} to {domain_max}")
    if domain_min < 0 or domain_max > (NUM_DOMAINS - 1):
        raise AssertionError("ATS domains must be in [0, 6]")

    counts = np.bincount(domains.astype(int), minlength=NUM_DOMAINS)
    for d in range(NUM_DOMAINS):
        print(f"  Domain {d}: {int(counts[d])} pairs")
        if counts[d] < MIN_PAIRS_PER_DOMAIN:
            raise AssertionError(
                f"Domain {d} has only {int(counts[d])} pairs; "
                f"min is {MIN_PAIRS_PER_DOMAIN}"
            )

    return domains


def _validate_rsg(rsg_csv: Path, mapping_path: Path) -> np.ndarray:
    if not rsg_csv.exists():
        raise FileNotFoundError(f"RSG CSV not found: {rsg_csv}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"RSG mapping not found: {mapping_path}")

    _, template_ids = load_rsg_data(str(rsg_csv))
    print(f"RSG rows: {len(template_ids)}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if "id_to_idx" not in mapping:
        raise KeyError("rsg_label_mapping.json missing id_to_idx")

    id_to_idx = {int(k): int(v) for k, v in mapping["id_to_idx"].items()}
    valid_mask = np.array([int(t) in id_to_idx for t in template_ids], dtype=bool)
    valid_count = int(valid_mask.sum())
    print(f"RSG valid: {valid_count} / {len(template_ids)}")
    if valid_count < 1000:
        raise AssertionError(f"Too few valid RSG samples: {valid_count}")

    return valid_mask


def _build_splits(domains: np.ndarray, rsg_valid_mask: np.ndarray, seed: int) -> dict:
    idx = np.arange(len(domains))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.25,
        random_state=seed,
        stratify=domains,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.40,
        random_state=seed,
        stratify=domains[temp_idx],
    )

    rsg_valid_count = int(rsg_valid_mask.sum())
    rsg_idx = np.arange(rsg_valid_count)
    rsg_train_idx, rsg_val_idx = train_test_split(
        rsg_idx,
        test_size=0.20,
        random_state=seed,
    )

    print("ATS split sizes:")
    print(f"  train: {len(train_idx)}")
    print(f"  val:   {len(val_idx)}")
    print(f"  test:  {len(test_idx)}")
    print("RSG split sizes (valid-only indices):")
    print(f"  train: {len(rsg_train_idx)}")
    print(f"  val:   {len(rsg_val_idx)}")

    return {
        "ats_train": train_idx.tolist(),
        "ats_val": val_idx.tolist(),
        "ats_test": test_idx.tolist(),
        "rsg_train": rsg_train_idx.tolist(),
        "rsg_val": rsg_val_idx.tolist(),
    }


def _save_splits(splits: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {output_path}")


def main() -> None:
    args = _parse_args()

    if args.rsg_source:
        _copy_rsg_data(Path(args.rsg_source), RSG_CSV_PATH, args.overwrite)
    else:
        if not RSG_CSV_PATH.exists():
            raise FileNotFoundError(
                "RSG_CSV_PATH missing. Pass --rsg-source to copy weak_labels.csv"
            )
        print(f"RSG CSV present at {RSG_CSV_PATH}")

    ats_csv = Path(args.ats_csv)
    domains = _validate_ats(ats_csv)
    rsg_valid_mask = _validate_rsg(RSG_CSV_PATH, RSG_MAPPING_JSON)

    splits = _build_splits(domains, rsg_valid_mask, args.seed)
    output_path = UNIFIED_MODEL_DIR / "data_splits.json"
    _save_splits(splits, output_path)


if __name__ == "__main__":
    main()
