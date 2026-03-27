"""
merge_synthetic.py — Merges synthetic Legal/Education pairs with existing training data.

Loads ALL existing splits (training_pairs.csv + val_split.csv + test_split.csv),
appends synthetic data, re-splits (75/15/10), and saves back to the same files
so train.py picks up the merged data automatically.

Usage:
    python merge_synthetic.py

Outputs:
    ats-ai-core/data/labeled/training_pairs.csv  (replaced with merged train split)
    ats-ai-core/data/labeled/val_split.csv        (replaced with merged val split)
    ats-ai-core/data/labeled/test_split.csv       (replaced with merged test split)
    ats-ai-core/data/labeled/merged_with_synthetic.csv  (full merged dataset)
    data/labeled/merged_with_synthetic.csv               (copy)
"""

import shutil
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

ROOT_DIR = Path(__file__).parent.resolve()

# Paths — the actual training data is in ats-ai-core/
ATS_CORE_DIR = ROOT_DIR / "ats-ai-core"
LABELED_DIR = ATS_CORE_DIR / "data" / "labeled"
EXISTING_TRAIN_CSV = LABELED_DIR / "training_pairs.csv"
EXISTING_VAL_CSV = LABELED_DIR / "val_split.csv"
EXISTING_TEST_CSV = LABELED_DIR / "test_split.csv"

# Synthetic data
SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"
LEGAL_SYNTHETIC_CSV = SYNTHETIC_DIR / "legal_synthetic.csv"
EDUCATION_SYNTHETIC_CSV = SYNTHETIC_DIR / "education_synthetic.csv"

# Output copies
OUTPUT_CSV = ROOT_DIR / "data" / "labeled" / "merged_with_synthetic.csv"
ATS_CORE_MERGED_CSV = LABELED_DIR / "merged_with_synthetic.csv"

DOMAIN_LABELS = {
    0: "IT / Software",
    1: "Non-IT / Management",
    2: "Design / Creative",
    3: "Healthcare",
    4: "Finance / Banking",
    5: "Legal",
    6: "Education",
}

# Split ratios matching src/config.py
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.10


def load_and_align_synthetic(csv_path: Path) -> pd.DataFrame:
    """Load synthetic CSV and align columns to match training_pairs format."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} pairs from {csv_path.name}")

    aligned = pd.DataFrame({
        "resume_text": df["resume_text"],
        "jd_text": df["jd_text"],
        "score": df["ats_score"],
        "domain_index": df["domain_label"],
        "label_source": "synthetic",
    })
    return aligned


def main():
    print("=" * 60)
    print("MERGE SYNTHETIC DATA INTO TRAINING SET")
    print("=" * 60)

    # 1. Load ALL existing splits to reconstruct the full original dataset
    print(f"\n[1] Loading existing data splits...")
    parts = []
    for csv_path in [EXISTING_TRAIN_CSV, EXISTING_VAL_CSV, EXISTING_TEST_CSV]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {csv_path.name}: {len(df)} rows")
            parts.append(df)
        else:
            print(f"  WARNING: {csv_path.name} not found, skipping")

    if not parts:
        print("  ERROR: No existing data found!")
        return

    existing_df = pd.concat(parts, ignore_index=True)
    print(f"  Total existing: {len(existing_df)} pairs")

    # Original domain counts
    print("\n  Original domain distribution:")
    for idx in sorted(existing_df["domain_index"].unique()):
        count = int((existing_df["domain_index"] == idx).sum())
        label = DOMAIN_LABELS.get(idx, f"Unknown({idx})")
        print(f"    Domain {idx} ({label}): {count}")

    original_legal = int((existing_df["domain_index"] == 5).sum())
    original_education = int((existing_df["domain_index"] == 6).sum())

    # 2. Load synthetic data
    print(f"\n[2] Loading synthetic data...")

    if not LEGAL_SYNTHETIC_CSV.exists():
        print(f"  ERROR: {LEGAL_SYNTHETIC_CSV} not found! Run generate_synthetic.py first.")
        return
    if not EDUCATION_SYNTHETIC_CSV.exists():
        print(f"  ERROR: {EDUCATION_SYNTHETIC_CSV} not found! Run generate_synthetic.py first.")
        return

    legal_df = load_and_align_synthetic(LEGAL_SYNTHETIC_CSV)
    education_df = load_and_align_synthetic(EDUCATION_SYNTHETIC_CSV)

    # 3. Merge
    print(f"\n[3] Merging datasets...")
    merged_df = pd.concat([existing_df, legal_df, education_df], ignore_index=True)
    print(f"  Merged total: {len(merged_df)} pairs")

    # 4. Verify domain counts
    print(f"\n[4] Verifying domain counts after merge:")
    for idx in sorted(merged_df["domain_index"].unique()):
        count = int((merged_df["domain_index"] == idx).sum())
        label = DOMAIN_LABELS.get(idx, f"Unknown({idx})")
        print(f"    Domain {idx} ({label}): {count}")

    new_legal = int((merged_df["domain_index"] == 5).sum())
    new_education = int((merged_df["domain_index"] == 6).sum())

    print(f"\n  Legal:     {original_legal} + {new_legal - original_legal} new = {new_legal} total")
    print(f"  Education: {original_education} + {new_education - original_education} new = {new_education} total")

    # Verify other domains unchanged
    for idx in sorted(existing_df["domain_index"].unique()):
        if idx not in (5, 6):
            old_count = int((existing_df["domain_index"] == idx).sum())
            new_count = int((merged_df["domain_index"] == idx).sum())
            if old_count != new_count:
                print(f"  WARNING: Domain {idx} changed from {old_count} to {new_count}!")
            else:
                print(f"  Domain {idx} ({DOMAIN_LABELS.get(idx, '?')}): unchanged at {old_count}")

    # 5. Save full merged dataset
    print(f"\n[5] Saving full merged dataset...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved to: {OUTPUT_CSV}")

    ATS_CORE_MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(ATS_CORE_MERGED_CSV, index=False)
    print(f"  Saved to: {ATS_CORE_MERGED_CSV}")

    # 6. Re-split merged data (split AFTER merging, not before)
    print(f"\n[6] Re-splitting merged data (train/val/test = 75/15/10)...")
    merged_shuffled = merged_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    n = len(merged_shuffled)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * VALIDATION_SPLIT)

    test_split = merged_shuffled.iloc[:n_test]
    val_split = merged_shuffled.iloc[n_test: n_test + n_val]
    train_split = merged_shuffled.iloc[n_test + n_val:]

    print(f"  Train: {len(train_split)}  Val: {len(val_split)}  Test: {len(test_split)}")

    # Backup originals
    print(f"\n[7] Backing up original splits and saving new ones...")
    for csv_path in [EXISTING_TRAIN_CSV, EXISTING_VAL_CSV, EXISTING_TEST_CSV]:
        if csv_path.exists():
            backup = csv_path.with_suffix(".csv.bak")
            if not backup.exists():
                shutil.copy2(csv_path, backup)
                print(f"  Backed up: {csv_path.name} -> {backup.name}")

    # Save new splits (replaces originals so train.py uses merged data)
    train_split.to_csv(EXISTING_TRAIN_CSV, index=False)
    print(f"  Saved train split: {EXISTING_TRAIN_CSV.name} ({len(train_split)} rows)")

    val_split.to_csv(EXISTING_VAL_CSV, index=False)
    print(f"  Saved val split:   {EXISTING_VAL_CSV.name} ({len(val_split)} rows)")

    test_split.to_csv(EXISTING_TEST_CSV, index=False)
    print(f"  Saved test split:  {EXISTING_TEST_CSV.name} ({len(test_split)} rows)")

    # 8. Print per-split domain distribution
    print(f"\n[8] Domain distribution per split:")
    for name, split_df in [("Train", train_split), ("Val", val_split), ("Test", test_split)]:
        print(f"\n  {name} ({len(split_df)} rows):")
        for idx in sorted(split_df["domain_index"].unique()):
            count = int((split_df["domain_index"] == idx).sum())
            label = DOMAIN_LABELS.get(idx, f"Unknown({idx})")
            print(f"    Domain {idx} ({label}): {count}")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nFull merged dataset: {ATS_CORE_MERGED_CSV}")
    print(f"Total pairs: {len(merged_df)}")
    print(f"Train/Val/Test splits replaced in: {LABELED_DIR}")
    print("\nCRITICAL: TF-IDF must be fit on training portion only (no data leakage).")
    print("\nNext steps:")
    print("  cd ats-ai-core")
    print("  python train.py --skip-conversion")
    print("  python evaluation/ats_eval.py")


if __name__ == "__main__":
    main()
