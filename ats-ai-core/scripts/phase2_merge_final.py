"""
Phase 2 — Merge all data sources into merged_final.csv.

Combines:
1. Original labeled data (minus any previous synthetic rows)
2. Corrected Legal + Education synthetic data
3. New IT + Finance supplemental data

Prints domain distribution and saves merged_final.csv.
"""

from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABELED_DIR = Path(__file__).resolve().parent.parent / "data" / "labeled"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

DOMAIN_MAP = {
    0: "IT / Software",
    1: "Non-IT / Management",
    2: "Design / Creative",
    3: "Healthcare",
    4: "Finance / Banking",
    5: "Legal",
    6: "Education",
}


def load_original_data() -> pd.DataFrame:
    """Load original training data from .bak files or current files, excluding synthetic rows."""
    frames = []
    for fname in ["training_pairs.csv.bak", "val_split.csv.bak", "test_split.csv.bak"]:
        path = LABELED_DIR / fname
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            # Fall back to non-bak version
            base = fname.replace(".bak", "")
            path = LABELED_DIR / base
            if path.exists():
                frames.append(pd.read_csv(path))

    original = pd.concat(frames, ignore_index=True)

    # Remove any previously merged synthetic data
    if "label_source" in original.columns:
        original = original[original["label_source"] != "synthetic"]

    return original


def load_synthetic_file(path: Path) -> pd.DataFrame:
    """Load a synthetic CSV and normalize column names to match training format."""
    df = pd.read_csv(path)
    # Rename columns if needed
    rename_map = {}
    if "ats_score" in df.columns:
        rename_map["ats_score"] = "score"
    if "domain_label" in df.columns:
        rename_map["domain_label"] = "domain_index"
    if rename_map:
        df = df.rename(columns=rename_map)
    df["label_source"] = "synthetic"
    return df


def main():
    print("=" * 55)
    print("  MERGING ALL DATA INTO merged_final.csv")
    print("=" * 55)

    # Load original data
    original = load_original_data()
    print(f"\n  Original data rows: {len(original)}")

    # Load all synthetic files
    synthetic_files = [
        SYNTHETIC_DIR / "legal_synthetic.csv",
        SYNTHETIC_DIR / "education_synthetic.csv",
        SYNTHETIC_DIR / "it_supplemental.csv",
        SYNTHETIC_DIR / "finance_supplemental.csv",
    ]

    synthetic_frames = []
    for sf in synthetic_files:
        if sf.exists():
            df = load_synthetic_file(sf)
            print(f"  Loaded {sf.name}: {len(df)} rows")
            synthetic_frames.append(df)
        else:
            print(f"  WARNING: {sf.name} not found — skipping")

    synthetic = pd.concat(synthetic_frames, ignore_index=True)
    print(f"  Total synthetic rows: {len(synthetic)}")

    # Merge
    merged = pd.concat([original, synthetic], ignore_index=True)
    merged = merged.dropna(subset=["resume_text", "jd_text", "score"])
    merged = merged[merged["domain_index"] >= 0]

    # Print domain counts
    counts = Counter(merged["domain_index"].astype(int).tolist())
    print(f"\n  Domain distribution in merged_final.csv:")
    all_ok = True
    for idx in sorted(DOMAIN_MAP.keys()):
        name = DOMAIN_MAP[idx]
        c = counts.get(idx, 0)
        flag = ""
        if c < 600:
            flag = " ** BELOW 600 **"
            all_ok = False
        if c > 2500:
            flag = " ** ABOVE 2500 **"
            all_ok = False
        print(f"    {name:24s}: {c}{flag}")
    print(f"    {'TOTAL':24s}: {len(merged)}")

    if not all_ok:
        print("\n  WARNING: Some domains outside 600-2500 range!")

    # Save
    output_path = LABELED_DIR / "merged_final.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
