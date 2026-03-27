"""
Phase 1 — Audit and fix fresher score labels in synthetic CSVs.

Reads legal_synthetic.csv and education_synthetic.csv, detects fresher
profiles, assesses keyword alignment between resume and JD, then re-scores
according to the correction rules from AGENT_FAIRNESS_AND_DOMAIN_FIX.md.

Outputs corrected CSVs in-place, then re-merges with original training
data to produce merged_corrected.csv.
"""

import re
import random
from pathlib import Path

import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
LABELED_DIR = Path(__file__).resolve().parent.parent / "data" / "labeled"

LEGAL_CSV = SYNTHETIC_DIR / "legal_synthetic.csv"
EDUCATION_CSV = SYNTHETIC_DIR / "education_synthetic.csv"

# ── Fresher detection patterns ──────────────────────────────────────────────
FRESHER_PATTERNS = re.compile(
    r"fresher|fresh\s*graduate|entry[- ]level|intern(?:ship)?|"
    r"0\s*years?\s*(?:of\s*)?(?:experience|professional)|"
    r"final[- ]year\s*(?:student|B\.?Ed|LLB|B\.?Com|B\.?Sc|B\.?A|M\.?B\.?A)|"
    r"looking for entry|recent(?:ly)?\s*graduat|"
    r"no\s*(?:prior\s*)?(?:professional\s*)?experience|"
    r"B\.?Ed\s*student|LLB\s*student|"
    r"trainee|articleship|pupillage",
    re.IGNORECASE,
)


def is_fresher(resume_text: str) -> bool:
    """Detect if a resume belongs to a fresher profile."""
    return bool(FRESHER_PATTERNS.search(str(resume_text)))


# ── Domain keyword pools for alignment scoring ──────────────────────────────
LEGAL_KEYWORDS = [
    "contract", "drafting", "corporate", "m&a", "merger", "acquisition",
    "due diligence", "compliance", "regulatory", "llb", "llm", "legal research",
    "litigation", "civil litigation", "court", "filing", "pleading", "brief",
    "case management", "discovery", "trial", "oral argument", "bar admission",
    "risk assessment", "internal audit", "policy", "gdpr", "legal framework",
    "paralegal", "documentation", "client communication", "hearing", "scheduling",
    "patent", "trademark", "ip", "intellectual property", "licensing",
    "copyright", "wipo", "prior art", "negotiation", "advisory", "counsel",
    "advocate", "arbitration", "mediation", "sebi", "banking law",
    "insolvency", "bankruptcy", "criminal law", "constitutional", "jurisdiction",
]

EDUCATION_KEYWORDS = [
    "lesson planning", "classroom management", "curriculum", "student assessment",
    "b.ed", "teaching certification", "differentiated instruction", "pedagogy",
    "course design", "lecture", "research publication", "phd", "master",
    "academic writing", "mentoring", "syllabus", "peer review", "professor",
    "e-learning", "lms", "moodle", "canvas", "articulate", "captivate",
    "instructional design", "addie", "learning objective", "multimedia",
    "academic planning", "timetable", "faculty coordination", "accreditation",
    "student records", "program administration", "academic coordinator",
    "special needs", "iep", "counseling", "behavioral support", "inclusive",
    "rci", "therapeutic", "special education", "school", "teaching",
    "teacher", "education", "training", "workshop", "certification",
    "assessment", "evaluation", "grading", "cbse", "icse", "ncert",
]


def compute_keyword_overlap(resume_text: str, jd_text: str, keyword_pool: list[str]) -> float:
    """Compute the fraction of JD-relevant keywords found in the resume.

    1. Find which keywords from the pool appear in the JD.
    2. Of those, find how many appear in the resume.
    3. Return the ratio.
    """
    resume_lower = str(resume_text).lower()
    jd_lower = str(jd_text).lower()

    jd_keywords = [kw for kw in keyword_pool if kw in jd_lower]
    if not jd_keywords:
        return 0.0

    matched = sum(1 for kw in jd_keywords if kw in resume_lower)
    return matched / len(jd_keywords)


def classify_match(overlap: float) -> str:
    """Classify keyword overlap into match level."""
    if overlap >= 0.55:
        return "strong"
    elif overlap >= 0.30:
        return "moderate"
    elif overlap >= 0.12:
        return "weak"
    else:
        return "mismatched"


# ── Score correction rules ──────────────────────────────────────────────────
FRESHER_SCORE_RANGES = {
    "strong":     (60, 85),
    "moderate":   (35, 60),
    "weak":       (15, 35),
    "mismatched": (5, 20),
}

EXPERIENCED_SCORE_RANGES = {
    "strong":     (70, 95),
    "moderate":   (45, 70),
    "weak":       (20, 45),
    "mismatched": (5, 25),
}


def correct_score(row: pd.Series, keyword_pool: list[str]) -> float:
    """Compute a corrected score for a single row."""
    fresher = is_fresher(row["resume_text"])
    overlap = compute_keyword_overlap(row["resume_text"], row["jd_text"], keyword_pool)
    match_level = classify_match(overlap)

    if fresher:
        lo, hi = FRESHER_SCORE_RANGES[match_level]
    else:
        lo, hi = EXPERIENCED_SCORE_RANGES[match_level]

    # Generate a score within the range, with some variance
    new_score = round(random.uniform(lo, hi), 1)
    return new_score


def audit_and_fix(csv_path: Path, keyword_pool: list[str], domain_name: str) -> pd.DataFrame:
    """Audit a synthetic CSV and return corrected DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"\n{'='*55}")
    print(f"  AUDITING: {domain_name} ({csv_path.name})")
    print(f"{'='*55}")
    print(f"  Total rows: {len(df)}")

    # Detect freshers
    df["is_fresher"] = df["resume_text"].apply(is_fresher)
    n_fresher = df["is_fresher"].sum()
    print(f"  Freshers detected: {n_fresher} ({n_fresher/len(df)*100:.1f}%)")

    # Before stats
    fresher_mask = df["is_fresher"]
    print(f"\n  BEFORE CORRECTION:")
    print(f"    Fresher mean score:     {df.loc[fresher_mask, 'ats_score'].mean():.1f}")
    print(f"    Experienced mean score: {df.loc[~fresher_mask, 'ats_score'].mean():.1f}")
    gap_before = df.loc[~fresher_mask, "ats_score"].mean() - df.loc[fresher_mask, "ats_score"].mean()
    print(f"    Gap (exp - fresher):    {gap_before:.1f}")
    print(f"    Freshers with score > 85: {(df.loc[fresher_mask, 'ats_score'] > 85).sum()}")

    # Apply corrections
    df["ats_score"] = df.apply(lambda r: correct_score(r, keyword_pool), axis=1)

    # After stats
    print(f"\n  AFTER CORRECTION:")
    print(f"    Fresher mean score:     {df.loc[fresher_mask, 'ats_score'].mean():.1f}")
    print(f"    Experienced mean score: {df.loc[~fresher_mask, 'ats_score'].mean():.1f}")
    gap_after = df.loc[~fresher_mask, "ats_score"].mean() - df.loc[fresher_mask, "ats_score"].mean()
    print(f"    Gap (exp - fresher):    {gap_after:.1f}")
    print(f"    Freshers with score > 85: {(df.loc[fresher_mask, 'ats_score'] > 85).sum()}")
    print(f"    Experienced with score < 20: {(df.loc[~fresher_mask, 'ats_score'] < 20).sum()}")

    # Score distribution after
    print(f"\n  Score distribution after correction:")
    bins = [(85, 100, "Excellent"), (65, 84, "Good"), (45, 64, "Moderate"),
            (25, 44, "Weak"), (0, 24, "Poor")]
    for lo, hi, label in bins:
        count = ((df["ats_score"] >= lo) & (df["ats_score"] <= hi)).sum()
        print(f"    {label:12s} ({lo:3d}-{hi:3d}): {count}")

    # Verify gap is within 10-15 pts
    if abs(gap_after) > 15:
        print(f"\n  WARNING: Gap {gap_after:.1f} exceeds 15 pts target. Adjusting...")
        # Pull fresher scores slightly up or experienced slightly down
        # We'll adjust freshers up since we don't want to reduce experienced
        target_gap = 12.0  # target 12 pts
        current_exp_mean = df.loc[~fresher_mask, "ats_score"].mean()
        desired_fresher_mean = current_exp_mean - target_gap
        current_fresher_mean = df.loc[fresher_mask, "ats_score"].mean()
        shift = desired_fresher_mean - current_fresher_mean
        if shift > 0:
            df.loc[fresher_mask, "ats_score"] = (
                df.loc[fresher_mask, "ats_score"] + shift
            ).clip(0, 100).round(1)
            gap_after = df.loc[~fresher_mask, "ats_score"].mean() - df.loc[fresher_mask, "ats_score"].mean()
            print(f"    Adjusted fresher scores up by {shift:.1f}. New gap: {gap_after:.1f}")

    # Drop helper column
    df.drop(columns=["is_fresher"], inplace=True)

    return df


def merge_corrected_with_original():
    """Merge corrected synthetic data with original training data."""
    print(f"\n{'='*55}")
    print(f"  MERGING CORRECTED DATA")
    print(f"{'='*55}")

    # Load original training data (all three splits)
    train_df = pd.read_csv(LABELED_DIR / "training_pairs.csv.bak") if (LABELED_DIR / "training_pairs.csv.bak").exists() else pd.read_csv(LABELED_DIR / "training_pairs.csv")
    val_df = pd.read_csv(LABELED_DIR / "val_split.csv.bak") if (LABELED_DIR / "val_split.csv.bak").exists() else pd.read_csv(LABELED_DIR / "val_split.csv")
    test_df = pd.read_csv(LABELED_DIR / "test_split.csv.bak") if (LABELED_DIR / "test_split.csv.bak").exists() else pd.read_csv(LABELED_DIR / "test_split.csv")

    # Combine original data
    original = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # Remove any previously merged synthetic data (label_source == 'synthetic')
    if "label_source" in original.columns:
        original = original[original["label_source"] != "synthetic"]

    print(f"  Original data rows: {len(original)}")
    print(f"  Original columns: {list(original.columns)}")

    # Load corrected synthetic data
    legal = pd.read_csv(LEGAL_CSV)
    edu = pd.read_csv(EDUCATION_CSV)
    synthetic = pd.concat([legal, edu], ignore_index=True)

    # Rename columns to match training format
    synthetic = synthetic.rename(columns={
        "ats_score": "score",
        "domain_label": "domain_index",
    })
    synthetic["label_source"] = "synthetic"

    print(f"  Synthetic data rows: {len(synthetic)}")

    # Merge
    merged = pd.concat([original, synthetic], ignore_index=True)
    merged = merged.dropna(subset=["resume_text", "jd_text", "score"])
    merged = merged[merged["domain_index"] >= 0]

    # Print domain counts
    from collections import Counter
    domain_map = {0: "IT / Software", 1: "Non-IT / Management", 2: "Design / Creative",
                  3: "Healthcare", 4: "Finance / Banking", 5: "Legal", 6: "Education"}
    counts = Counter(merged["domain_index"].astype(int).tolist())
    print(f"\n  Domain distribution after merge:")
    for idx in sorted(domain_map.keys()):
        name = domain_map[idx]
        print(f"    {name:24s}: {counts.get(idx, 0)}")
    print(f"    {'TOTAL':24s}: {len(merged)}")

    # Save
    output_path = LABELED_DIR / "merged_corrected.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    return output_path


def main():
    # Phase 1 Task 1: Audit and fix scores
    legal_df = audit_and_fix(LEGAL_CSV, LEGAL_KEYWORDS, "Legal")
    legal_df.to_csv(LEGAL_CSV, index=False)
    print(f"\n  Corrected Legal CSV saved to: {LEGAL_CSV}")

    edu_df = audit_and_fix(EDUCATION_CSV, EDUCATION_KEYWORDS, "Education")
    edu_df.to_csv(EDUCATION_CSV, index=False)
    print(f"\n  Corrected Education CSV saved to: {EDUCATION_CSV}")

    # Phase 1 Task 2: Merge
    merge_corrected_with_original()

    print("\n" + "=" * 55)
    print("  PHASE 1 AUDIT COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
