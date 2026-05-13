"""
B-3c Task 1: RSG Data Augmentation
=====================================
Identifies RSG classes with < 50 samples and generates synthetic
variations using synonym replacement to boost each to >= 50 samples.

Strategy:
  - For each minority class (n < 50), sample existing texts and apply
    word-level synonym replacement using NLTK WordNet.
  - Each augmented text replaces 20-40% of words with synonyms.
  - Multiple augmentations per source text to reach target count.
  - Output: balanced CSV with original + synthetic samples.
"""
import os
import sys
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from src.config import RSG_CSV_PATH, RSG_BALANCED_CSV_PATH

# NLTK setup
import nltk
try:
    from nltk.corpus import wordnet
    # Quick test to see if wordnet data is downloaded
    wordnet.synsets("test")
except LookupError:
    print("Downloading NLTK wordnet data...")
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet

# -- Paths ----------------------------------------------------------------
RSG_CSV = RSG_CSV_PATH
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
OUTPUT_CSV = RSG_BALANCED_CSV_PATH

# -- Config ---------------------------------------------------------------
MIN_SAMPLES_PER_CLASS = 50
REPLACE_RATIO_MIN = 0.20   # Replace 20-40% of words
REPLACE_RATIO_MAX = 0.40
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def get_synonyms(word):
    """Get synonyms for a word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def synonym_replace(text, replace_ratio=0.3):
    """Replace a fraction of words with WordNet synonyms.

    Only replaces words that have synonyms available.
    Preserves punctuation, capitalization patterns, and structure.
    """
    words = text.split()
    if len(words) < 3:
        return text

    n_replace = max(1, int(len(words) * replace_ratio))

    # Find candidate positions (skip very short words, numbers, emails)
    candidates = []
    for i, word in enumerate(words):
        clean = re.sub(r'[^\w]', '', word)
        if len(clean) > 3 and clean.isalpha():
            candidates.append(i)

    if not candidates:
        return text

    # Select positions to replace
    positions = random.sample(candidates, min(n_replace, len(candidates)))

    new_words = words.copy()
    for pos in positions:
        original = words[pos]
        clean = re.sub(r'[^\w]', '', original)
        syns = get_synonyms(clean.lower())
        if syns:
            replacement = random.choice(syns)
            # Preserve leading/trailing punctuation
            prefix = ""
            suffix = ""
            if original and not original[0].isalpha():
                prefix = original[0]
            if original and not original[-1].isalpha():
                suffix = original[-1]
            new_words[pos] = prefix + replacement + suffix

    return " ".join(new_words)


def augment_text(text, n_variants=1):
    """Generate n synonym-replaced variants of a text."""
    variants = []
    for _ in range(n_variants):
        ratio = random.uniform(REPLACE_RATIO_MIN, REPLACE_RATIO_MAX)
        aug = synonym_replace(text, replace_ratio=ratio)
        # Only keep if it's actually different
        if aug != text:
            variants.append(aug)
        else:
            # Retry with higher ratio
            aug = synonym_replace(text, replace_ratio=min(ratio + 0.1, 0.5))
            variants.append(aug)
    return variants


def main():
    print("=" * 65)
    print("  B-3c TASK 1: RSG DATA AUGMENTATION")
    print("=" * 65)

    # Load original data
    print(f"\n[1/4] Loading RSG data from: {RSG_CSV}")
    df = pd.read_csv(str(RSG_CSV)).dropna()
    print(f"  Original samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Analyze class distribution
    print(f"\n[2/4] Analyzing class distribution...")
    class_counts = df["template_index"].value_counts().sort_index()
    total_classes = len(class_counts)
    minority_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]

    print(f"  Total classes:       {total_classes}")
    print(f"  Min samples/class:   {class_counts.min()}")
    print(f"  Max samples/class:   {class_counts.max()}")
    print(f"  Minority classes (n < {MIN_SAMPLES_PER_CLASS}): {len(minority_classes)}")

    print(f"\n  Minority class breakdown:")
    for cls_id, count in minority_classes.items():
        need = MIN_SAMPLES_PER_CLASS - count
        print(f"    Class {cls_id:>3d}: {count:>3d} samples, need {need:>3d} more")

    # Generate synthetic samples
    print(f"\n[3/4] Generating synthetic samples...")
    synthetic_rows = []
    total_generated = 0

    for cls_id, current_count in minority_classes.items():
        need = MIN_SAMPLES_PER_CLASS - current_count
        class_texts = df[df["template_index"] == cls_id]["profile_text"].values

        generated = 0
        attempt = 0
        max_attempts = need * 5  # Safety limit

        while generated < need and attempt < max_attempts:
            # Pick a random source text from this class
            source = random.choice(class_texts)
            variants = augment_text(source, n_variants=1)
            for v in variants:
                if generated < need:
                    synthetic_rows.append({
                        "profile_text": v,
                        "template_index": cls_id
                    })
                    generated += 1
            attempt += 1

        total_generated += generated
        print(f"    Class {cls_id:>3d}: generated {generated}/{need} synthetic samples")

    print(f"\n  Total synthetic samples generated: {total_generated}")

    # Merge original + synthetic
    print(f"\n[4/4] Merging and saving balanced dataset...")
    synthetic_df = pd.DataFrame(synthetic_rows)
    balanced_df = pd.concat([df[["profile_text", "template_index"]], synthetic_df],
                            ignore_index=True)

    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Verify balance
    new_counts = balanced_df["template_index"].value_counts().sort_index()
    still_minority = new_counts[new_counts < MIN_SAMPLES_PER_CLASS]

    print(f"  Original samples:  {len(df)}")
    print(f"  Synthetic samples: {len(synthetic_df)}")
    print(f"  Balanced total:    {len(balanced_df)}")
    print(f"  Min samples/class: {new_counts.min()}")
    print(f"  Max samples/class: {new_counts.max()}")

    if len(still_minority) > 0:
        print(f"\n  [WARN] {len(still_minority)} classes still below {MIN_SAMPLES_PER_CLASS}")
    else:
        print(f"\n  [OK] All classes have >= {MIN_SAMPLES_PER_CLASS} samples")

    # Save
    balanced_df.to_csv(str(OUTPUT_CSV), index=False)
    print(f"\n  Saved: {OUTPUT_CSV}")
    print(f"  File size: {OUTPUT_CSV.stat().st_size / 1e6:.1f} MB")

    # Print full distribution
    print(f"\n  --- Balanced Class Distribution ---")
    for cls_id in sorted(new_counts.index):
        orig = class_counts.get(cls_id, 0)
        new = new_counts[cls_id]
        aug = new - orig
        marker = " (+{})".format(aug) if aug > 0 else ""
        print(f"    Class {cls_id:>3d}: {new:>3d} samples{marker}")

    print("\n" + "=" * 65)
    print("  AUGMENTATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
