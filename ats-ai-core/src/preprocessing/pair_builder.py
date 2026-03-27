"""
T-08 · src/preprocessing/pair_builder.py

Builds resume–JD training pairs from the unpaired LiveCareer resume
dataset (DS-1) and LinkedIn job postings dataset (DS-2).

Pairing strategy:
  - 70% same-domain pairs  (positive signal for domain classifier)
  - 30% cross-domain pairs (hard negative signal for score head)
  - Max 2 000 pairs per domain to prevent class imbalance
  - Each resume paired with at most 3 JDs to avoid data leakage
"""

import logging
import random
from pathlib import Path

import pandas as pd

from src.config import LABELED_DIR, NUM_DOMAINS, RANDOM_SEED
from src.preprocessing.domain_mapper import map_text
from src.preprocessing.normalizer import normalize_text
from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)

MAX_PAIRS_PER_DOMAIN: int = 2000
MAX_JDS_PER_RESUME: int = 3
SAME_DOMAIN_RATIO: float = 0.70


def _assign_domain(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add a 'domain_index' column by running domain_mapper over *text_col*.

    Args:
        df: DataFrame containing the text column.
        text_col: Column name holding the text to classify.

    Returns:
        DataFrame with 'domain_index' column added/overwritten.
    """
    df = df.copy()
    df["domain_index"] = df[text_col].apply(map_text)
    return df


def build_pairs(
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create labeled resume–JD pairs from unpaired resume and job datasets.

    Weak ATS scores are generated using TF-IDF cosine similarity between
    resume and JD text. Domain classification labels come from domain_mapper.

    Args:
        resumes_df: DataFrame with at least 'resume_text' column.
        jobs_df: DataFrame with at least 'jd_text' column.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            resume_text, jd_text, score (0–100), domain_index, label_source
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415

    random.seed(seed)
    logger.info("Building training pairs from %d resumes and %d jobs",
                len(resumes_df), len(jobs_df))

    # ── Clean and normalise ──────────────────────────────────────────────────
    resumes = resumes_df.copy()
    jobs = jobs_df.copy()
    resumes["resume_text"] = resumes["resume_text"].apply(
        lambda t: normalize_text(clean_text(str(t)))
    )
    jobs["jd_text"] = jobs["jd_text"].apply(
        lambda t: normalize_text(clean_text(str(t)))
    )

    # ── Tag domains ─────────────────────────────────────────────────────────
    if "domain_index" not in resumes.columns or (resumes["domain_index"] == -1).all():
        resumes = _assign_domain(resumes, "resume_text")
    if "domain_index" not in jobs.columns or (jobs["domain_index"] == -1).all():
        jobs = _assign_domain(jobs, "jd_text")

    # Drop rows with unknown domain (index -1)
    resumes = resumes[resumes["domain_index"] >= 0].reset_index(drop=True)
    jobs    = jobs[jobs["domain_index"] >= 0].reset_index(drop=True)

    # ── Build TF-IDF vocab over all JDs (fit once, no data leakage risk
    #    here because TF-IDF is only used to generate weak labels, not
    #    as a model feature at inference time) ────────────────────────────
    logger.info("Fitting TF-IDF vectoriser over %d JD texts …", len(jobs))
    vectoriser = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    jd_matrix = vectoriser.fit_transform(jobs["jd_text"].tolist())

    # ── Pairing loop ─────────────────────────────────────────────────────────
    pair_records: list[dict] = []
    domain_counts: dict[int, int] = {d: 0 for d in range(NUM_DOMAINS)}

    domain_to_jds = {d: jobs.index[jobs["domain_index"] == d].tolist() for d in range(NUM_DOMAINS)}
    domain_to_cross_jds = {d: [idx for od, indices in domain_to_jds.items() if od != d for idx in indices] for d in range(NUM_DOMAINS)}

    for _, resume_row in resumes.iterrows():
        r_domain = int(resume_row["domain_index"])
        r_text = str(resume_row["resume_text"])

        # Encode resume
        r_vec = vectoriser.transform([r_text])

        # Candidate JDs: same-domain + random cross-domain
        same_domain_idx = domain_to_jds.get(r_domain, [])
        cross_domain_idx = domain_to_cross_jds.get(r_domain, [])

        n_same  = round(MAX_JDS_PER_RESUME * SAME_DOMAIN_RATIO)
        n_cross = MAX_JDS_PER_RESUME - n_same

        selected_same  = random.sample(same_domain_idx, min(n_same, len(same_domain_idx)))
        selected_cross = random.sample(cross_domain_idx, min(n_cross, len(cross_domain_idx)))
        selected = selected_same + selected_cross

        for j_idx in selected:
            j_row   = jobs.loc[j_idx]
            j_domain = int(j_row["domain_index"])

            # Cap per domain
            if domain_counts.get(j_domain, 0) >= MAX_PAIRS_PER_DOMAIN:
                continue

            # Weak score: cosine similarity scaled to 0–100
            j_vec = jd_matrix[j_idx]
            sim = float(cosine_similarity(r_vec, j_vec)[0][0])
            weak_score = round(sim * 100, 2)

            pair_records.append({
                "resume_text": r_text,
                "jd_text": str(j_row["jd_text"]),
                "score": weak_score,
                "domain_index": j_domain,
                "label_source": "weak",
            })
            domain_counts[j_domain] = domain_counts.get(j_domain, 0) + 1

    pairs_df = pd.DataFrame(pair_records)
    logger.info("Built %d weak-labeled pairs across %d domains", len(pairs_df), NUM_DOMAINS)
    return pairs_df


def merge_with_gold(
    weak_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weak-labeled pairs with gold-labeled pairs from DS-3.

    Gold labels always take precedence. Near-duplicate pairs (same
    resume + JD prefix) are deduplicated in favour of the gold label.

    Args:
        weak_df: DataFrame of weak-labeled pairs (label_source='weak').
        gold_df: DataFrame of gold-labeled pairs (label_source='gold').

    Returns:
        Combined DataFrame, gold rows first.
    """
    combined = pd.concat([gold_df, weak_df], ignore_index=True)
    # Dedup: keep first occurrence (gold rows are first)
    combined["_key"] = (
        combined["resume_text"].str[:80] + "||" + combined["jd_text"].str[:80]
    )
    combined = combined.drop_duplicates(subset=["_key"], keep="first")
    combined = combined.drop(columns=["_key"])
    logger.info("Merged dataset: %d total pairs (%d gold, %d weak)",
                len(combined),
                (combined["label_source"] == "gold").sum(),
                (combined["label_source"] == "weak").sum())
    return combined.reset_index(drop=True)


def save_training_pairs(df: pd.DataFrame, path: Path = LABELED_DIR / "training_pairs.csv") -> None:
    """Save the final training pairs DataFrame to CSV.

    Args:
        df: Training pairs DataFrame.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d training pairs to %s", len(df), path)
