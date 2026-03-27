"""
T-01 · src/preprocessing/data_loader.py

Loaders for all three ATS training datasets. Each loader returns a
DataFrame with a unified schema so downstream modules never need to
know which source the data came from.

Unified schema columns:
    resume_text  : str   — cleaned resume body text
    jd_text      : str   — job description body text
    score        : float — ATS score 0–100 (NaN for unpaired rows)
    domain_index : int   — domain label 0–6 (-1 if unknown)
    label_source : str   — "gold" | "weak" | "unpaired_resume" | "unpaired_jd"
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    LINKEDIN_CSV,
    LIVECARER_CATEGORY_MAP,
    RESUME_CSV,
    RESUME_SCORE_DIR,
)

logger = logging.getLogger(__name__)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _coerce_str(series: pd.Series) -> pd.Series:
    """Cast to string and strip whitespace; replace nulls with empty string."""
    return series.fillna("").astype(str).str.strip()


def _drop_short_rows(df: pd.DataFrame, col: str, min_chars: int = 50) -> pd.DataFrame:
    """Drop rows where *col* has fewer than *min_chars* characters."""
    mask = df[col].str.len() >= min_chars
    dropped = (~mask).sum()
    if dropped:
        logger.warning("Dropped %d rows with short '%s' (< %d chars)", dropped, col, min_chars)
    return df[mask].reset_index(drop=True)


# ── Dataset 1: LiveCareer Resume CSV ─────────────────────────────────────────

def load_resume_dataset(path: Path = RESUME_CSV) -> pd.DataFrame:
    """Load the LiveCareer resume dataset (Kaggle: snehaanbhawal/resume-dataset).

    Expected CSV columns: ID, Resume_str, Resume_html, Category.

    Args:
        path: Path to Resume.csv.

    Returns:
        DataFrame with unified schema. ``score`` and ``jd_text`` are NaN/empty
        because this dataset has no paired JDs.
    """
    logger.info("Loading LiveCareer resume dataset from %s", path)
    df = pd.read_csv(path)

    required = {"Resume_str", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LiveCareer CSV missing expected columns: {missing}")

    out = pd.DataFrame()
    out["resume_text"] = _coerce_str(df["Resume_str"])
    out["jd_text"] = ""
    out["score"] = float("nan")
    out["domain_index"] = (
        df["Category"]
        .fillna("")
        .str.strip()
        .map(lambda c: LIVECARER_CATEGORY_MAP.get(c, -1))
        .astype(int)
    )
    out["label_source"] = "unpaired_resume"

    out = _drop_short_rows(out, "resume_text", min_chars=100)
    logger.info("LiveCareer: %d resumes loaded, %d with known domain",
                len(out), (out["domain_index"] >= 0).sum())
    return out


# ── Dataset 2: LinkedIn Job Postings CSV ─────────────────────────────────────

def load_linkedin_dataset(path: Path = LINKEDIN_CSV) -> pd.DataFrame:
    """Load the LinkedIn job postings dataset (Kaggle: arshkon/linkedin-job-postings).

    Expected CSV columns include: job_id, title, description, skills_desc,
    formatted_experience_level (optional).

    Args:
        path: Path to job_postings.csv.

    Returns:
        DataFrame with unified schema. ``resume_text`` and ``score`` are empty/NaN.
    """
    logger.info("Loading LinkedIn job postings from %s", path)

    # Large file — only load needed columns to save RAM
    usecols = ["job_id", "title", "description", "skills_desc"]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)

    if "description" not in df.columns:
        raise ValueError("LinkedIn CSV missing 'description' column")

    # Combine description + skills_desc for richer JD text
    desc = _coerce_str(df["description"])
    skills = _coerce_str(df.get("skills_desc", pd.Series([""] * len(df))))
    combined = (desc + " " + skills).str.strip()

    out = pd.DataFrame()
    out["resume_text"] = ""
    out["jd_text"] = combined
    out["score"] = float("nan")
    out["domain_index"] = -1  # domain tagged later by domain_mapper
    out["label_source"] = "unpaired_jd"

    out = _drop_short_rows(out, "jd_text", min_chars=100)
    logger.info("LinkedIn: %d job postings loaded", len(out))
    return out


# ── Dataset 3: HuggingFace Resume Score Details ───────────────────────────────

def load_resume_score_dataset(directory: Path = RESUME_SCORE_DIR) -> pd.DataFrame:
    """Load the netsol/resume-score-details dataset.

    Each JSON file in *directory* contains a single record with structure::

        {
          "input":  {"resume": "...", "job_description": "...", ...},
          "output": {"scores": {"aggregated_scores": {"macro_scores": 7.7, "micro_scores": 7.3}}},
          "details": ...
        }

    Scores are on a 0–10 scale and converted to 0–100 for the ATS pipeline.

    Args:
        directory: Path to directory of individual JSON files.

    Returns:
        DataFrame with unified schema and label_source = "gold".
    """
    import json as _json

    logger.info("Loading Resume Score Details from %s", directory)
    json_files = sorted(Path(directory).glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No .json files found in {directory}. "
            "Download the dataset from https://huggingface.co/datasets/netsol/resume-score-details"
        )

    records: list[dict] = []
    skipped = 0
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            inp = data.get("input", {})
            out = data.get("output", {})

            resume_text = str(inp.get("resume", ""))
            jd_text = str(inp.get("job_description", ""))

            # Extract aggregated score (0–10 scale) → convert to 0–100
            agg = out.get("scores", {}).get("aggregated_scores", {})
            macro = agg.get("macro_scores", None)
            micro = agg.get("micro_scores", None)
            if macro is not None and micro is not None:
                score = ((float(macro) + float(micro)) / 2.0) * 10.0
            elif macro is not None:
                score = float(macro) * 10.0
            elif micro is not None:
                score = float(micro) * 10.0
            else:
                skipped += 1
                continue

            records.append({
                "resume_text": resume_text,
                "jd_text": jd_text,
                "score": round(min(max(score, 0), 100), 2),
                "domain_index": -1,
                "label_source": "gold",
            })
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", fpath.name, exc)
            skipped += 1

    if skipped:
        logger.warning("Skipped %d files during loading", skipped)

    raw = pd.DataFrame(records)
    out = pd.DataFrame()
    out["resume_text"] = _coerce_str(raw["resume_text"])
    out["jd_text"] = _coerce_str(raw["jd_text"])
    out["score"] = pd.to_numeric(raw["score"], errors="coerce").clip(0, 100)
    out["domain_index"] = raw["domain_index"].astype(int)
    out["label_source"] = raw["label_source"]

    # Drop rows with null score
    before = len(out)
    out = out.dropna(subset=["score"]).reset_index(drop=True)
    if len(out) < before:
        logger.warning("Dropped %d rows with null score from gold dataset", before - len(out))

    out = _drop_short_rows(out, "resume_text", min_chars=100)
    out = _drop_short_rows(out, "jd_text", min_chars=50)

    logger.info("Resume Score Details: %d labeled pairs loaded", len(out))
    return out


# ── Combined loader ───────────────────────────────────────────────────────────

def load_all_datasets(
    resume_path: Path = RESUME_CSV,
    linkedin_path: Path = LINKEDIN_CSV,
    score_dir: Path = RESUME_SCORE_DIR,
) -> dict[str, pd.DataFrame]:
    """Load all three datasets and return them in a named dictionary.

    Args:
        resume_path: Path to LiveCareer Resume.csv.
        linkedin_path: Path to LinkedIn job_postings.csv.
        score_dir: Directory containing HuggingFace parquet/jsonl files.

    Returns:
        Dictionary with keys "resumes", "jobs", "scored_pairs".
    """
    return {
        "resumes": load_resume_dataset(resume_path),
        "jobs": load_linkedin_dataset(linkedin_path),
        "scored_pairs": load_resume_score_dataset(score_dir),
    }
