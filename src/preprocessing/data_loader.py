"""
data_loader.py — Loaders for all three ATS datasets.

Provides:
    - load_livecareer_resumes()  → DS-1 LiveCareer resume CSV
    - load_linkedin_jobs()       → DS-2 LinkedIn job postings CSV
    - load_resume_scores()       → DS-3 HuggingFace resume-score-details
    - load_all()                 → Convenience wrapper returning all three DataFrames

Each loader returns a clean pandas DataFrame. Malformed rows, null-heavy
records, and encoding issues are handled internally.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────

# Check for nested Resume folder (common when unzipping)
_LIVECAREER_BASE = RAW_DIR / "resume_dataset"
if (_LIVECAREER_BASE / "Resume" / "Resume.csv").exists():
    _LIVECAREER_DEFAULT_PATH: Path = _LIVECAREER_BASE / "Resume" / "Resume.csv"
else:
    _LIVECAREER_DEFAULT_PATH: Path = _LIVECAREER_BASE / "Resume.csv"
_LINKEDIN_DEFAULT_PATH: Path = RAW_DIR / "linkedin_jobs" / "job_postings.csv"
_RESUME_SCORES_DEFAULT_DIR: Path = RAW_DIR / "resume_score_details"

# Minimum text length to keep a record (very short texts are noise)
_MIN_TEXT_LENGTH: int = 50

# Maximum fraction of nulls allowed in a row before dropping it
_MAX_NULL_FRACTION: float = 0.5


# ────────────────────────────────────────────
# DS-1: LiveCareer Resume CSV
# ────────────────────────────────────────────

def load_livecareer_resumes(
    path: Optional[Path] = None,
    *,
    drop_html: bool = True,
) -> pd.DataFrame:
    """Load the LiveCareer resume dataset (DS-1).

    Args:
        path: Path to the Resume.csv file. Uses default from config if None.
        drop_html: If True, drops the Resume_html column to save memory.

    Returns:
        DataFrame with columns:
            - resume_text (str): Raw resume text
            - category (str): Original LiveCareer category label
            - resume_id (int): Unique resume identifier

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        ValueError: If required columns are missing from the CSV.
    """
    csv_path = path or _LIVECAREER_DEFAULT_PATH
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"LiveCareer resume CSV not found at: {csv_path}\n"
            f"Download from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset"
        )

    logger.info("Loading LiveCareer resumes from %s", csv_path)

    # Read with multiple encodings as fallback
    df = _read_csv_safe(csv_path)

    # Validate required columns
    required_cols = {"ID", "Resume_str", "Category"}
    _validate_columns(df, required_cols, "LiveCareer")

    # Rename to unified schema
    df = df.rename(columns={
        "ID": "resume_id",
        "Resume_str": "resume_text",
        "Category": "category",
    })

    # Drop HTML column if requested
    if drop_html and "Resume_html" in df.columns:
        df = df.drop(columns=["Resume_html"])

    # Clean up
    df = _drop_high_null_rows(df)
    df["resume_text"] = df["resume_text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # Filter out rows with extremely short text
    original_len = len(df)
    df = df[df["resume_text"].str.len() >= _MIN_TEXT_LENGTH].copy()
    dropped = original_len - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with resume text shorter than %d chars",
                     dropped, _MIN_TEXT_LENGTH)

    # Reset index
    df = df.reset_index(drop=True)

    logger.info("Loaded %d resumes across %d categories",
                 len(df), df["category"].nunique())

    return df


# ────────────────────────────────────────────
# DS-2: LinkedIn Job Postings CSV
# ────────────────────────────────────────────

def load_linkedin_jobs(
    path: Optional[Path] = None,
    *,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the LinkedIn job postings dataset (DS-2).

    Args:
        path: Path to the job_postings.csv file. Uses default from config if None.
        max_rows: If set, limits the number of rows loaded (useful for dev).

    Returns:
        DataFrame with columns:
            - job_id (int/str): Unique job posting identifier
            - title (str): Job title
            - jd_text (str): Job description text
            - location (str): Job location
            - company_name (str): Employer name
            - skills_desc (str): Skills description (if available)
            - experience_level (str): Required experience level (if available)

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        ValueError: If required columns are missing from the CSV.
    """
    csv_path = path or _LINKEDIN_DEFAULT_PATH
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"LinkedIn job postings CSV not found at: {csv_path}\n"
            f"Download from: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings"
        )

    logger.info("Loading LinkedIn job postings from %s", csv_path)

    df = _read_csv_safe(csv_path, nrows=max_rows)

    # Validate required columns
    required_cols = {"description"}
    _validate_columns(df, required_cols, "LinkedIn")

    # Standardize column names
    rename_map: dict[str, str] = {
        "description": "jd_text",
    }

    # Optional columns — only rename if present
    optional_renames: dict[str, str] = {
        "formatted_experience_level": "experience_level",
        "skills_desc": "skills_desc",
        "job_id": "job_id",
        "title": "title",
        "location": "location",
        "company_name": "company_name",
    }
    for old, new in optional_renames.items():
        if old in df.columns:
            rename_map[old] = new

    df = df.rename(columns=rename_map)

    # Clean up
    df = _drop_high_null_rows(df)
    df["jd_text"] = df["jd_text"].astype(str).str.strip()

    # Filter out rows with extremely short JD text
    original_len = len(df)
    df = df[df["jd_text"].str.len() >= _MIN_TEXT_LENGTH].copy()
    dropped = original_len - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with JD text shorter than %d chars",
                     dropped, _MIN_TEXT_LENGTH)

    # Keep only the columns we use
    keep_cols = [
        c for c in [
            "job_id", "title", "jd_text", "location",
            "company_name", "skills_desc", "experience_level",
        ]
        if c in df.columns
    ]
    df = df[keep_cols].reset_index(drop=True)

    logger.info("Loaded %d job postings", len(df))

    return df


# ────────────────────────────────────────────
# DS-3: HuggingFace Resume Score Details
# ────────────────────────────────────────────

def load_resume_scores(
    directory: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the resume-score-details dataset (DS-3).

    Supports both Parquet and JSON formats. Scans the directory for any
    .parquet or .json files and concatenates them.

    Args:
        directory: Directory containing the dataset files. Uses default
                   from config if None.

    Returns:
        DataFrame with columns:
            - resume_text (str): Resume text
            - jd_text (str): Job description text
            - score (float): ATS match score 0–100
            - missing_keywords (str): Comma-separated missing keywords (if available)
            - feedback (str): Feedback text (if available)

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no valid data files are found or required columns missing.
    """
    data_dir = directory or _RESUME_SCORES_DEFAULT_DIR
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Resume score details directory not found at: {data_dir}\n"
            f"Download from: https://huggingface.co/datasets/netsol/resume-score-details"
        )

    logger.info("Loading resume score details from %s", data_dir)

    frames: list[pd.DataFrame] = []

    # Load parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    for pf in parquet_files:
        logger.info("  Reading parquet: %s", pf.name)
        frames.append(pd.read_parquet(pf))

    # Load JSON files (JSON Lines or regular JSON)
    json_files = sorted(data_dir.glob("*.json"))
    for jf in json_files:
        logger.info("  Reading JSON: %s", jf.name)
        df_json = _read_json_safe(jf)
        frames.append(df_json)

    # Load JSONL files
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    for jf in jsonl_files:
        logger.info("  Reading JSONL: %s", jf.name)
        frames.append(pd.read_json(jf, lines=True))

    # Load CSV files as fallback
    csv_files = sorted(data_dir.glob("*.csv"))
    for cf in csv_files:
        logger.info("  Reading CSV: %s", cf.name)
        frames.append(_read_csv_safe(cf))

    if not frames:
        raise ValueError(
            f"No .parquet, .json, .jsonl, or .csv files found in: {data_dir}"
        )

    df = pd.concat(frames, ignore_index=True)

    # Standardize column names (the HF dataset may use various names)
    col_mapping: dict[str, str] = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in ("resume_text", "resume", "resume_str"):
            col_mapping[col] = "resume_text"
        elif col_lower in ("job_description", "jd_text", "jd", "description"):
            col_mapping[col] = "jd_text"
        elif col_lower in ("score", "ats_score", "match_score"):
            col_mapping[col] = "score"
        elif col_lower in ("missing_keywords", "keywords"):
            col_mapping[col] = "missing_keywords"
        elif col_lower in ("feedback", "feedback_text"):
            col_mapping[col] = "feedback"

    df = df.rename(columns=col_mapping)

    # Validate core columns
    required_cols = {"resume_text", "jd_text", "score"}
    _validate_columns(df, required_cols, "Resume Score Details")

    # Clean up
    df = _drop_high_null_rows(df)
    df["resume_text"] = df["resume_text"].astype(str).str.strip()
    df["jd_text"] = df["jd_text"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Drop rows where score is null after conversion
    null_scores = df["score"].isna().sum()
    if null_scores > 0:
        logger.warning("Dropping %d rows with non-numeric scores", null_scores)
        df = df.dropna(subset=["score"])

    # Clamp score to 0–100 range
    df["score"] = df["score"].clip(0.0, 100.0)

    # Filter out very short texts
    original_len = len(df)
    df = df[
        (df["resume_text"].str.len() >= _MIN_TEXT_LENGTH)
        & (df["jd_text"].str.len() >= _MIN_TEXT_LENGTH)
    ].copy()
    dropped = original_len - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with text shorter than %d chars",
                     dropped, _MIN_TEXT_LENGTH)

    # Keep relevant columns
    keep_cols = [
        c for c in ["resume_text", "jd_text", "score", "missing_keywords", "feedback"]
        if c in df.columns
    ]
    df = df[keep_cols].reset_index(drop=True)

    logger.info("Loaded %d labeled resume–JD pairs (score range: %.1f – %.1f)",
                 len(df), df["score"].min(), df["score"].max())

    return df


# ────────────────────────────────────────────
# Convenience: load all three datasets
# ────────────────────────────────────────────

def load_all(
    *,
    livecareer_path: Optional[Path] = None,
    linkedin_path: Optional[Path] = None,
    resume_scores_dir: Optional[Path] = None,
    linkedin_max_rows: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Load all three datasets and return them as a dict.

    Args:
        livecareer_path: Override path for DS-1.
        linkedin_path: Override path for DS-2.
        resume_scores_dir: Override directory for DS-3.
        linkedin_max_rows: Cap LinkedIn rows for dev/testing.

    Returns:
        Dict with keys: 'livecareer', 'linkedin', 'resume_scores'
        Each value is a cleaned pandas DataFrame.
    """
    return {
        "livecareer": load_livecareer_resumes(path=livecareer_path),
        "linkedin": load_linkedin_jobs(path=linkedin_path, max_rows=linkedin_max_rows),
        "resume_scores": load_resume_scores(directory=resume_scores_dir),
    }


# ────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────

def _read_csv_safe(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read a CSV with encoding fallbacks.

    Tries utf-8 first, then latin-1, then cp1252. Handles common
    CSV quirks like mixed line endings.

    Args:
        path: Path to the CSV file.
        nrows: Optional row limit.

    Returns:
        Parsed DataFrame.
    """
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_err: UnicodeDecodeError | UnicodeError | None = None

    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                nrows=nrows,
                on_bad_lines="skip",
                engine="python",
            )
        except (UnicodeDecodeError, UnicodeError) as exc:
            logger.debug("Encoding %s failed for %s: %s", enc, path, exc)
            last_err = exc
            continue

    raise ValueError(
        f"Could not read CSV at {path} with any supported encoding"
    ) from last_err


def _read_json_safe(path: Path) -> pd.DataFrame:
    """Read a JSON file, handling both JSONL and standard JSON arrays.

    Pandas 3.x's read_json(lines=True) silently produces integer column
    names when given a regular JSON array instead of raising an error.
    This helper detects that situation and falls back appropriately.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed DataFrame with string column names.
    """
    # Try JSONL format first
    try:
        df = pd.read_json(path, lines=True)
        # Detect if pandas produced garbage columns (integer names)
        if len(df.columns) > 0 and all(isinstance(c, int) for c in df.columns):
            logger.debug("JSONL parse produced int columns for %s, retrying as JSON array", path)
            df = pd.read_json(path)
        return df
    except (ValueError, TypeError):
        return pd.read_json(path)


def _validate_columns(
    df: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: DataFrame to validate.
        required: Set of required column names.
        dataset_name: Human-readable name for error messages.

    Raises:
        ValueError: If any required columns are missing.
    """
    # Case-insensitive check
    actual_lower = {str(c).lower().strip() for c in df.columns}
    missing = {r for r in required if r.lower() not in actual_lower}

    if missing:
        raise ValueError(
            f"{dataset_name} dataset is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def _drop_high_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where more than MAX_NULL_FRACTION of columns are null.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame with high-null rows removed.
    """
    threshold = int(len(df.columns) * (1 - _MAX_NULL_FRACTION))
    original_len = len(df)
    df = df.dropna(thresh=threshold)
    dropped = original_len - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows exceeding %.0f%% null threshold",
                     dropped, _MAX_NULL_FRACTION * 100)
    return df
