"""
test_data_loader.py — Unit tests for src/preprocessing/data_loader.py

Tests cover:
    - LiveCareer resume loader (DS-1)
    - LinkedIn job postings loader (DS-2)
    - HuggingFace resume score details loader (DS-3)
    - Edge cases: missing files, missing columns, encoding issues
    - Helper functions
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.data_loader import (
    _drop_high_null_rows,
    _read_csv_safe,
    _read_json_safe,
    _validate_columns,
    load_all,
    load_livecareer_resumes,
    load_linkedin_jobs,
    load_resume_scores,
)

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


# ────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────

@pytest.fixture()
def livecareer_csv(tmp_path: Path) -> Path:
    """Create a minimal LiveCareer-style CSV."""
    csv_path = tmp_path / "Resume.csv"
    data = {
        "ID": [1, 2, 3, 4],
        "Resume_str": [
            "Experienced Python developer with 5 years building web applications using Django and Flask frameworks.",
            "Marketing manager with expertise in digital campaigns, SEO optimization, and team leadership.",
            "x",  # Too short — should be dropped
            "Graphic designer skilled in Adobe Photoshop, Illustrator, Figma, and UI/UX design principles.",
        ],
        "Resume_html": [
            "<html><body>Developer resume</body></html>",
            "<html><body>Marketing resume</body></html>",
            "<html><body>Short</body></html>",
            "<html><body>Designer resume</body></html>",
        ],
        "Category": [
            "Information-Technology",
            "Business-Development",
            "Other",
            "Designer",
        ],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


@pytest.fixture()
def linkedin_csv(tmp_path: Path) -> Path:
    """Create a minimal LinkedIn job postings CSV."""
    csv_path = tmp_path / "job_postings.csv"
    data = {
        "job_id": [101, 102, 103],
        "title": ["Software Engineer", "Product Manager", "Data Analyst"],
        "description": [
            "We are looking for a software engineer with strong Python skills, experience in cloud platforms, and CI/CD pipelines.",
            "Seeking a product manager who can drive strategy, manage stakeholders, and lead cross-functional teams.",
            "short",  # Too short — should be dropped
        ],
        "location": ["New York", "San Francisco", "Remote"],
        "company_name": ["TechCorp", "StartupInc", "DataCo"],
        "skills_desc": ["Python, AWS, Docker", "Leadership, Agile", "SQL, Excel"],
        "formatted_experience_level": ["Mid-Senior level", "Director", "Entry level"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


@pytest.fixture()
def resume_scores_dir(tmp_path: Path) -> Path:
    """Create a directory with a minimal resume-score-details JSON file."""
    scores_dir = tmp_path / "resume_score_details"
    scores_dir.mkdir()
    data = [
        {
            "resume_text": "Backend developer with expertise in Python, Django, REST APIs, and PostgreSQL database management.",
            "job_description": "Looking for a backend engineer with Python experience, REST API design, and database optimization skills.",
            "score": 78.5,
            "missing_keywords": "kubernetes, docker",
            "feedback": "Good match but missing containerization skills.",
        },
        {
            "resume_text": "Data scientist with experience in machine learning, TensorFlow, and statistical analysis using NumPy and pandas.",
            "job_description": "We need a data scientist proficient in deep learning, NLP, and deploying ML models into production environments.",
            "score": 62.0,
            "missing_keywords": "NLP, deployment",
            "feedback": "Moderate match — add NLP and deployment experience.",
        },
        {
            "resume_text": "Entry level graduate with coursework in computer science, algorithms, and data structures fundamentals.",
            "job_description": "Senior architect role requiring 10+ years in distributed systems, microservices, and cloud-native architectures.",
            "score": 15.0,
            "missing_keywords": "distributed systems, microservices, cloud",
            "feedback": "Poor match — significant experience gap.",
        },
    ]
    json_path = scores_dir / "scores.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return scores_dir


@pytest.fixture()
def resume_scores_parquet_dir(tmp_path: Path) -> Path:
    """Create a directory with a minimal resume-score-details Parquet file."""
    pytest.importorskip("pyarrow")
    scores_dir = tmp_path / "resume_score_details_parquet"
    scores_dir.mkdir()
    df = pd.DataFrame({
        "resume_text": [
            "Experienced frontend developer specializing in React, TypeScript, and responsive web design patterns.",
            "Product designer with portfolio showcasing user research, wireframing, and high-fidelity prototyping.",
        ],
        "job_description": [
            "Frontend engineer needed for React, TypeScript. Experience with design systems and component libraries preferred.",
            "UX designer role requiring Figma expertise, user research skills, and experience with design thinking methodology.",
        ],
        "score": [85.0, 70.0],
    })
    parquet_path = scores_dir / "scores.parquet"
    df.to_parquet(parquet_path, index=False)
    return scores_dir


# ────────────────────────────────────────────
# Tests: LiveCareer Loader (DS-1)
# ────────────────────────────────────────────

class TestLoadLivecareerResumes:
    """Tests for load_livecareer_resumes()."""

    def test_loads_valid_csv(self, livecareer_csv: Path) -> None:
        """Should load and return a DataFrame with expected columns."""
        df = load_livecareer_resumes(path=livecareer_csv)
        assert isinstance(df, pd.DataFrame)
        assert "resume_text" in df.columns
        assert "category" in df.columns
        assert "resume_id" in df.columns

    def test_drops_short_resumes(self, livecareer_csv: Path) -> None:
        """Should drop resumes shorter than MIN_TEXT_LENGTH."""
        df = load_livecareer_resumes(path=livecareer_csv)
        # "x" is too short → should be dropped, leaving 3 rows
        assert len(df) == 3

    def test_drops_html_column(self, livecareer_csv: Path) -> None:
        """HTML column should be dropped by default."""
        df = load_livecareer_resumes(path=livecareer_csv, drop_html=True)
        assert "Resume_html" not in df.columns

    def test_keeps_html_column(self, livecareer_csv: Path) -> None:
        """HTML column should be kept when drop_html=False."""
        df = load_livecareer_resumes(path=livecareer_csv, drop_html=False)
        assert "Resume_html" in df.columns

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="LiveCareer"):
            load_livecareer_resumes(path=tmp_path / "nonexistent.csv")

    def test_missing_columns(self, tmp_path: Path) -> None:
        """Should raise ValueError if required columns are missing."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"col_a": [1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_livecareer_resumes(path=bad_csv)

    def test_whitespace_stripped(self, livecareer_csv: Path) -> None:
        """Resume text and category should be stripped of whitespace."""
        df = load_livecareer_resumes(path=livecareer_csv)
        for text in df["resume_text"]:
            assert text == text.strip()
        for cat in df["category"]:
            assert cat == cat.strip()


# ────────────────────────────────────────────
# Tests: LinkedIn Loader (DS-2)
# ────────────────────────────────────────────

class TestLoadLinkedinJobs:
    """Tests for load_linkedin_jobs()."""

    def test_loads_valid_csv(self, linkedin_csv: Path) -> None:
        """Should load and return a DataFrame with expected columns."""
        df = load_linkedin_jobs(path=linkedin_csv)
        assert isinstance(df, pd.DataFrame)
        assert "jd_text" in df.columns
        assert "title" in df.columns

    def test_drops_short_descriptions(self, linkedin_csv: Path) -> None:
        """Should drop JDs shorter than MIN_TEXT_LENGTH."""
        df = load_linkedin_jobs(path=linkedin_csv)
        # "short" is too short → should be dropped, leaving 2 rows
        assert len(df) == 2

    def test_renames_columns(self, linkedin_csv: Path) -> None:
        """Should rename description → jd_text, formatted_experience_level → experience_level."""
        df = load_linkedin_jobs(path=linkedin_csv)
        assert "jd_text" in df.columns
        assert "experience_level" in df.columns
        assert "description" not in df.columns
        assert "formatted_experience_level" not in df.columns

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="LinkedIn"):
            load_linkedin_jobs(path=tmp_path / "nonexistent.csv")

    def test_max_rows(self, linkedin_csv: Path) -> None:
        """Should respect max_rows parameter."""
        df = load_linkedin_jobs(path=linkedin_csv, max_rows=1)
        assert len(df) <= 1


# ────────────────────────────────────────────
# Tests: Resume Score Details Loader (DS-3)
# ────────────────────────────────────────────

class TestLoadResumeScores:
    """Tests for load_resume_scores()."""

    def test_loads_json_files(self, resume_scores_dir: Path) -> None:
        """Should load resume scores from JSON files."""
        df = load_resume_scores(directory=resume_scores_dir)
        assert isinstance(df, pd.DataFrame)
        assert "resume_text" in df.columns
        assert "jd_text" in df.columns
        assert "score" in df.columns
        assert len(df) == 3

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_loads_parquet_files(self, resume_scores_parquet_dir: Path) -> None:
        """Should load resume scores from Parquet files."""
        df = load_resume_scores(directory=resume_scores_parquet_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_score_range(self, resume_scores_dir: Path) -> None:
        """Scores should be clamped to 0–100."""
        df = load_resume_scores(directory=resume_scores_dir)
        assert df["score"].min() >= 0.0
        assert df["score"].max() <= 100.0

    def test_directory_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError, match="Resume score"):
            load_resume_scores(directory=tmp_path / "nonexistent_dir")

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Should raise ValueError for directory with no data files."""
        empty_dir = tmp_path / "empty_scores"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .parquet"):
            load_resume_scores(directory=empty_dir)

    def test_column_name_normalization(self, resume_scores_dir: Path) -> None:
        """Should normalize varying column names to standard schema."""
        # Our fixture uses 'job_description' → should become 'jd_text'
        df = load_resume_scores(directory=resume_scores_dir)
        assert "jd_text" in df.columns
        assert "job_description" not in df.columns

    def test_optional_columns_preserved(self, resume_scores_dir: Path) -> None:
        """Should preserve optional columns like missing_keywords and feedback."""
        df = load_resume_scores(directory=resume_scores_dir)
        assert "missing_keywords" in df.columns
        assert "feedback" in df.columns


# ────────────────────────────────────────────
# Tests: Helper Functions
# ────────────────────────────────────────────

class TestHelpers:
    """Tests for internal helper functions."""

    def test_read_csv_safe_utf8(self, tmp_path: Path) -> None:
        """Should read UTF-8 CSV successfully."""
        csv_path = tmp_path / "utf8.csv"
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(
            csv_path, index=False, encoding="utf-8"
        )
        df = _read_csv_safe(csv_path)
        assert len(df) == 2

    def test_read_csv_safe_latin1(self, tmp_path: Path) -> None:
        """Should fall back to latin-1 encoding."""
        csv_path = tmp_path / "latin1.csv"
        pd.DataFrame({"a": [1], "b": ["café"]}).to_csv(
            csv_path, index=False, encoding="latin-1"
        )
        df = _read_csv_safe(csv_path)
        assert len(df) == 1

    def test_validate_columns_pass(self) -> None:
        """Should not raise when all required columns exist."""
        df = pd.DataFrame({"ID": [1], "Resume_str": ["x"], "Category": ["IT"]})
        _validate_columns(df, {"ID", "Resume_str", "Category"}, "Test")

    def test_validate_columns_fail(self) -> None:
        """Should raise ValueError when columns are missing."""
        df = pd.DataFrame({"ID": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_columns(df, {"ID", "Resume_str"}, "Test")

    def test_drop_high_null_rows(self) -> None:
        """Should drop rows where >50% of columns are null."""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": ["x", None, "z"],
            "c": [10, None, 30],
            "d": [100, None, 300],
        })
        cleaned = _drop_high_null_rows(df)
        assert len(cleaned) == 2  # Row with all-None dropped


# ────────────────────────────────────────────
# Tests: load_all()
# ────────────────────────────────────────────

class TestLoadAll:
    """Tests for the load_all() convenience function."""

    def test_load_all_returns_dict(
        self,
        livecareer_csv: Path,
        linkedin_csv: Path,
        resume_scores_dir: Path,
    ) -> None:
        """Should return a dict with three DataFrames."""
        result = load_all(
            livecareer_path=livecareer_csv,
            linkedin_path=linkedin_csv,
            resume_scores_dir=resume_scores_dir,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"livecareer", "linkedin", "resume_scores"}
        for key in result:
            assert isinstance(result[key], pd.DataFrame)
            assert len(result[key]) > 0
