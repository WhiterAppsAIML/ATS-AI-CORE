"""
T-05 · src/preprocessing/normalizer.py

Skill synonym normalisation and text-level normalisation helpers.
Converts common abbreviations and aliases to their canonical forms so
the encoder sees consistent vocabulary across all training samples.
"""

import re

# ── Synonym table ─────────────────────────────────────────────────────────────
# Maps alias → canonical form (lowercase keys for matching)

SKILL_SYNONYMS: dict[str, str] = {
    # Programming languages
    "js": "javascript", "ts": "typescript", "py": "python",
    "c sharp": "c#", "golang": "go",
    # ML / AI
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "cv": "computer vision",
    "ai": "artificial intelligence", "gen ai": "generative ai",
    # Data
    "sql server": "microsoft sql server", "mssql": "microsoft sql server",
    "nosql": "nosql databases", "postgres": "postgresql",
    "mongo": "mongodb",
    # Cloud
    "gcp": "google cloud platform", "aws": "amazon web services",
    "az": "azure", "k8s": "kubernetes",
    # Web
    "react js": "react", "reactjs": "react", "react.js": "react",
    "node js": "node.js", "nodejs": "node.js", "vue js": "vue.js",
    "vuejs": "vue.js", "angular js": "angular",
    # DevOps
    "ci/cd": "ci cd", "cicd": "ci cd",
    # Other
    "oop": "object oriented programming",
    "rest": "rest api", "restful": "rest api",
    "ui/ux": "ui ux", "ui/ux design": "ui ux design",
    "ms office": "microsoft office",
    "ms excel": "microsoft excel", "ms word": "microsoft word",
}

# Pattern: word-boundary sensitive replacements
_SYNONYM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE), canonical)
    for alias, canonical in SKILL_SYNONYMS.items()
]


def normalize_skills(text: str) -> str:
    """Replace skill aliases with their canonical forms in *text*.

    Args:
        text: Any text containing skill mentions.

    Returns:
        Text with aliases replaced by canonical skill names.
    """
    for pattern, canonical in _SYNONYM_PATTERNS:
        text = pattern.sub(canonical, text)
    return text


def normalize_text(text: str) -> str:
    """Apply skill normalisation to a text string.

    Args:
        text: Cleaned text from text_cleaner.

    Returns:
        Normalised text.
    """
    return normalize_skills(text)


def normalize_series(texts: list[str]) -> list[str]:
    """Apply :func:`normalize_text` to a list of strings.

    Args:
        texts: List of cleaned text strings.

    Returns:
        List of normalised strings.
    """
    return [normalize_text(t) for t in texts]
