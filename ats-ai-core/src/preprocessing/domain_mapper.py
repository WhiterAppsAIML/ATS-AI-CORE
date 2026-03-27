"""
T-06 · src/preprocessing/domain_mapper.py

Maps raw text (job titles, resume categories, JD text) to the seven
model domain indices defined in src/config.py.

Domain index table:
    0 = IT / Software
    1 = Non-IT / Management
    2 = Design / Creative
    3 = Healthcare
    4 = Finance / Banking
    5 = Legal
    6 = Education
"""

import re

from src.config import DOMAIN_LABELS, LIVECARER_CATEGORY_MAP

# ── Keyword-based domain detection ───────────────────────────────────────────
# Each domain has a list of strong signal keywords.
# Matching is case-insensitive and word-boundary aware.

_DOMAIN_KEYWORDS: dict[int, list[str]] = {
    0: [  # IT / Software
        "software", "developer", "engineer", "programming", "python",
        "java", "javascript", "react", "node", "backend", "frontend",
        "full stack", "fullstack", "devops", "cloud", "aws", "azure",
        "data science", "machine learning", "deep learning", "ai",
        "database", "sql", "api", "microservice", "kubernetes", "docker",
        "cybersecurity", "network engineer", "sysadmin", "qa", "testing",
        "mobile developer", "flutter", "android", "ios",
    ],
    1: [  # Non-IT / Management
        "manager", "management", "business analyst", "operations",
        "hr", "human resources", "recruiter", "talent acquisition",
        "project manager", "scrum master", "agile", "pmo",
        "supply chain", "logistics", "sales", "marketing",
        "civil engineer", "mechanical engineer", "electrical engineer",
        "manufacturing", "production", "procurement",
    ],
    2: [  # Design / Creative
        "designer", "ux", "ui", "user experience", "user interface",
        "graphic design", "visual design", "product design",
        "motion graphic", "illustrator", "photoshop", "figma",
        "animation", "creative director", "art director",
    ],
    3: [  # Healthcare
        "doctor", "nurse", "physician", "clinical", "medical",
        "pharmacist", "therapist", "dentist", "radiologist",
        "healthcare", "hospital", "patient care", "surgery",
        "public health", "health informatics", "biomedical",
    ],
    4: [  # Finance / Banking
        "finance", "financial", "accountant", "accounting", "auditor",
        "banking", "investment", "equity", "portfolio", "risk",
        "compliance", "tax", "cfa", "cpa", "fintech", "trading",
        "credit analyst", "loan", "insurance", "actuary",
    ],
    5: [  # Legal
        "lawyer", "attorney", "legal", "counsel", "advocate",
        "paralegal", "litigation", "contract", "intellectual property",
        "compliance officer", "regulatory", "law firm", "judiciary",
    ],
    6: [  # Education
        "teacher", "professor", "lecturer", "tutor", "instructor",
        "curriculum", "academic", "school", "university", "training",
        "e-learning", "educator", "faculty", "principal",
        "educational", "pedagogy",
    ],
}

# Pre-compile patterns for speed
_COMPILED_DOMAIN_PATTERNS: dict[int, re.Pattern] = {
    idx: re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        re.IGNORECASE
    )
    for idx, keywords in _DOMAIN_KEYWORDS.items()
}


def map_category_string(category: str) -> int:
    """Map a LiveCareer Category string to a domain index.

    Uses the explicit lookup table from config first; falls back to
    keyword heuristic on the category string itself.

    Args:
        category: Raw category label, e.g. "Java Developer".

    Returns:
        Domain index 0–6, or -1 if unmapped.
    """
    idx = LIVECARER_CATEGORY_MAP.get(category.strip(), None)
    if idx is not None:
        return idx
    return _keyword_match(category)


def map_text(text: str) -> int:
    """Infer domain index from free-form text (job title or JD body).

    Counts keyword hits per domain and returns the domain with the
    highest count. Returns -1 if no domain scores above zero.

    Args:
        text: Job title, description, or any free-form text.

    Returns:
        Domain index 0–6, or -1 if undetermined.
    """
    return _keyword_match(text)


def _keyword_match(text: str) -> int:
    """Score text against all domain keyword lists.

    Args:
        text: Input text string.

    Returns:
        Domain index with highest keyword hit count, or -1 if no match.
    """
    scores: dict[int, int] = {idx: 0 for idx in _COMPILED_DOMAIN_PATTERNS}
    for idx, pat in _COMPILED_DOMAIN_PATTERNS.items():
        scores[idx] = len(pat.findall(text))

    best_idx = max(scores, key=lambda k: scores[k])
    return best_idx if scores[best_idx] > 0 else -1


def label_to_name(domain_index: int) -> str:
    """Convert a domain index to its human-readable name.

    Args:
        domain_index: Integer 0–6.

    Returns:
        Domain name string, or "Unknown" for -1 or out-of-range values.
    """
    return DOMAIN_LABELS.get(domain_index, "Unknown")
