"""
T-16 · src/keyword_gap/classifier.py

Classifies extracted keywords into hard_skill, soft_skill,
domain_term, or other. Uses seed lists from rubrics/keyword_categories.json
with a fallback noun-phrase heuristic.
"""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from src.config import KEYWORD_CATEGORIES_JSON

logger = logging.getLogger(__name__)
KeywordType = Literal["hard_skill", "soft_skill", "domain_term", "other"]


@lru_cache(maxsize=1)
def _load_seed_lists(path: Path = KEYWORD_CATEGORIES_JSON) -> dict[str, list[str]]:
    """Load keyword category seed lists from JSON.

    Args:
        path: Path to keyword_categories.json.

    Returns:
        Dict with keys 'hard_skill_signals' and 'soft_skill_signals'.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "hard": [kw.lower() for kw in data.get("hard_skill_signals", [])],
        "soft": [kw.lower() for kw in data.get("soft_skill_signals", [])],
    }


def classify_keyword(keyword: str) -> KeywordType:
    """Classify a single keyword string into a skill category.

    Classification order:
      1. Exact or substring match against hard-skill seed list → "hard_skill"
      2. Exact or substring match against soft-skill seed list → "soft_skill"
      3. Heuristic: contains digit, dot, or is an acronym → "hard_skill"
      4. Default → "other"

    Args:
        keyword: The keyword string to classify (case-insensitive).

    Returns:
        One of "hard_skill", "soft_skill", "domain_term", "other".
    """
    kw_lower = keyword.lower().strip()

    try:
        seeds = _load_seed_lists()
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("keyword_categories.json not found; using heuristic only.")
        seeds = {"hard": [], "soft": []}

    # ── Exact / partial seed match ────────────────────────────────────────────
    for hard_seed in seeds["hard"]:
        if hard_seed in kw_lower or kw_lower in hard_seed:
            return "hard_skill"

    for soft_seed in seeds["soft"]:
        if soft_seed in kw_lower or kw_lower in soft_seed:
            return "soft_skill"

    # ── Heuristic patterns ────────────────────────────────────────────────────
    # Version numbers or tech acronyms (e.g. "python3", "aws", "ci cd", "k8s")
    if re.search(r"\d", kw_lower):
        return "hard_skill"
    if re.match(r"^[a-z0-9+#.\-/]{1,8}$", kw_lower) and len(kw_lower) <= 8:
        return "hard_skill"

    return "other"


def classify_keywords(
    keywords: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Classify a list of keyword dicts in-place (updates 'type' field).

    Args:
        keywords: List of keyword dicts from the extractor
            (each must have a 'keyword' key).

    Returns:
        The same list with 'type' field updated.
    """
    for kw_dict in keywords:
        kw_dict["type"] = classify_keyword(str(kw_dict.get("keyword", "")))
    return keywords


def split_by_type(
    keywords: list[dict[str, object]],
) -> dict[str, list[str]]:
    """Split a classified keyword list into typed groups.

    Args:
        keywords: List of classified keyword dicts.

    Returns:
        Dict with keys 'hard_skills', 'soft_skills', 'other',
        each holding a list of keyword strings.
    """
    result: dict[str, list[str]] = {"hard_skills": [], "soft_skills": [], "other": []}
    for kw in keywords:
        kw_type = str(kw.get("type", "other"))
        kw_word = str(kw.get("keyword", ""))
        if kw_type == "hard_skill":
            result["hard_skills"].append(kw_word)
        elif kw_type == "soft_skill":
            result["soft_skills"].append(kw_word)
        else:
            result["other"].append(kw_word)
    return result
