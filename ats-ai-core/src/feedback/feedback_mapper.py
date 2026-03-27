"""
T-18 · src/feedback/feedback_mapper.py

Deterministic rule-based feedback generator. Maps (domain, score_band,
dimension_scores) to a ranked list of 3–5 actionable feedback strings.

No generative AI — all feedback comes from rubrics/feedback_rules.json.
Fresher-variant feedback is used when is_fresher=True.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from src.config import DOMAIN_LABELS, FEEDBACK_RULES_JSON, get_score_band

logger = logging.getLogger(__name__)

MIN_FEEDBACK_ITEMS: int = 3
MAX_FEEDBACK_ITEMS: int = 5

# Ordered list: dimensions checked from weakest first
DIMENSION_PRIORITY: list[str] = [
    "skill_alignment",
    "keyword_coverage",
    "semantic_contextual_fit",
    "achievement_impact",
    "structural_completeness",
]


@lru_cache(maxsize=1)
def _load_rules(path: Path = FEEDBACK_RULES_JSON) -> dict:
    """Load feedback rules from JSON file.

    Args:
        path: Path to feedback_rules.json.

    Returns:
        Nested dict: domain → score_band → dimension → feedback_dict.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def generate_feedback(
    domain_index: int,
    score: float,
    dimension_scores: dict[str, float] | None = None,
    is_fresher: bool = False,
) -> list[str]:
    """Generate ordered actionable feedback for a resume–JD pair.

    Feedback items are selected by:
      1. Mapping score → score band
      2. Looking up the domain + band in feedback_rules.json
      3. Ordering dimensions from weakest to strongest subscore
      4. Selecting up to MAX_FEEDBACK_ITEMS rules
      5. Using fresher_variant strings when is_fresher=True

    Args:
        domain_index: Predicted domain index 0–6.
        score: ATS score 0–100.
        dimension_scores: Optional dict of dimension name → subscore [0–1].
            If None, all dimensions are weighted equally.
        is_fresher: If True, use fresher-friendly feedback variants.

    Returns:
        List of 3–5 actionable feedback strings. Returns a generic
        fallback list if no matching rules are found.
    """
    score_band   = get_score_band(score)
    domain_name  = DOMAIN_LABELS.get(domain_index, "IT / Software")

    try:
        rules = _load_rules()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load feedback rules: %s", exc)
        return _fallback_feedback(score_band, is_fresher)

    # Navigate: domain → band → dimension
    band_rules = (
        rules
        .get(domain_name, rules.get("IT / Software", {}))
        .get(score_band, {})
    )

    if not band_rules:
        logger.debug(
            "No rules found for domain='%s', band='%s'. Using fallback.",
            domain_name, score_band,
        )
        return _fallback_feedback(score_band, is_fresher)

    # Order dimensions: weakest score first
    ordered_dims = _order_dimensions(dimension_scores)

    feedback_items: list[str] = []
    for dim in ordered_dims:
        if dim not in band_rules:
            continue
        rule = band_rules[dim]
        if isinstance(rule, dict):
            text = rule.get("fresher_variant", rule.get("feedback", "")) if is_fresher \
                   else rule.get("feedback", "")
        else:
            text = str(rule)
        if text:
            feedback_items.append(text)
        if len(feedback_items) >= MAX_FEEDBACK_ITEMS:
            break

    # Pad to minimum if needed
    while len(feedback_items) < MIN_FEEDBACK_ITEMS:
        generic = _generic_fallback_for_band(score_band, is_fresher)
        for item in generic:
            if item not in feedback_items:
                feedback_items.append(item)
                break
        else:
            break   # Avoid infinite loop

    return feedback_items[:MAX_FEEDBACK_ITEMS]


def _order_dimensions(
    dimension_scores: dict[str, float] | None,
) -> list[str]:
    """Return dimensions ordered from weakest to strongest subscore.

    Args:
        dimension_scores: Dict of dimension → score [0–1], or None.

    Returns:
        Ordered list of dimension names (weakest first).
    """
    if not dimension_scores:
        return DIMENSION_PRIORITY

    scored = {
        dim: dimension_scores.get(dim, 0.5)
        for dim in DIMENSION_PRIORITY
    }
    return sorted(scored, key=lambda d: scored[d])


def _fallback_feedback(score_band: str, is_fresher: bool) -> list[str]:
    """Generic fallback feedback when no domain-specific rules exist.

    Args:
        score_band: Score band label.
        is_fresher: Whether to use fresher-friendly language.

    Returns:
        List of generic feedback strings.
    """
    return _generic_fallback_for_band(score_band, is_fresher)[:MAX_FEEDBACK_ITEMS]


def _generic_fallback_for_band(score_band: str, is_fresher: bool) -> list[str]:
    """Return generic feedback items for a given score band.

    Args:
        score_band: Score band label.
        is_fresher: Whether to use fresher-friendly language.

    Returns:
        List of generic feedback strings.
    """
    exp_phrase = (
        "academic projects, certifications, or internships"
        if is_fresher else "relevant work experience"
    )
    band_map: dict[str, list[str]] = {
        "Excellent Match": [
            "Your resume is strongly aligned with this role — consider applying.",
            "Highlight your most recent achievements to stand out further.",
            "Ensure all listed technologies match the exact versions mentioned in the JD.",
        ],
        "Good Match": [
            f"Add a few more keywords from the JD to your skills section.",
            f"Quantify your achievements with numbers or percentages where possible.",
            f"Ensure your {exp_phrase} clearly maps to the role requirements.",
        ],
        "Moderate Match": [
            f"Identify the top 5 missing keywords from the JD and add them naturally to your resume.",
            f"Rewrite bullet points to use the same terminology as the job description.",
            f"Expand your {exp_phrase} section to include domain-relevant details.",
            "Add or update your Skills section to reflect the technologies in the JD.",
        ],
        "Weak Match": [
            f"Significant gaps detected. Focus on closing the skill gap for the top 3 required technologies.",
            f"Tailor your resume summary to specifically target this job domain.",
            f"Consider adding {exp_phrase} that demonstrates the core skills required.",
            "Review the JD carefully and restructure your resume around its requirements.",
        ],
        "Poor Match": [
            "This role appears to be in a different domain than your current resume.",
            "Consider applying to roles that better match your existing skill set.",
            "If you wish to transition, build foundational skills in the target domain first.",
        ],
    }
    return band_map.get(score_band, band_map["Moderate Match"])
