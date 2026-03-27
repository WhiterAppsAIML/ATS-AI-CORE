"""
T-11 · src/ats_engine/rubric_layer.py

Post-inference rubric application. NOT a Keras layer — this is a
pure Python post-processing step that adjusts the raw model score
using domain-specific dimension weights.

Called AFTER TFLite inference, not inside the model graph.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from src.config import DOMAIN_LABELS, DOMAIN_WEIGHTS_JSON

logger = logging.getLogger(__name__)

DIMENSION_NAMES: list[str] = [
    "skill_alignment",
    "semantic_contextual_fit",
    "keyword_coverage",
    "structural_completeness",
    "achievement_impact",
]


@lru_cache(maxsize=1)
def _load_weights(path: Path = DOMAIN_WEIGHTS_JSON) -> dict[str, dict[str, float]]:
    """Load and validate domain weight table from JSON.

    Cached after first load. Raises ValueError if any domain row does
    not sum to 1.0 within floating point tolerance.

    Args:
        path: Path to domain_weights.json.

    Returns:
        Nested dict: domain_name → {dimension: weight}.
    """
    with open(path, encoding="utf-8") as f:
        raw: dict = json.load(f)

    # Remove comment key if present
    weights = {k: v for k, v in raw.items() if not k.startswith("_")}

    for domain, dim_weights in weights.items():
        total = sum(dim_weights.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Domain weight row '{domain}' sums to {total:.4f}, expected 1.0. "
                "Fix rubrics/domain_weights.json before training."
            )
    logger.info("Domain weights loaded and validated for %d domains", len(weights))
    return weights


def apply_rubric(
    raw_score: float,
    domain_index: int,
    dimension_scores: dict[str, float],
) -> float:
    """Apply domain-specific rubric weights to produce the final ATS score.

    The raw model score is blended with explicit dimension subscores
    weighted by the domain's rubric table. If dimension_scores is empty,
    the raw score is returned as-is.

    Args:
        raw_score: Raw similarity score from the model [0.0, 1.0].
        domain_index: Predicted domain index 0–6.
        dimension_scores: Dict mapping dimension name → subscore [0.0, 1.0].
            Keys must be a subset of DIMENSION_NAMES.

    Returns:
        Final ATS score in range [0.0, 100.0].
    """
    if not dimension_scores:
        return float(raw_score) * 100.0

    domain_name = DOMAIN_LABELS.get(domain_index, "IT / Software")
    weights_map = _load_weights()
    domain_weights = weights_map.get(domain_name, weights_map.get("IT / Software"))

    weighted_sum = 0.0
    weight_total = 0.0
    for dim, weight in domain_weights.items():
        if dim in dimension_scores:
            weighted_sum += dimension_scores[dim] * weight
            weight_total += weight

    if weight_total < 1e-6:
        return float(raw_score) * 100.0

    # Blend: 60% model raw score, 40% rubric-weighted subscore
    rubric_score = weighted_sum / weight_total
    final = 0.6 * float(raw_score) + 0.4 * rubric_score
    return round(min(max(final * 100.0, 0.0), 100.0), 2)


def validate_weights_file(path: Path = DOMAIN_WEIGHTS_JSON) -> bool:
    """Validate that domain_weights.json is correct without raising.

    Args:
        path: Path to the JSON file to validate.

    Returns:
        True if all domain rows sum to 1.0; False otherwise.
    """
    try:
        _load_weights(path)
        return True
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Weight validation failed: %s", exc)
        return False
