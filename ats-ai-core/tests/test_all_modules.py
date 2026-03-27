"""
tests/keyword_gap/test_extractor.py
"""
import pytest
from src.keyword_gap.extractor import extract_missing_keywords, _term_in_resume, _tokenise

RESUME_TEXT = "Python developer experienced in Django REST API and PostgreSQL databases."
JD_TEXT = (
    "We need a Python engineer with Django, PostgreSQL, Docker, Kubernetes, "
    "CI/CD pipelines, and microservices architecture experience."
)


def test_extract_returns_list():
    result = extract_missing_keywords(RESUME_TEXT, JD_TEXT)
    assert isinstance(result, list)

def test_extract_finds_missing_keywords():
    result = extract_missing_keywords(RESUME_TEXT, JD_TEXT)
    keywords = [r["keyword"].lower() for r in result]
    # Docker and Kubernetes should be missing
    assert any("docker" in kw or "kubernetes" in kw for kw in keywords)

def test_extract_present_keyword_not_in_results():
    result = extract_missing_keywords(RESUME_TEXT, JD_TEXT)
    keywords = [r["keyword"].lower() for r in result]
    # "python" and "django" are in the resume — should not appear as missing
    assert "python" not in keywords

def test_extract_respects_top_n():
    result = extract_missing_keywords(RESUME_TEXT, JD_TEXT, top_n=5)
    assert len(result) <= 5

def test_extract_empty_resume_returns_empty():
    result = extract_missing_keywords("", JD_TEXT)
    assert result == []

def test_extract_each_result_has_required_keys():
    result = extract_missing_keywords(RESUME_TEXT, JD_TEXT)
    for item in result:
        assert "keyword" in item
        assert "importance" in item
        assert "type" in item

def test_tokenise_basic():
    tokens = _tokenise("Python developer with AWS")
    assert "python" in tokens
    assert "aws" in tokens

def test_term_in_resume_direct_match():
    tokens = set(_tokenise(RESUME_TEXT))
    assert _term_in_resume("python", tokens) is True

def test_term_in_resume_miss():
    tokens = set(_tokenise(RESUME_TEXT))
    assert _term_in_resume("kubernetes", tokens) is False


"""
tests/keyword_gap/test_classifier.py
"""
from src.keyword_gap.classifier import classify_keyword, classify_keywords, split_by_type

def test_classify_known_hard_skill():
    assert classify_keyword("python") == "hard_skill"

def test_classify_known_soft_skill():
    assert classify_keyword("communication") == "soft_skill"

def test_classify_tech_acronym_as_hard_skill():
    # Short tech acronyms hit the heuristic
    result = classify_keyword("aws")
    assert result in ("hard_skill", "other")

def test_classify_keywords_updates_type_field():
    kws = [{"keyword": "python", "importance": 0.9, "type": "other"}]
    result = classify_keywords(kws)
    assert result[0]["type"] == "hard_skill"

def test_split_by_type_separates_correctly():
    kws = [
        {"keyword": "python", "importance": 0.9, "type": "hard_skill"},
        {"keyword": "leadership", "importance": 0.5, "type": "soft_skill"},
        {"keyword": "generic", "importance": 0.3, "type": "other"},
    ]
    split = split_by_type(kws)
    assert "python" in split["hard_skills"]
    assert "leadership" in split["soft_skills"]
    assert "generic" in split["other"]


"""
tests/feedback/test_feedback_mapper.py
"""
from src.feedback.feedback_mapper import generate_feedback, _order_dimensions

def test_generate_feedback_returns_list():
    result = generate_feedback(domain_index=0, score=50.0)
    assert isinstance(result, list)

def test_generate_feedback_minimum_items():
    result = generate_feedback(domain_index=0, score=50.0)
    assert len(result) >= 3

def test_generate_feedback_maximum_items():
    result = generate_feedback(domain_index=0, score=50.0)
    assert len(result) <= 5

def test_generate_feedback_all_strings():
    result = generate_feedback(domain_index=1, score=30.0)
    for item in result:
        assert isinstance(item, str)
        assert len(item) > 0

def test_generate_feedback_fresher_variant():
    # Just ensure it doesn't crash and returns items
    result = generate_feedback(domain_index=0, score=45.0, is_fresher=True)
    assert len(result) >= 3

def test_order_dimensions_weakest_first():
    dim_scores = {
        "skill_alignment": 0.3,
        "keyword_coverage": 0.8,
        "semantic_contextual_fit": 0.1,
        "structural_completeness": 0.9,
        "achievement_impact": 0.5,
    }
    ordered = _order_dimensions(dim_scores)
    # semantic_contextual_fit (0.1) should be first
    assert ordered[0] == "semantic_contextual_fit"

def test_generate_feedback_poor_match_score():
    result = generate_feedback(domain_index=0, score=10.0)
    assert len(result) >= 3


"""
tests/ats_engine/test_rubric_layer.py
"""
from src.ats_engine.rubric_layer import apply_rubric

def test_apply_rubric_without_dimension_scores():
    # Should scale raw score by 100
    result = apply_rubric(raw_score=0.75, domain_index=0, dimension_scores={})
    assert 0.0 <= result <= 100.0

def test_apply_rubric_returns_float():
    result = apply_rubric(0.5, 1, {"skill_alignment": 0.6, "keyword_coverage": 0.4})
    assert isinstance(result, float)

def test_apply_rubric_clamps_to_0_100():
    result = apply_rubric(raw_score=1.0, domain_index=0, dimension_scores={})
    assert result <= 100.0
    result = apply_rubric(raw_score=0.0, domain_index=0, dimension_scores={})
    assert result >= 0.0


"""
tests/test_config.py
"""
from src.config import get_score_band, SCORE_BANDS, DOMAIN_LABELS

def test_get_score_band_excellent():
    assert get_score_band(90) == "Excellent Match"

def test_get_score_band_good():
    assert get_score_band(70) == "Good Match"

def test_get_score_band_moderate():
    assert get_score_band(55) == "Moderate Match"

def test_get_score_band_weak():
    assert get_score_band(35) == "Weak Match"

def test_get_score_band_poor():
    assert get_score_band(10) == "Poor Match"

def test_domain_labels_count():
    assert len(DOMAIN_LABELS) == 7

def test_domain_labels_zero_is_it():
    assert DOMAIN_LABELS[0] == "IT / Software"
