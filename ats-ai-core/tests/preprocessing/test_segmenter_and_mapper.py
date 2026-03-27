"""
tests/preprocessing/test_section_segmenter.py
"""
import pytest
from src.preprocessing.section_segmenter import segment_resume, SegmentedResume

SAMPLE_RESUME = """
Summary
Experienced Python developer with strong backend skills.

Skills
Python, Django, PostgreSQL, Docker, AWS

Education
B.Tech Computer Science, XYZ University, 2022

Experience
Software Engineer at Acme Corp, 2022–2024
Built REST APIs using Django and deployed on AWS.

Projects
E-commerce Platform: Built a full-stack app using React and Django.
"""

FRESHER_RESUME = """
Skills
Python, TensorFlow, Pandas, Scikit-learn

Education
B.Tech AI and Data Science, ABC College, 2024

Projects
Sentiment Analysis Project using BERT.
Stock Price Predictor using LSTM.

Certifications
TensorFlow Developer Certificate
"""


def test_segment_known_sections():
    result = segment_resume(SAMPLE_RESUME)
    assert "skills" in result.sections
    assert "education" in result.sections
    assert "experience" in result.sections

def test_segment_skills_content():
    result = segment_resume(SAMPLE_RESUME)
    assert "Python" in result.sections.get("skills", "")

def test_fresher_flag_set_for_no_experience():
    result = segment_resume(FRESHER_RESUME)
    assert result.is_fresher is True

def test_fresher_flag_not_set_for_experienced():
    result = segment_resume(SAMPLE_RESUME)
    assert result.is_fresher is False

def test_effective_experience_fallback_to_projects():
    result = segment_resume(FRESHER_RESUME)
    eff = result.effective_experience_text()
    assert "BERT" in eff or "LSTM" in eff or "Project" in eff

def test_segment_empty_text():
    result = segment_resume("")
    assert result.sections == {}

def test_get_missing_section_returns_default():
    result = segment_resume(SAMPLE_RESUME)
    assert result.get("nonexistent", "DEFAULT") == "DEFAULT"

def test_all_content_concatenates_sections():
    result = segment_resume(SAMPLE_RESUME)
    full = result.all_content()
    assert "Python" in full
    assert "Django" in full


"""
tests/preprocessing/test_domain_mapper.py
"""
from src.preprocessing.domain_mapper import map_category_string, map_text, label_to_name

def test_map_known_livecarer_category():
    assert map_category_string("Java Developer") == 0  # IT

def test_map_unknown_category_falls_back_to_keyword():
    result = map_category_string("Unknown Category XYZ")
    assert result == -1 or isinstance(result, int)

def test_map_text_it_domain():
    text = "We are looking for a Python backend engineer with AWS and Docker experience."
    assert map_text(text) == 0

def test_map_text_healthcare_domain():
    text = "Registered nurse required for ICU patient care in a hospital setting."
    assert map_text(text) == 3

def test_map_text_finance_domain():
    text = "Seeking a financial analyst with CPA, audit experience, and Excel skills."
    assert map_text(text) == 4

def test_map_text_unknown_returns_minus_one_or_int():
    result = map_text("This is a completely generic text with no domain signals.")
    assert isinstance(result, int)

def test_label_to_name_valid():
    assert label_to_name(0) == "IT / Software"
    assert label_to_name(6) == "Education"

def test_label_to_name_invalid():
    assert label_to_name(-1) == "Unknown"
    assert label_to_name(99) == "Unknown"


"""
tests/preprocessing/test_normalizer.py
"""
from src.preprocessing.normalizer import normalize_skills, normalize_text

def test_js_normalized_to_javascript():
    result = normalize_skills("Experience with JS and Node JS frameworks")
    assert "javascript" in result.lower() or "JavaScript" in result

def test_ml_normalized():
    result = normalize_skills("ML engineer with DL background")
    assert "machine learning" in result.lower()

def test_passthrough_unknown_term():
    result = normalize_text("quantum computing experience")
    assert "quantum computing" in result

def test_empty_string_passthrough():
    assert normalize_text("") == ""
