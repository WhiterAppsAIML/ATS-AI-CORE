"""
tests/preprocessing/test_text_cleaner.py
"""
import pytest
from src.preprocessing.text_cleaner import (
    clean_text, strip_html, normalise_unicode, normalise_whitespace,
    remove_boilerplate, mask_pii,
)

def test_strip_html_removes_tags():
    result = strip_html("<p>Hello <b>World</b></p>")
    assert "<" not in result
    assert "Hello" in result
    assert "World" in result

def test_strip_html_no_html_passthrough():
    text = "Plain text without any tags"
    assert strip_html(text) == text

def test_normalise_unicode_smart_quotes():
    result = normalise_unicode("\u201cHello\u201d it\u2019s a test")
    assert '"Hello"' in result
    assert "it's" in result

def test_remove_boilerplate_page_number():
    result = remove_boilerplate("Some content\nPage 1 of 3\nMore content")
    assert "Page 1 of 3" not in result

def test_mask_pii_removes_email():
    result = mask_pii("Contact me at john.doe@example.com for details")
    assert "john.doe@example.com" not in result
    assert "[EMAIL]" in result

def test_mask_pii_removes_url():
    result = mask_pii("Visit https://linkedin.com/in/johndoe for profile")
    assert "linkedin.com" not in result
    assert "[URL]" in result

def test_clean_text_full_pipeline():
    raw = "<p>Python developer with 3yrs exp.</p>\nPage 1 of 2\nContact: dev@test.com"
    result = clean_text(raw)
    assert "<p>" not in result
    assert "Page 1 of 2" not in result
    assert "[EMAIL]" in result
    assert "Python developer" in result

def test_clean_text_empty_returns_empty():
    assert clean_text("") == ""
    assert clean_text("   ") == ""

def test_normalise_whitespace_collapses_spaces():
    result = normalise_whitespace("hello    world\n\n\n\ntest")
    assert "  " not in result
