"""
test_text_cleaner.py — Tests for src/preprocessing/text_cleaner.py
"""

import pytest
from src.preprocessing.text_cleaner import TextCleaner, quick_clean, deep_clean

class TestTextCleaner:
    
    @pytest.fixture
    def cleaner(self):
        return TextCleaner()

    def test_basic_cleaning(self, cleaner):
        text = "Hello,   WORLD!!! This is a test."
        assert cleaner.clean(text) == "hello world this is a test"

    def test_html_removal(self, cleaner):
        text = "<div>Hello</div> <p>World</p>"
        assert cleaner.clean(text) == "hello world"

    def test_url_and_email_removal(self, cleaner):
        text = "Contact me at user@example.com or visit http://example.com"
        # Since 'user' 'at' 'example' 'com' are preserved if not using stopword removal, 
        # we check for the absence of the specific strings.
        cleaned = cleaner.clean(text)
        assert "user@example.com" not in cleaned
        assert "http" not in cleaned

    def test_tech_term_preservation(self, cleaner):
        text = "Expert in C++, C# and .NET development."
        cleaned = cleaner.clean(text)
        assert "c++" in cleaned
        assert "c#" in cleaned
        assert ".net" in cleaned

    def test_whitespace_normalization(self, cleaner):
        text = "Word1 \n\n Word2 \t Word3"
        assert cleaner.clean(text) == "word1 word2 word3"

    def test_stopwords_removal(self):
        cleaner = TextCleaner(remove_stopwords=True)
        text = "this is a test of the system"
        # 'this', 'is', 'a', 'of', 'the' are common English stopwords
        assert cleaner.clean(text) == "test system"

    def test_lemmatization(self):
        cleaner = TextCleaner(use_lemmatization=True)
        text = "running runs ran"
        # Lemma of all these should be 'run' or similar depending on part of speech default
        # NLTK defaults to noun lemmatization unless specified, but 'running' often stays 'running' 
        # without POS tags. 'runs' usually becomes 'run'.
        cleaned = cleaner.clean(text)
        assert "run" in cleaned

    def test_quick_clean(self):
        text = "Clean! Me! 123"
        assert quick_clean(text) == "clean me 123"

    def test_deep_clean(self):
        text = "The running foxes are fast."
        # fox (lemma), fast. 'The', 'are' (stopwords) removed.
        cleaned = deep_clean(text)
        assert "fox" in cleaned
        assert "fast" in cleaned
        assert "the" not in cleaned
        assert "are" not in cleaned

    def test_empty_input(self, cleaner):
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""
        assert cleaner.clean("   ") == ""
