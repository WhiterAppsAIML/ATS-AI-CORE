"""
text_cleaner.py — Robust text preprocessing pipeline for resumes and job descriptions.

Features:
    - HTML tag removal
    - Special character/punctuation removal (preserving common technical ones like C++, .NET)
    - Whitespace normalization
    - Lowercasing
    - Optional Stopword removal & Lemmatization (via NLTK)
"""

import re
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Setup logging
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            logger.warning("Failed to download NLTK resource %s: %s", res, e)

# Initial download
download_nltk_resources()

class TextCleaner:
    """Pipeline for cleaning and normalizing resume and job description text."""

    def __init__(self, use_lemmatization: bool = False, remove_stopwords: bool = False):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Technical term preservation (don't strip '+' from C++, etc.)
        self.tech_terms_regex = re.compile(r'\b(c\+\+|c#|\.net)\b', re.IGNORECASE)

    def clean(self, text: str) -> str:
        """
        Main cleaning entry point.
        
        Args:
            text: Raw input string
            
        Returns:
            Cleaned, normalized string
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # 2. Preserve technical terms temporarily by replacing them with placeholders
        # (This prevents 'C++' from becoming 'C')
        placeholders = {
            "C++": "___CPP___",
            "C#": "___CSHARP___",
            ".NET": "___DOTNET___"
        }
        for term, placeholder in placeholders.items():
            text = re.sub(re.escape(term), placeholder, text, flags=re.IGNORECASE)

        # 3. Basic normalization
        text = text.lower()
        
        # 4. Remove URLs and Email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'http\S+|www\S+', ' ', text)

        # 5. Remove punctuation and special characters (keeping placeholders safe)
        # We replace most punctuation with spaces to avoid joining words
        punctuation_to_remove = string.punctuation.replace('_', '') # Keep _ for our placeholders
        table = str.maketrans(punctuation_to_remove, ' ' * len(punctuation_to_remove))
        text = text.translate(table)

        # 6. Normalize whitespace
        text = ' '.join(text.split())

        # 7. Tokenization based cleaning (Stopwords / Lemmatization)
        if self.remove_stopwords or self.use_lemmatization:
            tokens = word_tokenize(text)
            
            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in self.stop_words]
            
            if self.use_lemmatization:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                
            text = ' '.join(tokens)

        # 8. Restore technical terms
        for term, placeholder in placeholders.items():
            text = text.replace(placeholder.lower(), term.lower())

        return text.strip()

def quick_clean(text: str) -> str:
    """Convenience functional wrapper for light cleaning."""
    return TextCleaner(use_lemmatization=False, remove_stopwords=False).clean(text)

def deep_clean(text: str) -> str:
    """Convenience functional wrapper for heavy cleaning (SEO/TF-IDF ready)."""
    return TextCleaner(use_lemmatization=True, remove_stopwords=True).clean(text)
