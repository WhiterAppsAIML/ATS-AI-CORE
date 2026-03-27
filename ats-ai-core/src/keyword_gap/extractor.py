"""
T-15 · src/keyword_gap/extractor.py

TF-IDF-based missing keyword extractor. Identifies high-importance
terms in the job description that are absent from the resume.

This is POST-PROCESSING code — it runs after TFLite inference using
the raw text inputs, not inside the model graph.
"""

import logging
import re
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

KeywordType = Literal["hard_skill", "soft_skill", "domain_term", "other"]


class KeywordGapExtractor:
    """Extracts missing keywords from a resume relative to a job description.

    Uses TF-IDF to score term importance in the JD, then diffs against
    the resume vocabulary to identify gaps.
    """

    def __init__(self, top_n: int = 20) -> None:
        """Initialise the extractor.

        Args:
            top_n: Maximum number of missing keywords to return.
        """
        self.top_n = top_n
        self._vectoriser = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True,
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,30}\b",
        )

    def extract_missing_keywords(
        self,
        resume_text: str,
        jd_text: str,
    ) -> list[dict[str, object]]:
        """Return ranked list of keywords present in JD but absent from resume.

        Args:
            resume_text: Cleaned resume text.
            jd_text: Cleaned job description text.

        Returns:
            List of dicts, each with keys:
                keyword   (str)   — the missing term
                importance (float) — TF-IDF weight in the JD
                type      (str)   — "hard_skill" | "soft_skill" | "domain_term" | "other"
        """
        if not jd_text.strip() or not resume_text.strip():
            return []

        # Fit TF-IDF on JD only — we want importance relative to this specific JD
        try:
            self._vectoriser.fit([jd_text])
        except ValueError:
            logger.warning("TF-IDF fit failed — JD may be too short.")
            return []

        feature_names: list[str] = self._vectoriser.get_feature_names_out().tolist()
        jd_vector = self._vectoriser.transform([jd_text]).toarray()[0]

        # Build resume token set (lowercased)
        resume_tokens = set(_tokenise(resume_text))

        results: list[dict[str, object]] = []
        for term, importance in sorted(
            zip(feature_names, jd_vector), key=lambda x: -x[1]
        ):
            if importance < 1e-6:
                continue
            # Check if the term (or its unigram components) appears in the resume
            if _term_in_resume(term, resume_tokens):
                continue
            results.append({
                "keyword":    term,
                "importance": round(float(importance), 4),
                "type":       "other",  # classifier fills this in next step
            })
            if len(results) >= self.top_n:
                break

        return results


def _tokenise(text: str) -> list[str]:
    """Tokenise text into lowercased words for set-membership checks.

    Args:
        text: Input text string.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,30}\b", text.lower())


def _term_in_resume(term: str, resume_tokens: set[str]) -> bool:
    """Check whether *term* or all its component unigrams appear in the resume.

    Args:
        term: Single- or multi-word term to check.
        resume_tokens: Set of lowercased resume tokens.

    Returns:
        True if the term is covered by the resume vocabulary.
    """
    term_lower = term.lower()
    if term_lower in resume_tokens:
        return True
    # For bigrams: consider present if both unigrams are in resume
    parts = term_lower.split()
    if len(parts) > 1 and all(p in resume_tokens for p in parts):
        return True
    return False


def extract_missing_keywords(
    resume_text: str,
    jd_text: str,
    top_n: int = 20,
) -> list[dict[str, object]]:
    """Module-level convenience wrapper for :class:`KeywordGapExtractor`.

    Args:
        resume_text: Cleaned resume text.
        jd_text: Cleaned job description text.
        top_n: Maximum keywords to return.

    Returns:
        Ranked list of missing keyword dicts (see KeywordGapExtractor).
    """
    extractor = KeywordGapExtractor(top_n=top_n)
    return extractor.extract_missing_keywords(resume_text, jd_text)
