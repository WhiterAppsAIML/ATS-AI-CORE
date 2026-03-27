"""
T-03 · src/preprocessing/text_cleaner.py

Cleans raw resume and job-description text sourced from the three
training datasets. Strips HTML, normalises whitespace, removes
boilerplate, and preserves section-header keywords so the downstream
section segmenter can split the text correctly.
"""

import re
import unicodedata
from html.parser import HTMLParser


# ── HTML stripper ─────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Minimal HTML parser that accumulates visible text only."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags: set[str] = {"script", "style", "head", "meta", "link"}
        self._current_tag: str = ""

    def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[override]
        self._current_tag = tag.lower()
        # Insert a space so words don't merge across tags
        if tag.lower() not in self._skip_tags:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        self._current_tag = ""
        self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._current_tag not in self._skip_tags:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(text: str) -> str:
    """Remove all HTML tags from *text* and return plain text.

    Args:
        text: Raw string that may contain HTML markup.

    Returns:
        Plain text with tags removed. Inline elements collapsed to spaces.
    """
    if not text or "<" not in text:
        return text
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


# ── Unicode normalisation ────────────────────────────────────────────────────

def normalise_unicode(text: str) -> str:
    """Normalise unicode to NFKC and replace common non-ASCII punctuation.

    Args:
        text: Input string, possibly with smart quotes, em-dashes, etc.

    Returns:
        NFKC-normalised string with punctuation replaced by ASCII equivalents.
    """
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'",   # smart single quotes
        "\u201c": '"', "\u201d": '"',   # smart double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dash
        "\u2022": "*", "\u2023": "*",   # bullets
        "\u00a0": " ",                  # non-breaking space
        "\u200b": "",                   # zero-width space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


# ── Boilerplate removal ───────────────────────────────────────────────────────

# Patterns that typically appear in page headers/footers of parsed PDFs
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"confidential", re.IGNORECASE),
    re.compile(r"curriculum\s+vitae", re.IGNORECASE),
    re.compile(r"resume\s*[-–]\s*\w+[\w\s]*\d{4}", re.IGNORECASE),  # "Resume - John Doe 2023"
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),                        # lone page numbers
    re.compile(r"(?:tel|ph|fax|mob)[\s.:]*\+?[\d\s\-()]{7,}", re.IGNORECASE),  # phone
]


def remove_boilerplate(text: str) -> str:
    """Remove common resume/JD boilerplate patterns.

    Args:
        text: Input text after HTML stripping.

    Returns:
        Text with boilerplate lines replaced by spaces.
    """
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


# ── Whitespace and special character normalisation ───────────────────────────

def normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into single spaces; normalise newlines.

    Args:
        text: Input string.

    Returns:
        String with normalised whitespace. Paragraph breaks (double newline)
        are preserved as a single newline to keep section boundaries intact.
    """
    # Collapse multiple blank lines → single newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace tabs and non-newline whitespace with single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def remove_special_characters(text: str) -> str:
    """Remove non-printable and uncommon special characters.

    Keeps: alphanumerics, common punctuation (.,:;!?'"-/()), newlines,
    and the @ and # symbols (useful in tech resumes).

    Args:
        text: Normalised text.

    Returns:
        Cleaned text with special characters removed.
    """
    # Keep alphanumerics, whitespace, and a safe punctuation set
    text = re.sub(r"[^\w\s.,;:!?()\-@#%&+/\n]", " ", text)
    return text


# ── Email and URL masking ─────────────────────────────────────────────────────

def mask_pii(text: str) -> str:
    """Replace email addresses and URLs with placeholder tokens.

    This prevents the encoder from learning patterns tied to specific
    personal identifiers.

    Args:
        text: Cleaned text.

    Returns:
        Text with emails replaced by ``[EMAIL]`` and URLs by ``[URL]``.
    """
    text = re.sub(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text,
    )
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        text,
    )
    return text


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean_text(text: str, mask_personal_info: bool = True) -> str:
    """Full text cleaning pipeline for resume and JD text.

    Applies in order:
      1. HTML stripping
      2. Unicode normalisation
      3. Boilerplate removal
      4. Special character removal
      5. Whitespace normalisation
      6. PII masking (optional)

    Args:
        text: Raw text string from any of the three datasets.
        mask_personal_info: If True, replace emails and URLs with tokens.

    Returns:
        Cleaned, normalised text string ready for encoding or segmentation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = strip_html(text)
    text = normalise_unicode(text)
    text = remove_boilerplate(text)
    text = remove_special_characters(text)
    text = normalise_whitespace(text)
    if mask_personal_info:
        text = mask_pii(text)

    return text


def clean_series(texts: list[str], mask_personal_info: bool = True) -> list[str]:
    """Apply :func:`clean_text` to a list of strings.

    Args:
        texts: List of raw text strings.
        mask_personal_info: Passed through to :func:`clean_text`.

    Returns:
        List of cleaned strings.
    """
    return [clean_text(t, mask_personal_info) for t in texts]
