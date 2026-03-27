"""
T-04 · src/preprocessing/section_segmenter.py

Splits a cleaned resume text into named sections using regex-based
header detection. Returns a dict mapping section name → section text.

Supported section names (canonical):
    summary, skills, education, experience, projects,
    certifications, awards, publications, other

Fresher rule: if 'experience' section is absent or very short,
'projects' is promoted to equivalent weight in downstream scoring.
"""

import re
from dataclasses import dataclass, field

# ── Section header patterns ───────────────────────────────────────────────────
# Each tuple: (canonical_name, list_of_regex_patterns)
# Patterns are tried in order; first match wins for that line.

_SECTION_HEADERS: list[tuple[str, list[str]]] = [
    ("summary", [
        r"summary", r"professional\s+summary", r"career\s+objective",
        r"objective", r"profile", r"about\s+me", r"overview",
        r"personal\s+statement",
    ]),
    ("skills", [
        r"skill", r"technical\s+skill", r"core\s+competenc",
        r"technologies", r"tools?\s+&\s+technolog", r"expertise",
        r"key\s+skill", r"areas?\s+of\s+expertise",
    ]),
    ("experience", [
        r"experience", r"work\s+experience", r"employment",
        r"professional\s+experience", r"work\s+history",
        r"career\s+history", r"internship",
    ]),
    ("education", [
        r"education", r"academic", r"qualification",
        r"degree", r"university", r"college",
    ]),
    ("projects", [
        r"project", r"personal\s+project", r"academic\s+project",
        r"key\s+project", r"portfolio",
    ]),
    ("certifications", [
        r"certification", r"certificate", r"accreditation",
        r"license", r"credential",
    ]),
    ("awards", [
        r"award", r"honor", r"honour", r"achievement",
        r"recognition", r"scholarship",
    ]),
    ("publications", [
        r"publication", r"research", r"paper", r"journal",
    ]),
]

# Pre-compile all patterns for speed
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (name, [re.compile(r"\b" + pat + r"\b", re.IGNORECASE) for pat in pats])
    for name, pats in _SECTION_HEADERS
]

# A header line is typically short and matches a known section keyword
_HEADER_LINE_RE = re.compile(
    r"^[\s\W]*(" +
    "|".join(pat for _, pats in _SECTION_HEADERS for pat in pats) +
    r")[\s\W]*$",
    re.IGNORECASE,
)

# Minimum characters for a section to be considered non-empty
_MIN_SECTION_CHARS: int = 20
# If experience section is shorter than this, fresher flag is set
_FRESHER_EXPERIENCE_THRESHOLD: int = 80


@dataclass
class SegmentedResume:
    """Holds the sections extracted from a single resume.

    Attributes:
        sections: Mapping from canonical section name to raw text.
        is_fresher: True if the experience section is absent or very short.
        raw_text: The original (unsegmented) cleaned text.
    """

    sections: dict[str, str] = field(default_factory=dict)
    is_fresher: bool = False
    raw_text: str = ""

    def get(self, section: str, default: str = "") -> str:
        """Return section text or *default* if absent.

        Args:
            section: Canonical section name.
            default: Value to return if section is not found.

        Returns:
            Section text string.
        """
        return self.sections.get(section, default)

    def effective_experience_text(self) -> str:
        """Return experience text, falling back to projects for freshers.

        Returns:
            Experience section text, or projects text if experience is absent.
        """
        exp = self.sections.get("experience", "")
        if len(exp) < _FRESHER_EXPERIENCE_THRESHOLD:
            return self.sections.get("projects", exp)
        return exp

    def all_content(self) -> str:
        """Concatenate all section texts into a single string.

        Returns:
            Full resume text as a single space-joined string.
        """
        return " ".join(self.sections.values())


# ── Core segmentation logic ───────────────────────────────────────────────────

def _classify_line(line: str) -> str | None:
    """Return the canonical section name if *line* looks like a section header.

    Args:
        line: A single line from the resume text.

    Returns:
        Canonical section name, or None if the line is not a header.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        # Section headers are typically short
        return None
    if not _HEADER_LINE_RE.match(stripped):
        return None
    for name, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(stripped):
                return name
    return None


def segment_resume(text: str) -> SegmentedResume:
    """Split a cleaned resume text into named sections.

    Uses a single-pass line-by-line algorithm:
    - Lines matching a known header pattern open a new section.
    - All subsequent lines are appended to the current section until
      the next header is found.

    Args:
        text: Cleaned resume text (output of text_cleaner.clean_text).

    Returns:
        :class:`SegmentedResume` with populated sections dict and fresher flag.
    """
    result = SegmentedResume(raw_text=text)

    if not text.strip():
        return result

    lines = text.splitlines()
    current_section: str = "other"
    buffer: dict[str, list[str]] = {"other": []}

    for line in lines:
        section_name = _classify_line(line)
        if section_name:
            current_section = section_name
            if current_section not in buffer:
                buffer[current_section] = []
        else:
            buffer.setdefault(current_section, []).append(line)

    # Assemble sections; drop empty ones
    for name, lines_list in buffer.items():
        content = "\n".join(lines_list).strip()
        if len(content) >= _MIN_SECTION_CHARS:
            result.sections[name] = content

    # Fresher detection
    experience_text = result.sections.get("experience", "")
    result.is_fresher = len(experience_text) < _FRESHER_EXPERIENCE_THRESHOLD

    return result


def segment_batch(texts: list[str]) -> list[SegmentedResume]:
    """Apply :func:`segment_resume` to a list of resume texts.

    Args:
        texts: List of cleaned resume text strings.

    Returns:
        List of :class:`SegmentedResume` objects.
    """
    return [segment_resume(t) for t in texts]
