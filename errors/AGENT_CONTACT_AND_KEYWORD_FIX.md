# AGENT PROMPT INJECTION — Contact Extraction + Keyword Noise Fix
**Priority: HIGH | Sprint: 8.6 — Pre-TFLite Polish**
**Author: Sai | Fix before TFLite conversion**

---

## Context

The test run (`tools/test_model.py`) confirmed all three ATS outputs are working.
Two issues need to be fixed before TFLite conversion:

1. **Contact details not extracted** — name, email, phone must be pulled from the
   resume and shown in terminal output + returned in the result dict
2. **Keyword noise in Output 2** — words like "basic", "highly", "accurate",
   "assist", "industry" are stop-word-level noise appearing as "missing hard skills".
   The keyword extractor needs a proper filter.

Fix both issues in this injection. Do not touch the model, training, or TFLite.

---

## Fix 1 — Contact Detail Extraction

### Task 1A — Create `src/preprocessing/contact_extractor.py`

```python
"""
contact_extractor.py — Extracts candidate contact details from raw resume text.

Extracts: full name, email address, phone number.
Uses regex for email and phone (deterministic).
Uses positional heuristics for name (first non-empty line of resume).
"""

import re
from pathlib import Path


def extract_contact_details(resume_text: str) -> dict[str, str | None]:
    """
    Extract contact details from raw resume text.

    Args:
        resume_text: Full resume text as a plain string.

    Returns:
        dict with keys: 'name', 'email', 'phone'.
        Values are strings if found, None if not found.

    Example:
        {
            "name": "Sai Kumar",
            "email": "sai.kumar@email.com",
            "phone": "+91 98765 43210"
        }
    """
    return {
        "name":  _extract_name(resume_text),
        "email": _extract_email(resume_text),
        "phone": _extract_phone(resume_text),
    }


def _extract_email(text: str) -> str | None:
    """Extract the first email address found in the resume text."""
    pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def _extract_phone(text: str) -> str | None:
    """
    Extract the first phone number found in the resume text.
    Handles formats: +91-XXXXX-XXXXX, +91 XXXXX XXXXX,
    (XXX) XXX-XXXX, XXX-XXX-XXXX, 10-digit Indian numbers.
    """
    patterns = [
        r'\+?[\d\s\-().]{10,17}',          # International / with country code
        r'\b[6-9]\d{9}\b',                  # Indian 10-digit mobile
        r'\(\d{3}\)\s*\d{3}[-.\s]\d{4}',   # US format (XXX) XXX-XXXX
        r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b' # XXX-XXX-XXXX
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(0).strip()
            # Must have at least 10 digits to be a real phone number
            digits = re.sub(r'\D', '', candidate)
            if len(digits) >= 10:
                return candidate
    return None


def _extract_name(text: str) -> str | None:
    """
    Extract candidate name using positional heuristics.

    Strategy: The name is almost always in the first 1-3 non-empty lines
    of a resume, before any contact details appear. It is typically:
    - All-caps or Title Case
    - 2-4 words
    - Contains no digits, @, or special characters
    - NOT a section header like "RESUME", "CURRICULUM VITAE", "CV"

    Args:
        text: Full resume text.

    Returns:
        Name string if detected, None otherwise.
    """
    SKIP_TOKENS = {
        "resume", "curriculum", "vitae", "cv", "profile",
        "biodata", "bio-data", "summary", "objective"
    }
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines[:8]:  # Check only the first 8 lines
        # Skip lines with email, phone, URLs, or special chars
        if re.search(r'[@/\\|#*]', line):
            continue
        if re.search(r'\d{4,}', line):  # Skip lines with long digit sequences
            continue
        if any(token in line.lower() for token in SKIP_TOKENS):
            continue

        words = line.split()
        if 2 <= len(words) <= 5:
            # Each word should look like a name part (mostly letters)
            if all(re.match(r"^[A-Za-z.\-']+$", w) for w in words):
                return line.strip()

    return None
```

---

### Task 1B — Update `src/ats_engine/inference.py`

The `run_inference()` function currently returns:
```python
{"ats_score": float, "domain_label": int}
```

Update it to also return contact details:
```python
{"ats_score": float, "domain_label": int,
 "name": str | None, "email": str | None, "phone": str | None}
```

**Change:** Add `resume_text` as a parameter to `run_inference()` and call
`extract_contact_details()` from `src.preprocessing.contact_extractor`.

Updated signature:
```python
def run_inference(resume_text: str, jd_text: str) -> dict:
    """
    Run ATS model inference and extract contact details from resume.

    Args:
        resume_text: Full resume as plain text string.
        jd_text: Full job description as plain text string.

    Returns:
        dict with keys:
            'ats_score'    : float, range 0–100
            'domain_label' : int, range 0–6
            'name'         : str or None
            'email'        : str or None
            'phone'        : str or None
    """
```

The signature change is backward-compatible — `resume_text` was already
the first argument. Only the return dict grows.

---

### Task 1C — Update `tools/test_model.py` terminal output

Add a CANDIDATE INFO section to `print_results()`, printed before OUTPUT 1:

```
============================================================
  ATS SCORING ENGINE -- TEST RESULTS
============================================================

  RESUME FILE   : [filename]
  DETECTED FILE : [PDF / TXT]

------------------------------------------------------------
  CANDIDATE INFO
------------------------------------------------------------

  Name    : [extracted name or "Not detected"]
  Email   : [extracted email or "Not detected"]
  Phone   : [extracted phone or "Not detected"]

------------------------------------------------------------
  OUTPUT 1 -- ATS SCORE
------------------------------------------------------------
  ...
```

If a field is None, print `Not detected` — never print `None`.

---

## Fix 2 — Keyword Noise Filter

### Task 2A — Update `src/keyword_gap/extractor.py`

The current extractor is returning stop-word-level noise as "missing keywords"
(e.g., "basic", "highly", "accurate", "assist", "industry", "team").

Two changes needed:

**Change 1 — Expand the stop word list**

Add a domain-aware noise filter. After TF-IDF extraction, remove any keyword
that matches this filter before returning results:

```python
KEYWORD_NOISE = {
    # Generic English stop words that leak through TF-IDF
    "basic", "highly", "industry", "accurate", "assist", "control",
    "support", "ensure", "provide", "maintain", "manage", "ability",
    "strong", "good", "excellent", "required", "preferred", "including",
    "experience", "knowledge", "skill", "skills", "work", "working",
    "role", "position", "candidate", "applicant", "responsible",
    "opportunity", "company", "organization", "team", "teams",
    "minimum", "maximum", "least", "must", "will", "shall",
    "across", "within", "around", "toward", "various", "relevant",
    "related", "based", "focused", "oriented", "driven",
    # Single characters and very short tokens
}

def _is_noise(keyword: str) -> bool:
    """Return True if the keyword is noise and should be filtered out."""
    kw = keyword.lower().strip()
    if len(kw) <= 2:
        return True
    if kw in KEYWORD_NOISE:
        return True
    # Filter purely numeric tokens
    if kw.isdigit():
        return True
    # Filter tokens that are just adverbs/adjectives with no skill meaning
    if kw.endswith("ly") and len(kw) < 10:
        return True
    return False
```

Apply `_is_noise()` to filter both hard_skills and soft_skills lists
before returning from `extract_missing_keywords()`.

**Change 2 — Minimum keyword length of 3 characters**

Any extracted keyword shorter than 3 characters must be dropped regardless
of TF-IDF score.

**Change 3 — Hard skill vs soft skill classification**

Confirm the classifier is using `rubrics/keyword_categories.json` for
hard/soft skill classification. If it is falling back to heuristics only,
the JSON file takes priority.

Soft skills list must include at minimum:
`communication, leadership, teamwork, collaboration, problem-solving,
critical thinking, time management, adaptability, interpersonal,
presentation, negotiation, conflict resolution, decision making`

Any keyword not matched to a known hard skill category defaults to soft skill —
never returns as a hard skill unless it matches a known technical term.

---

### Task 2B — Re-run the same test to verify keyword quality

After both fixes, re-run the test with the same resume and JD:

```bash
python tools/test_model.py --resume path/to/842654124-CV-for-pharmaceuticals-company.pdf
```

Use the same JD as before.

**The keyword output must no longer contain:** basic, highly, accurate, assist,
industry, control (unless these are genuinely domain-critical terms in the JD).

Report the new keyword list so Sai can verify the quality improvement.

---

## Task 3 — Unit Tests

**File:** `tests/preprocessing/test_contact_extractor.py`

Write at least 5 test cases:

```python
def test_email_standard():
    text = "John Doe\njohn.doe@gmail.com\n+91 98765 43210"
    result = extract_contact_details(text)
    assert result["email"] == "john.doe@gmail.com"

def test_phone_indian_mobile():
    text = "Priya Sharma\npriya@email.com\n9876543210"
    result = extract_contact_details(text)
    assert result["phone"] == "9876543210"

def test_name_title_case():
    text = "Sai Kumar Reddy\nsai.kumar@email.com\n+91-98765-43210"
    result = extract_contact_details(text)
    assert result["name"] == "Sai Kumar Reddy"

def test_name_not_detected_when_header():
    text = "CURRICULUM VITAE\nJohn Smith\njohn@email.com"
    result = extract_contact_details(text)
    # "CURRICULUM VITAE" must be skipped; "John Smith" must be detected
    assert result["name"] == "John Smith"

def test_missing_fields_return_none():
    text = "This resume has no contact details at all."
    result = extract_contact_details(text)
    assert result["email"] is None
    assert result["phone"] is None
```

---

## Definition of Done

- [ ] `src/preprocessing/contact_extractor.py` created with full docstrings
- [ ] `run_inference()` returns name, email, phone in result dict
- [ ] Terminal output shows CANDIDATE INFO section before OUTPUT 1
- [ ] `None` values print as "Not detected" in terminal
- [ ] Keyword noise filter applied — basic/highly/accurate/assist removed
- [ ] Minimum keyword length of 3 characters enforced
- [ ] Hard/soft skill classification uses `keyword_categories.json` as primary source
- [ ] Test re-run confirms clean keyword output — new list reported to Sai
- [ ] Unit tests written and passing
- [ ] TFLite conversion NOT touched

---

## What NOT to Do

- Do NOT retrain the model
- Do NOT run TFLite conversion
- Do NOT modify `rubrics/feedback_rules.json`
- Do NOT touch `src/summary_generator/`
- Do NOT change the ATS scoring logic — only extraction and filtering

---

*Injection prepared by: Claude Assistant on behalf of Sai*
*Next step after Sai confirms clean output: AGENT_TFLITE_CONVERSION.md*
