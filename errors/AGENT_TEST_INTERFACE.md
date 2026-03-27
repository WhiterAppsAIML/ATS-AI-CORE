# AGENT PROMPT INJECTION — Keras Model Test Interface (Pre-TFLite Validation)
**Priority: HIGH | Sprint: 8.5 — Manual Validation Before Conversion**
**Author: Sai | Run this before executing AGENT_TFLITE_CONVERSION.md**

---

## Directive

Before TFLite conversion, Sai needs to manually test the Keras model end-to-end
by uploading a real resume file and pasting a real job description, and seeing
all three outputs:

1. ATS Score (0–100)
2. Missing Keywords (hard skills + soft skills, ranked)
3. Feedback (domain-specific, actionable)

Build a lightweight local test script that does exactly this — nothing more.
No web framework, no server, no Flutter. A single Python script that runs in
the terminal and accepts a resume file + JD text as inputs.

---

## Task 1 — Build `tools/test_model.py`

**File:** `tools/test_model.py`

Create this file. It must do the following:

### Inputs:
- Resume: accept a file path as a command-line argument (PDF or plain text .txt)
- Job Description: prompt the user to paste multi-line text in the terminal,
  ending with a blank line (press Enter twice to submit)

### Processing:
1. Extract text from the resume file (PDF → plain text, or read .txt directly)
2. Clean both resume text and JD text using the existing
   `src/preprocessing/text_cleaner.py` module
3. Run inference using the existing `src/ats_engine/inference.py` module
4. Run keyword extraction using `src/keyword_gap/extractor.py`
5. Run feedback generation using `src/feedback/feedback_mapper.py`

### Output (print to terminal, clearly formatted):

```
============================================================
  ATS SCORING ENGINE — TEST RESULTS
============================================================

  RESUME FILE   : [filename]
  DETECTED FILE : [PDF / TXT]

------------------------------------------------------------
  OUTPUT 1 — ATS SCORE
------------------------------------------------------------

  Score     : [XX] / 100
  Band      : [Excellent Match / Good Match / Moderate Match /
               Weak Match / Poor Match]
  Domain    : [domain name] (index [N])

------------------------------------------------------------
  OUTPUT 2 — MISSING KEYWORDS
------------------------------------------------------------

  Hard Skills Missing (ranked by JD importance):
    1. [keyword]
    2. [keyword]
    3. [keyword]
    ... (show top 10 maximum)

  Soft Skills Missing:
    1. [keyword]
    2. [keyword]
    ... (show top 5 maximum)

------------------------------------------------------------
  OUTPUT 3 — FEEDBACK
------------------------------------------------------------

  [1] [feedback item]
  [2] [feedback item]
  [3] [feedback item]
  ... (show all feedback items returned)

============================================================
  END OF REPORT
============================================================
```

### Usage from terminal:
```bash
# With a PDF resume
python tools/test_model.py --resume path/to/resume.pdf

# With a plain text resume
python tools/test_model.py --resume path/to/resume.txt
```

After the script starts, the terminal prompts:
```
Paste the Job Description below.
Press Enter twice (blank line) when done:

> [user pastes JD here]
>
[script runs and prints results]
```

---

## Task 2 — PDF Text Extraction

The test script must handle PDF resumes. Use `pdfplumber` for extraction —
it handles multi-column resume layouts better than PyPDF2.

```python
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract plain text from a PDF resume using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string, pages joined with newlines.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If no text could be extracted.
    """
    import pdfplumber
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    if not text_pages:
        raise ValueError(f"No text extracted from {pdf_path}. "
                         "File may be image-only — try a text-based PDF.")
    return "\n".join(text_pages)
```

Install if not present:
```bash
pip install pdfplumber
```

---

## Task 3 — Full Script Structure

The complete `tools/test_model.py` must follow this structure.
Write it with full type hints and docstrings per project RULES.md:

```python
"""
test_model.py — Manual end-to-end test interface for the ATS Keras model.

Usage:
    python tools/test_model.py --resume path/to/resume.pdf
    python tools/test_model.py --resume path/to/resume.txt

The script prompts for a job description in the terminal, then runs
all three ATS outputs: score, missing keywords, and feedback.
"""

import argparse
from pathlib import Path

# Internal modules — do not change import paths
from src.preprocessing.text_cleaner import clean_text
from src.ats_engine.inference import run_inference
from src.keyword_gap.extractor import extract_missing_keywords
from src.feedback.feedback_mapper import get_feedback


DOMAIN_NAMES = {
    0: "IT / Software",
    1: "Non-IT / Management",
    2: "Design / Creative",
    3: "Healthcare",
    4: "Finance / Banking",
    5: "Legal",
    6: "Education",
}

SCORE_BANDS = [
    (85, 100, "Excellent Match"),
    (65,  84, "Good Match"),
    (45,  64, "Moderate Match"),
    (25,  44, "Weak Match"),
    (0,   24, "Poor Match"),
]


def get_score_band(score: float) -> str:
    """Return the score band label for a given ATS score (0–100)."""
    ...


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF resume using pdfplumber."""
    ...


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text from a resume file.
    Supports .pdf and .txt formats.
    """
    ...


def read_jd_from_terminal() -> str:
    """
    Prompt the user to paste a job description in the terminal.
    Reads until a blank line is entered.

    Returns:
        Job description as a single string.
    """
    ...


def print_results(
    resume_path: Path,
    score: float,
    domain_label: int,
    keywords: dict,
    feedback: list[str],
) -> None:
    """Print all three ATS outputs to the terminal in formatted layout."""
    ...


def main() -> None:
    """Entry point — parse args, run pipeline, print results."""
    parser = argparse.ArgumentParser(
        description="Test the ATS Keras model with a real resume and job description."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        required=True,
        help="Path to resume file (.pdf or .txt)"
    )
    args = parser.parse_args()
    ...


if __name__ == "__main__":
    main()
```

Fill in all `...` sections completely. Do not leave stubs.

---

## Task 4 — Error Handling Requirements

The script must handle these failure cases gracefully — print a clear error
message and exit, do not crash with a traceback:

| Failure | Message to Print |
|---------|-----------------|
| Resume file not found | `ERROR: File not found — {path}` |
| Unsupported file type | `ERROR: Unsupported file type. Use .pdf or .txt` |
| PDF has no extractable text | `ERROR: Could not extract text from PDF. Use a text-based (not scanned) PDF.` |
| JD input is blank | `ERROR: Job description cannot be empty. Please paste the JD text.` |
| Model file missing | `ERROR: Keras model not found at model/ats_model/. Run training first.` |
| Any unexpected exception | `ERROR: [exception message]. Report this to Sai.` |

---

## Task 5 — Smoke Test Before Handing Back

After building the script, run it once with a short inline test to confirm
all three outputs are produced without errors:

```bash
# Create a quick test resume file
echo "Python developer. Skills: Django, REST APIs, PostgreSQL, Git.
Projects: E-commerce backend, REST API food delivery app.
Education: B.Tech Computer Science 2024." > /tmp/test_resume.txt

python tools/test_model.py --resume /tmp/test_resume.txt
```

When prompted for JD, paste:
```
Backend Engineer. Python, Django, FastAPI, Docker, PostgreSQL, CI/CD pipelines.
Strong problem solving and communication skills required.
```

**The script must print all three sections** (Score, Missing Keywords, Feedback)
with non-empty values. Paste the full terminal output back to Sai.

---

## What NOT to Do

- Do NOT build a web UI, Flask app, or Streamlit app — terminal only
- Do NOT run TFLite conversion — that comes after Sai tests this script
- Do NOT modify any existing src/ modules — only import and use them
- Do NOT hardcode resume paths or JD text in the script
- Do NOT touch `src/summary_generator/`

---

## Definition of Done

- [ ] `tools/test_model.py` created with full type hints and docstrings
- [ ] PDF extraction works via `pdfplumber`
- [ ] TXT file reading works
- [ ] Terminal JD input works (blank line to submit)
- [ ] All three outputs print in the formatted layout above
- [ ] All error cases handled gracefully
- [ ] Smoke test passes — full terminal output pasted back to Sai
- [ ] TFLite conversion NOT touched

---

*Injection prepared by: Claude Assistant on behalf of Sai*
*This is a pre-conversion manual validation step.*
*After Sai confirms the outputs look correct → proceed to AGENT_TFLITE_CONVERSION.md*
