"""
test_model.py -- Manual end-to-end test interface for the ATS Keras model.

Usage:
    python tools/test_model.py --resume resume.pdf --jd jd.txt
    python tools/test_model.py --resume resume.txt --jd jd.txt

Save your job description in a .txt file and pass it with --jd.
The script runs all three ATS outputs: score, missing keywords, and feedback.
"""

import argparse
import sys
from pathlib import Path

# Ensure the ats-ai-core package is importable from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AI_CORE_ROOT = _PROJECT_ROOT / "ats-ai-core"
sys.path.insert(0, str(_AI_CORE_ROOT))

from src.ats_engine.inference import run_ats_inference  # noqa: E402
from src.config import ATS_MODEL_DIR                    # noqa: E402


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF resume using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string, pages joined with newlines.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If no text could be extracted.
    """
    import pdfplumber

    text_pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    if not text_pages:
        raise ValueError(
            f"No text extracted from {pdf_path}. "
            "File may be image-only -- try a text-based PDF."
        )
    return "\n".join(text_pages)


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from a resume file. Supports .pdf and .txt formats.

    Args:
        file_path: Path to the resume file.

    Returns:
        Extracted text as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: For unsupported file types or empty PDFs.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found -- {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Use .pdf or .txt")


def print_results(
    resume_path: Path,
    score: float,
    score_band: str,
    domain_name: str,
    domain_index: int,
    keywords: dict[str, list[str]],
    feedback: list[str],
) -> None:
    """Print all three ATS outputs to the terminal in formatted layout.

    Args:
        resume_path: Path to the resume file that was tested.
        score: ATS score 0-100.
        score_band: Score band label.
        domain_name: Predicted domain name.
        domain_index: Predicted domain index.
        keywords: Dict with keys 'hard_skills', 'soft_skills', 'other'.
        feedback: List of feedback strings.
    """
    file_type = "PDF" if resume_path.suffix.lower() == ".pdf" else "TXT"

    hard_skills = keywords.get("hard_skills", [])
    soft_skills = keywords.get("soft_skills", [])

    print()
    print("=" * 60)
    print("  ATS SCORING ENGINE -- TEST RESULTS")
    print("=" * 60)
    print()
    print(f"  RESUME FILE   : {resume_path.name}")
    print(f"  DETECTED FILE : {file_type}")
    print()
    print("-" * 60)
    print("  OUTPUT 1 -- ATS SCORE")
    print("-" * 60)
    print()
    print(f"  Score     : {score:.0f} / 100")
    print(f"  Band      : {score_band}")
    print(f"  Domain    : {domain_name} (index {domain_index})")
    print()
    print("-" * 60)
    print("  OUTPUT 2 -- MISSING KEYWORDS")
    print("-" * 60)
    print()
    print("  Hard Skills Missing (ranked by JD importance):")
    if hard_skills:
        for i, kw in enumerate(hard_skills[:10], 1):
            if isinstance(kw, dict):
                print(f"    {i}. {kw.get('keyword', kw)}")
            else:
                print(f"    {i}. {kw}")
    else:
        print("    (none detected)")
    print()
    print("  Soft Skills Missing:")
    if soft_skills:
        for i, kw in enumerate(soft_skills[:5], 1):
            if isinstance(kw, dict):
                print(f"    {i}. {kw.get('keyword', kw)}")
            else:
                print(f"    {i}. {kw}")
    else:
        print("    (none detected)")
    print()
    print("-" * 60)
    print("  OUTPUT 3 -- FEEDBACK")
    print("-" * 60)
    print()
    if feedback:
        for i, item in enumerate(feedback, 1):
            print(f"  [{i}] {item}")
    else:
        print("  (no feedback generated)")
    print()
    print("=" * 60)
    print("  END OF REPORT")
    print("=" * 60)
    print()


def main() -> None:
    """Entry point -- parse args, run pipeline, print results."""
    parser = argparse.ArgumentParser(
        description="Test the ATS Keras model with a real resume and job description."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        required=True,
        help="Path to resume file (.pdf or .txt)",
    )
    parser.add_argument(
        "--jd",
        type=Path,
        required=True,
        help="Path to job description text file (.txt)",
    )
    args = parser.parse_args()

    try:
        # -- Step 1: Validate model exists --
        model_weights = ATS_MODEL_DIR / "final_model_weights.h5"
        if not model_weights.exists():
            print(f"ERROR: Keras model not found at {ATS_MODEL_DIR}/. Run training first.")
            sys.exit(1)

        # -- Step 2: Extract resume text --
        resume_text = extract_text_from_file(args.resume)

        # -- Step 3: Read job description from file --
        if not args.jd.exists():
            print(f"ERROR: JD file not found -- {args.jd}")
            sys.exit(1)
        jd_text = args.jd.read_text(encoding="utf-8")
        if not jd_text.strip():
            print("ERROR: Job description file is empty. Please add JD text to the file.")
            sys.exit(1)

        # -- Step 4: Run full inference pipeline --
        print("\nRunning ATS inference pipeline... (this may take a moment)\n")
        result = run_ats_inference(resume_text, jd_text)

        # -- Step 5: Print formatted results --
        print_results(
            resume_path=args.resume,
            score=result["ats_score"],
            score_band=result["score_band"],
            domain_name=result["domain_name"],
            domain_index=result["domain_index"],
            keywords=result["missing_keywords"],
            feedback=result["feedback"],
        )

    except FileNotFoundError:
        print(f"ERROR: File not found -- {args.resume}")
        sys.exit(1)
    except ValueError as e:
        msg = str(e)
        if "Unsupported file type" in msg:
            print("ERROR: Unsupported file type. Use .pdf or .txt")
        elif "No text extracted" in msg or "extract text" in msg.lower():
            print("ERROR: Could not extract text from PDF. Use a text-based (not scanned) PDF.")
        else:
            print(f"ERROR: {msg}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}. Report this to Sai.")
        sys.exit(1)


if __name__ == "__main__":
    main()
