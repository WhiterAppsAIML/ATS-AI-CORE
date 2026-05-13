"""
src/ats_engine/inference.py

Single inference entry point for the ATS AI Core pipeline.
Orchestrates text cleaning, normalisation, model prediction,
keyword gap analysis, and feedback generation into one call.

Usage:
    from src.ats_engine.inference import run_ats_inference
    result = run_ats_inference(resume_text, jd_text)
"""

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.config import (
    ATS_MODEL_DIR,
    DOMAIN_LABELS,
    get_score_band,
)
from src.ats_engine.model import build_ats_model
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.normalizer import normalize_text
from src.preprocessing.section_segmenter import segment_resume
from src.keyword_gap.extractor import extract_missing_keywords
from src.keyword_gap.classifier import classify_keywords, split_by_type
from src.feedback.feedback_mapper import generate_feedback

logger = logging.getLogger(__name__)


# ── Singleton model loader ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model(model_path: str | None = None) -> tf.keras.Model:
    """Load and cache the ATS model (built + weights).

    Args:
        model_path: Optional path to model weights (.h5).
            Defaults to ATS_MODEL_DIR / "final_model_weights.h5".

    Returns:
        Compiled Keras model with weights loaded.
    """
    weights = Path(model_path) if model_path else ATS_MODEL_DIR / "final_model_weights.h5"
    logger.info("Loading ATS model weights from %s", weights)

    model = build_ats_model()
    model.load_weights(str(weights))

    # Warm-up inference to compile the graph
    _ = model.predict(
        {"resume_text": np.array(["warmup"]), "jd_text": np.array(["warmup"])},
        verbose=0,
    )
    logger.info("ATS model loaded and warmed up.")
    return model


# ── Safe empty result ─────────────────────────────────────────────────────────

def _empty_result(reason: str) -> dict:
    """Return a zeroed-out result dict with a feedback message.

    Args:
        reason: Human-readable explanation of why inference was skipped.

    Returns:
        Dict matching the run_ats_inference output contract.
    """
    return {
        "ats_score": 0.0,
        "score_band": "Poor Match",
        "domain_index": 0,
        "domain_name": DOMAIN_LABELS.get(0, "Unknown"),
        "missing_keywords": {"hard_skills": [], "soft_skills": [], "other": []},
        "feedback": [reason],
        "is_fresher": False,
        "contact": {"name": "", "email": "", "phone": ""},
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_ats_inference(
    resume_text: str,
    jd_text: str,
    model_path: str | None = None,
) -> dict:
    """Run the full ATS inference pipeline on a single resume–JD pair.

    Steps:
        1. clean_text + normalize_text on both inputs
        2. segment_resume on the resume to detect is_fresher
        3. model.predict via a cached singleton model
        4. extract_missing_keywords → classify_keywords → split_by_type
        5. generate_feedback

    Args:
        resume_text: Raw resume text.
        jd_text: Raw job description text.
        model_path: Optional path to model weights file.

    Returns:
        Dict with keys: ats_score, score_band, domain_index, domain_name,
        missing_keywords, feedback, is_fresher.
    """
    # ── Edge case: empty input ────────────────────────────────────────────
    if not resume_text or not resume_text.strip():
        return _empty_result("Resume text is empty. Please provide a valid resume.")
    if not jd_text or not jd_text.strip():
        return _empty_result("Job description text is empty. Please provide a valid JD.")

    # ── Step 1: Clean and normalise ───────────────────────────────────────
    clean_resume = normalize_text(clean_text(resume_text))
    clean_jd = normalize_text(clean_text(jd_text))

    if len(clean_resume.strip()) < 10:
        return _empty_result("Resume too short after cleaning. Provide more content.")
    if len(clean_jd.strip()) < 10:
        return _empty_result("Job description too short after cleaning. Provide more content.")

    # ── Step 2: Segment resume → fresher detection ────────────────────────
    segmented = segment_resume(clean_resume)
    is_fresher = segmented.is_fresher

    # ── Step 3: Model prediction (cached singleton) ───────────────────────
    model = _load_model(model_path)
    predictions = model.predict(
        {"resume_text": np.array([clean_resume]), "jd_text": np.array([clean_jd])},
        verbose=0,
    )

    # model outputs: [ats_score (batch,1), domain_logits (batch,7)]
    score_raw = float(predictions[0][0][0])          # sigmoid output in [0, 1]
    domain_probs = predictions[1][0]                  # softmax probabilities [7]
    domain_index = int(np.argmax(domain_probs))       # argmax at inference time

    ats_score = round(score_raw * 100.0, 2)           # scale to 0–100
    score_band = get_score_band(ats_score)
    domain_name = DOMAIN_LABELS.get(domain_index, "Unknown")

    # ── Step 4: Keyword gap analysis ──────────────────────────────────────
    raw_keywords = extract_missing_keywords(clean_resume, clean_jd, top_n=15)
    classified_keywords = classify_keywords(raw_keywords)
    missing_keywords = split_by_type(classified_keywords)

    # ── Step 5: Feedback generation ───────────────────────────────────────
    feedback = generate_feedback(
        domain_index=domain_index,
        score=ats_score,
        is_fresher=is_fresher,
    )

    # ── Contact extraction ──────────────────────────────────────────────
    def _extract_contact(text: str) -> dict:
        """Extract name, email and phone from raw resume text."""
        # Email — standard pattern
        email_match = re.search(
            r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
            text
        )
        email = email_match.group(0) if email_match else ""

        # Phone — handles formats: +91-9876543210, (98765) 43210,
        #          9876543210, +1 (555) 123-4567
        phone_match = re.search(
            r'(\+?\d{1,3}[\s\-]?)?(\(?\d{3,5}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{4,5}',
            text
        )
        phone = phone_match.group(0).strip() if phone_match else ""

        # Name — take the first non-empty line of the resume
        # Most resumes start with the candidate's name on line 1
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        name = lines[0] if lines else ""
        # If first line looks like a section header or email, try line 2
        if name and (
            '@' in name or
            name.lower().startswith(('resume', 'curriculum', 'cv', 'profile'))
        ):
            name = lines[1] if len(lines) > 1 else ""

        return {
            "name":  name,
            "email": email,
            "phone": phone
        }

    contact_info = _extract_contact(resume_text)
    # ── End contact extraction ───────────────────────────────────────────

    return {
        "ats_score": ats_score,
        "score_band": score_band,
        "domain_index": domain_index,
        "domain_name": domain_name,
        "missing_keywords": missing_keywords,
        "feedback": feedback,
        "is_fresher": is_fresher,
        "contact": contact_info,
    }


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    sample_resume = """
    John Doe
    Software Engineer | Python, Java, React

    Summary
    Experienced software engineer with 5 years of experience building
    scalable web applications and microservices.

    Skills
    Python, Java, JavaScript, React, Node.js, PostgreSQL, Docker, Git,
    REST API, Agile, CI/CD

    Experience
    Senior Software Engineer — TechCorp Inc. (2021–Present)
    - Built microservices architecture serving 2M requests/day
    - Led migration from monolith to Kubernetes-based infrastructure
    - Reduced API response time by 40% through query optimisation

    Software Developer — StartupXYZ (2019–2021)
    - Developed React frontend with 95% test coverage
    - Implemented OAuth2 authentication and role-based access control

    Education
    B.Tech Computer Science — State University (2019)
    """

    sample_jd = """
    Senior Backend Engineer

    We are looking for a Senior Backend Engineer to join our platform team.

    Requirements:
    - 5+ years experience with Python or Go
    - Strong experience with AWS, Terraform, and Kubernetes
    - Expertise in designing RESTful APIs and event-driven architectures
    - Experience with PostgreSQL, Redis, and message queues (Kafka/RabbitMQ)
    - Proficiency in CI/CD pipelines and infrastructure as code
    - Excellent problem-solving and communication skills
    - Experience with monitoring tools (Datadog, Prometheus, Grafana)

    Nice to have:
    - Experience with GraphQL
    - Contributions to open source projects
    - Machine learning inference pipeline experience
    """

    print("=" * 60)
    print("ATS Inference — Quick Test")
    print("=" * 60)

    result = run_ats_inference(sample_resume, sample_jd)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    sys.exit(0)
