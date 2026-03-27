"""
config.py — Central configuration for all ATS AI Core modules.

All hyperparameters, file paths, and domain constants live here.
Never hardcode these values in model, trainer, or inference files.
"""

from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).parent.parent.resolve()
# The parent project directory holds all raw data in ats/data/raw/
PROJECT_DIR: Path = ROOT_DIR.parent.resolve()
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = PROJECT_DIR / "data" / "raw"       # actual raw data is in parent
PROCESSED_DIR: Path = DATA_DIR / "processed"
LABELED_DIR: Path = DATA_DIR / "labeled"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
MODEL_DIR: Path = ROOT_DIR / "model"
ATS_MODEL_DIR: Path = MODEL_DIR / "ats_model"
TFLITE_DIR: Path = MODEL_DIR / "tflite"
RUBRICS_DIR: Path = ROOT_DIR / "rubrics"
EVALUATION_DIR: Path = ROOT_DIR / "evaluation"

# ── Raw dataset paths ────────────────────────────────────────────────────────
RESUME_CSV: Path = RAW_DIR / "resume_dataset" / "Resume" / "Resume.csv"
LINKEDIN_CSV: Path = RAW_DIR / "linkedin_jobs" / "postings.csv"
RESUME_SCORE_DIR: Path = RAW_DIR / "resume_score_details"

# ── Processed / labeled data paths ──────────────────────────────────────────
TRAINING_PAIRS_CSV: Path = LABELED_DIR / "training_pairs.csv"
WEAK_LABELS_CSV: Path = LABELED_DIR / "weak_labels.csv"
GOLD_LABELS_CSV: Path = LABELED_DIR / "gold_labels.csv"

# ── Rubric paths ─────────────────────────────────────────────────────────────
DOMAIN_WEIGHTS_JSON: Path = RUBRICS_DIR / "domain_weights.json"
FEEDBACK_RULES_JSON: Path = RUBRICS_DIR / "feedback_rules.json"
KEYWORD_CATEGORIES_JSON: Path = RUBRICS_DIR / "keyword_categories.json"

# ── TFLite paths ─────────────────────────────────────────────────────────────
TFLITE_OUTPUT_PATH: Path = TFLITE_DIR / "ats_core.tflite"

# ── Encoder ──────────────────────────────────────────────────────────────────
# USE v4 (full) accepts raw strings directly — no SentencePiece needed.
# USE Lite v2 would require SentencePiece tokenization which is not implemented.
USE_LITE_URL: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
EMBEDDING_DIM: int = 512

# ── Domain index mapping ─────────────────────────────────────────────────────
DOMAIN_LABELS: dict[int, str] = {
    0: "IT / Software",
    1: "Non-IT / Management",
    2: "Design / Creative",
    3: "Healthcare",
    4: "Finance / Banking",
    5: "Legal",
    6: "Education",
}
NUM_DOMAINS: int = len(DOMAIN_LABELS)
DOMAIN_NAME_TO_INDEX: dict[str, int] = {v: k for k, v in DOMAIN_LABELS.items()}

# Maps LiveCareer raw Category strings → model domain index
LIVECARER_CATEGORY_MAP: dict[str, int] = {
    # IT / Software (domain 0)
    "Java Developer": 0, "Python Developer": 0, "Web Designing": 0,
    "DevOps Engineer": 0, "Data Science": 0, "Database": 0,
    "Hadoop": 0, "ETL Developer": 0, "Blockchain": 0,
    "Network Security Engineer": 0, "SAP Developer": 0,
    "Testing": 0, "DotNet Developer": 0,
    "INFORMATION-TECHNOLOGY": 0, "BPO": 0,
    # Non-IT / Management (domain 1)
    "Business Analyst": 1, "HR": 1, "PMO": 1, "Operations Manager": 1,
    "BUSINESS-DEVELOPMENT": 1, "CONSULTANT": 1, "CONSTRUCTION": 1,
    "PUBLIC-RELATIONS": 1, "SALES": 1, "ENGINEERING": 1,
    "AVIATION": 1, "AGRICULTURE": 1, "AUTOMOBILE": 1, "APPAREL": 1,
    "CHEF": 1,
    # Design / Creative (domain 2)
    "Arts": 2, "Designing": 2,
    "DESIGNER": 2, "ARTS": 2, "DIGITAL-MEDIA": 2,
    # Healthcare (domain 3)
    "Health and fitness": 3,
    "HEALTHCARE": 3, "FITNESS": 3,
    # Finance / Banking (domain 4)
    "Accountant": 4, "Finance": 4, "Banking": 4,
    "ACCOUNTANT": 4, "FINANCE": 4, "BANKING": 4,
    # Legal (domain 5)
    "Advocate": 5,
    "ADVOCATE": 5,
    # Education (domain 6)
    "Teacher": 6,
    "TEACHER": 6,
    # Legacy mixed-case mappings for non-CSV sources
    "Civil Engineer": 1, "Mechanical Engineer": 1,
    "Electrical Engineering": 1, "Sales": 1, "Public Relations": 1,
    "Aviation": 1, "Agriculture": 1,
}

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int = 32
EPOCHS: int = 60
LEARNING_RATE: float = 1e-4
SCORE_LOSS_WEIGHT: float = 0.35      # Increased — prevents MAE regression
DOMAIN_LOSS_WEIGHT: float = 0.65     # Reduced slightly — was over-rotating on domain
DOMAIN_CLASS_WEIGHTS: dict[int, float] = {
    0: 1.4,   # IT — needs more gradient signal
    1: 0.8,   # Non-IT — well represented, reduce weight
    2: 0.9,   # Design — strong, slight reduction
    3: 1.0,   # Healthcare — stable
    4: 1.5,   # Finance — needs more gradient signal
    5: 0.9,   # Legal — now well represented
    6: 1.0,   # Education — now well represented
}
VALIDATION_SPLIT: float = 0.15
TEST_SPLIT: float = 0.10
RANDOM_SEED: int = 42
EARLY_STOPPING_PATIENCE: int = 10
MIN_PAIRS_PER_DOMAIN: int = 150
MAX_TEXT_LENGTH: int = 512  # token cap for encoder

# ── Target metrics (pass/fail gates) ─────────────────────────────────────────
TARGET_MAE: float = 8.0          # on 0–100 scale
TARGET_DOMAIN_F1: float = 0.85
MAX_MODEL_SIZE_MB: float = 600.0
TFLITE_PARITY_TOLERANCE: float = 0.02   # on 0.0–1.0 scale

# ── Score bands ───────────────────────────────────────────────────────────────
SCORE_BANDS: list[tuple[int, int, str]] = [
    (85, 100, "Excellent Match"),
    (65, 84,  "Good Match"),
    (45, 64,  "Moderate Match"),
    (25, 44,  "Weak Match"),
    (0,  24,  "Poor Match"),
]


def get_score_band(score: float) -> str:
    """Return the score band label for a given ATS score (0–100).

    Args:
        score: ATS score between 0.0 and 100.0.

    Returns:
        Band label string (e.g. "Good Match").
    """
    score_int = int(round(score))
    for low, high, label in SCORE_BANDS:
        if low <= score_int <= high:
            return label
    return "Poor Match"
