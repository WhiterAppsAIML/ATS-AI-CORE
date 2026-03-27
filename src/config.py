"""
config.py — Central configuration for all ATS AI Core modules.
All hyperparameters, paths, and constants live here.
Never hardcode these values in model or training files.
"""
from pathlib import Path

# ── Paths ──
ROOT_DIR        = Path(__file__).parent.parent.resolve()
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
LABELED_DIR     = DATA_DIR / "labeled"
SYNTHETIC_DIR   = DATA_DIR / "synthetic"
MODEL_DIR       = ROOT_DIR / "model"
ATS_MODEL_DIR   = MODEL_DIR / "ats_model"
TFLITE_DIR      = MODEL_DIR / "tflite"
RUBRICS_DIR     = ROOT_DIR / "rubrics"

# ── Encoder ──
USE_LITE_URL    = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
EMBEDDING_DIM   = 512

# ── Domain labels ──
DOMAIN_LABELS: dict[int, str] = {
    0: "IT / Software",
    1: "Non-IT / Management",
    2: "Design / Creative",
    3: "Healthcare",
    4: "Finance / Banking",
    5: "Legal",
    6: "Education",
}
NUM_DOMAINS = len(DOMAIN_LABELS)

# ── Training ──
BATCH_SIZE          = 32
EPOCHS              = 50
LEARNING_RATE       = 1e-4
SCORE_LOSS_WEIGHT   = 0.7
DOMAIN_LOSS_WEIGHT  = 0.3
VALIDATION_SPLIT    = 0.15
TEST_SPLIT          = 0.10
RANDOM_SEED         = 42

# ── TFLite ──
TFLITE_OUTPUT_PATH  = TFLITE_DIR / "ats_core.tflite"
MAX_MODEL_SIZE_MB   = 30
TFLITE_PARITY_TOL   = 0.02   # Max allowed diff between Keras and TFLite output

# ── Score bands ──
SCORE_BANDS: list[tuple[int, int, str]] = [
    (85, 100, "Excellent Match"),
    (65, 84,  "Good Match"),
    (45, 64,  "Moderate Match"),
    (25, 44,  "Weak Match"),
    (0,  24,  "Poor Match"),
]
