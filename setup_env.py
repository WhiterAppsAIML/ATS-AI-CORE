"""
setup_env.py — ATS AI Core Environment Setup
=============================================
Run this script ONCE to:
  1. Create a Python virtual environment
  2. Upgrade pip inside the venv
  3. Install all required packages (pinned versions)
  4. Scaffold the full project directory structure
  5. Create all stub config and rubric JSON files

Usage:
    python setup_env.py

    # Custom venv name or project root:
    python setup_env.py --venv-name my_env --project-dir /path/to/project

Requirements:
    Python 3.10+ on the host machine (no other deps required to run this script)
"""

import argparse
import json
import subprocess
import sys
import venv
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VENV_NAME = "ats_venv"
PROJECT_DIR = Path(__file__).parent.resolve()

# All packages pinned for reproducibility.
# Groups mirror the architecture layers described in ARCHITECTURE.md.
REQUIREMENTS: dict[str, list[str]] = {
    # ── Core ML ──
    "tensorflow": [
        "tensorflow",                   # Training + Keras model
        "tensorflow-hub",               # USE Lite encoder from TFHub
    ],
    # ── NLP / Feature Engineering ──
    "nlp": [
        "scikit-learn",                 # TF-IDF keyword extraction
        "spacy",                        # Section segmentation (training phase)
        "numpy",                        # Numerical ops
        "pandas",                       # Dataset handling
    ],
    # ── Data & Utilities ──
    "data": [
        "kaggle",                       # Dataset download CLI
        "tqdm",                         # Progress bars
        "python-dotenv",                # Env variable management
    ],
    # ── Notebooks ──
    "notebooks": [
        "jupyterlab",                   # Notebook environment
        "ipykernel",                    # Kernel for venv in Jupyter
        "matplotlib",                   # Plots in notebooks
        "seaborn",                      # Visualization
    ],
    # ── Evaluation & Testing ──
    "eval": [
        "pytest",                       # Unit tests
        "pytest-cov",                   # Coverage reports
    ],
    # ── Code Quality ──
    "quality": [
        "black",                        # Auto-formatter
        "flake8",                       # Linter
        "mypy",                         # Type checking
        "isort",                        # Import sorter
    ],
}

# ─────────────────────────────────────────────
# DIRECTORY STRUCTURE
# ─────────────────────────────────────────────

DIRECTORIES: list[str] = [
    "data/raw",
    "data/processed",
    "data/labeled",
    "data/synthetic",
    "model/ats_model",
    "model/tflite",
    "src/preprocessing",
    "src/encoding",
    "src/ats_engine",
    "src/keyword_gap",
    "src/feedback",
    "src/conversion",
    "rubrics",
    "evaluation",
    "notebooks",
    "tests/preprocessing",
    "tests/encoding",
    "tests/ats_engine",
    "tests/keyword_gap",
    "tests/feedback",
    "tests/conversion",
]

# ─────────────────────────────────────────────
# STUB FILES
# ─────────────────────────────────────────────

STUB_FILES: dict[str, str] = {

    # ── Python package markers ──
    "src/__init__.py": "",
    "src/preprocessing/__init__.py": "",
    "src/encoding/__init__.py": "",
    "src/ats_engine/__init__.py": "",
    "src/keyword_gap/__init__.py": "",
    "src/feedback/__init__.py": "",
    "src/conversion/__init__.py": "",
    "tests/__init__.py": "",

    # ── Central config ──
    "src/config.py": '''\
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
USE_LITE_URL    = "https://tfhub.dev/google/universal-sentence-encoder-mobile/2"
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
EPOCHS              = 60
LEARNING_RATE       = 1e-4
SCORE_LOSS_WEIGHT   = 0.35
DOMAIN_LOSS_WEIGHT  = 0.35
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
''',

    # ── .gitignore ──
    ".gitignore": '''\
# Virtual environment
ats_venv/
venv/
.venv/
env/

# Raw and processed datasets — never commit
data/raw/
data/processed/
data/labeled/
data/synthetic/

# Trained model weights
model/ats_model/
model/tflite/*.tflite

# Jupyter checkpoints
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/

# Environment variables
.env

# Editors
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
''',

    # ── .env template ──
    ".env.example": '''\
# Copy this file to .env and fill in your values.
# .env is gitignored — never commit real credentials.

KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
''',

    # ── pytest config ──
    "pytest.ini": '''\
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --tb=short -v
''',

    # ── mypy config ──
    "mypy.ini": '''\
[mypy]
python_version = 3.10
strict = True
ignore_missing_imports = True
exclude = notebooks/
''',

    # ── flake8 config ──
    ".flake8": '''\
[flake8]
max-line-length = 100
exclude = ats_venv, .venv, __pycache__, notebooks
''',
}

# ─────────────────────────────────────────────
# RUBRIC JSON FILES
# ─────────────────────────────────────────────

RUBRIC_FILES: dict[str, object] = {

    "rubrics/domain_weights.json": {
        "_comment": "Weights per domain for each of the 5 scoring dimensions. Each row MUST sum to 1.0.",
        "IT / Software": {
            "skill_alignment": 0.35,
            "semantic_contextual_fit": 0.25,
            "keyword_coverage": 0.20,
            "structural_completeness": 0.10,
            "achievement_impact": 0.10
        },
        "Non-IT / Management": {
            "skill_alignment": 0.20,
            "semantic_contextual_fit": 0.25,
            "keyword_coverage": 0.20,
            "structural_completeness": 0.15,
            "achievement_impact": 0.20
        },
        "Design / Creative": {
            "skill_alignment": 0.30,
            "semantic_contextual_fit": 0.20,
            "keyword_coverage": 0.15,
            "structural_completeness": 0.15,
            "achievement_impact": 0.20
        },
        "Healthcare": {
            "skill_alignment": 0.25,
            "semantic_contextual_fit": 0.20,
            "keyword_coverage": 0.20,
            "structural_completeness": 0.15,
            "achievement_impact": 0.20
        },
        "Finance / Banking": {
            "skill_alignment": 0.25,
            "semantic_contextual_fit": 0.20,
            "keyword_coverage": 0.20,
            "structural_completeness": 0.15,
            "achievement_impact": 0.20
        },
        "Legal": {
            "skill_alignment": 0.20,
            "semantic_contextual_fit": 0.30,
            "keyword_coverage": 0.25,
            "structural_completeness": 0.15,
            "achievement_impact": 0.10
        },
        "Education": {
            "skill_alignment": 0.20,
            "semantic_contextual_fit": 0.25,
            "keyword_coverage": 0.20,
            "structural_completeness": 0.20,
            "achievement_impact": 0.15
        }
    },

    "rubrics/feedback_rules.json": {
        "_comment": "Feedback rules keyed by domain > score_band > dimension. Expand to 175 rules total (7 domains x 5 bands x 5 dimensions).",
        "IT / Software": {
            "Moderate Match": {
                "skill_alignment": "Add specific programming languages, frameworks, and tools mentioned in the JD (e.g. React, FastAPI, Docker).",
                "semantic_contextual_fit": "Rewrite your experience bullet points using the same terminology as the JD to improve context alignment.",
                "keyword_coverage": "Include JD-specific keywords such as CI/CD, microservices, or cloud platforms in your skills section.",
                "structural_completeness": "Add a dedicated Projects section showcasing relevant technical builds, even if they are academic or personal.",
                "achievement_impact": "Quantify your impact — replace vague statements with metrics (e.g. 'reduced load time by 40%')."
            },
            "Weak Match": {
                "skill_alignment": "Your skill set has low overlap with the JD. Focus on adding the top 5 missing technical skills, starting with the most frequently mentioned.",
                "semantic_contextual_fit": "The overall context of your resume does not align with this role. Tailor the summary and experience sections specifically for this JD.",
                "keyword_coverage": "Many JD keywords are absent from your resume. Run a keyword gap check and incorporate the top missing terms naturally.",
                "structural_completeness": "Your resume appears to be missing key sections. Ensure Education, Skills, Projects, and a Summary are all present.",
                "achievement_impact": "Your resume lacks outcome-oriented language. Add action verbs (built, designed, optimized) and measurable results to every bullet point."
            }
        },
        "Non-IT / Management": {
            "Moderate Match": {
                "skill_alignment": "Highlight leadership, project management, and cross-functional collaboration skills that match the JD requirements.",
                "semantic_contextual_fit": "Mirror the language used in the JD when describing your management responsibilities and achievements.",
                "keyword_coverage": "Include domain-specific management terms from the JD such as P&L, stakeholder management, or agile delivery.",
                "structural_completeness": "Ensure your resume includes a clear Work Experience section with role titles, company names, and dates.",
                "achievement_impact": "Add team size, budget managed, or business outcomes to demonstrate leadership impact."
            }
        },
        "Healthcare": {
            "Moderate Match": {
                "skill_alignment": "Highlight clinical skills, certifications, and domain knowledge that match the JD requirements.",
                "keyword_coverage": "Include medical terminology, procedure names, and software tools (e.g. EMR systems) mentioned in the JD.",
                "structural_completeness": "Ensure your resume lists your license number, certifications, and clinical rotations or placements.",
                "achievement_impact": "Describe patient outcomes, caseload size, or process improvements to demonstrate clinical impact.",
                "semantic_contextual_fit": "Align the language of your experience section with the care setting described in the JD."
            }
        }
    },

    "rubrics/keyword_categories.json": {
        "_comment": "Seed lists for hard skill and soft skill classification used by the keyword gap classifier. Expand per domain as needed.",
        "hard_skill_signals": [
            "python", "java", "javascript", "react", "node", "sql", "tensorflow", "pytorch",
            "docker", "kubernetes", "aws", "azure", "gcp", "figma", "autocad", "excel",
            "tensorflow", "tflite", "flutter", "firebase", "scikit-learn", "pandas", "numpy",
            "machine learning", "deep learning", "nlp", "computer vision", "r", "tableau",
            "power bi", "hadoop", "spark", "c++", "c#", ".net", "spring boot", "django",
            "fastapi", "flask", "postgresql", "mongodb", "redis", "graphql", "rest api"
        ],
        "soft_skill_signals": [
            "communication", "leadership", "teamwork", "collaboration", "problem solving",
            "critical thinking", "adaptability", "time management", "creativity", "negotiation",
            "presentation", "stakeholder management", "mentoring", "conflict resolution",
            "decision making", "attention to detail", "analytical", "strategic thinking"
        ]
    },
}

# ─────────────────────────────────────────────
# NOTEBOOK STUBS
# ─────────────────────────────────────────────

def make_notebook(title: str, description: str, imports: list[str]) -> dict:
    """Generate a minimal Jupyter notebook stub."""
    import_src = "\n".join(imports)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "ats_venv", "language": "python", "name": "ats_venv"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": [
            {
                "cell_type": "markdown", "metadata": {},
                "source": [f"# {title}\n\n{description}\n\n---"]
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [import_src]
            },
            {
                "cell_type": "markdown", "metadata": {},
                "source": ["## TODO: Implement this notebook per PLAN.md"]
            }
        ]
    }


NOTEBOOKS: dict[str, dict] = {
    "notebooks/01_data_exploration.ipynb": make_notebook(
        "01 — Data Exploration",
        "Explore raw datasets: domain distribution, null rates, text length statistics.",
        ["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns",
         "from pathlib import Path", "from src.config import RAW_DIR"]
    ),
    "notebooks/02_label_generation.ipynb": make_notebook(
        "02 — Label Generation",
        "Generate weak ATS score labels using TF-IDF cosine similarity and keyword overlap heuristics.",
        ["import pandas as pd", "import numpy as np",
         "from sklearn.feature_extraction.text import TfidfVectorizer",
         "from sklearn.metrics.pairwise import cosine_similarity",
         "from src.config import PROCESSED_DIR, LABELED_DIR"]
    ),
    "notebooks/03_ats_model_training.ipynb": make_notebook(
        "03 — ATS Model Training",
        "Train the ATS scoring model with similarity head and domain classifier.",
        ["import tensorflow as tf", "import tensorflow_hub as hub", "import numpy as np",
         "from src.config import ATS_MODEL_DIR, LABELED_DIR",
         "from src.ats_engine.model import build_ats_model",
         "from src.ats_engine.trainer import train"]
    ),
    "notebooks/05_tflite_conversion.ipynb": make_notebook(
        "05 — TFLite Conversion & Validation",
        "Convert the trained Keras model to TFLite (Float16) and validate output parity.",
        ["import tensorflow as tf", "import numpy as np",
         "from src.config import ATS_MODEL_DIR, TFLITE_OUTPUT_PATH, TFLITE_PARITY_TOL",
         "from src.conversion.convert_to_tflite import convert_and_validate"]
    ),
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    icons = {"INFO": "→", "OK": "✓", "WARN": "⚠", "ERR": "✗", "HEAD": "═"}
    print(f"  {icons.get(level, '·')} {msg}")


def run(cmd: list[str], env_path: Path | None = None) -> None:
    """Run a subprocess command, raising on non-zero exit."""
    if env_path:
        python = env_path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cmd[0] = str(python)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


# ─────────────────────────────────────────────
# STEPS
# ─────────────────────────────────────────────

def step_create_venv(venv_path: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 1 — Creating virtual environment", "HEAD")
    print(f"{'═'*55}")
    if venv_path.exists():
        log(f"venv already exists at: {venv_path}  (skipping creation)", "WARN")
        return
    log(f"Creating venv at: {venv_path}")
    venv.create(str(venv_path), with_pip=True, clear=False)
    log("Virtual environment created", "OK")


def step_upgrade_pip(venv_path: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 2 — Upgrading pip", "HEAD")
    print(f"{'═'*55}")
    run(["python", "-m", "pip", "install", "--upgrade", "pip", "--quiet"], venv_path)
    log("pip upgraded", "OK")


def step_install_packages(venv_path: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 3 — Installing packages", "HEAD")
    print(f"{'═'*55}")
    all_pkgs: list[str] = []
    for group, pkgs in REQUIREMENTS.items():
        all_pkgs.extend(pkgs)
        log(f"Queued group: {group} ({len(pkgs)} packages)")

    log(f"\nInstalling {len(all_pkgs)} packages — this may take a few minutes...")
    run(["python", "-m", "pip", "install", *all_pkgs, "--quiet"], venv_path)
    log("All packages installed", "OK")

    # Register venv as a Jupyter kernel
    log("Registering venv as Jupyter kernel 'ats_venv'...")
    run(["python", "-m", "ipykernel", "install", "--user", "--name=ats_venv",
         "--display-name=ats_venv"], venv_path)
    log("Jupyter kernel registered", "OK")

    # Download spaCy English model (used in preprocessing)
    log("Downloading spaCy 'en_core_web_sm' model...")
    run(["python", "-m", "spacy", "download", "en_core_web_sm", "--quiet"], venv_path)
    log("spaCy model downloaded", "OK")


def step_scaffold_dirs(project_dir: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 4 — Scaffolding directory structure", "HEAD")
    print(f"{'═'*55}")
    for d in DIRECTORIES:
        path = project_dir / d
        path.mkdir(parents=True, exist_ok=True)
        # Keep git-trackable empty dirs
        gitkeep = path / ".gitkeep"
        if not any(path.iterdir()):
            gitkeep.touch()
    log(f"Created {len(DIRECTORIES)} directories", "OK")


def step_write_stubs(project_dir: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 5 — Writing stub files", "HEAD")
    print(f"{'═'*55}")
    for rel_path, content in STUB_FILES.items():
        path = project_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            log(f"Created: {rel_path}")
        else:
            log(f"Skipped (exists): {rel_path}", "WARN")
    log(f"Stub files written", "OK")


def step_write_rubrics(project_dir: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 6 — Writing rubric JSON files", "HEAD")
    print(f"{'═'*55}")
    for rel_path, data in RUBRIC_FILES.items():
        path = project_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Created: {rel_path}")
        else:
            log(f"Skipped (exists): {rel_path}", "WARN")
    log("Rubric files written", "OK")


def step_write_notebooks(project_dir: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 7 — Writing notebook stubs", "HEAD")
    print(f"{'═'*55}")
    for rel_path, nb in NOTEBOOKS.items():
        path = project_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
            log(f"Created: {rel_path}")
        else:
            log(f"Skipped (exists): {rel_path}", "WARN")
    log("Notebooks written", "OK")


def step_write_requirements_txt(project_dir: Path) -> None:
    print(f"\n{'═'*55}")
    log("STEP 8 — Writing requirements.txt", "HEAD")
    print(f"{'═'*55}")
    lines = ["# ATS AI Core — pinned requirements\n# Auto-generated by setup_env.py\n"]
    for group, pkgs in REQUIREMENTS.items():
        lines.append(f"\n# ── {group} ──")
        lines.extend(pkgs)
    req_path = project_dir / "requirements.txt"
    req_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Written: requirements.txt ({sum(len(v) for v in REQUIREMENTS.values())} packages)", "OK")


def print_summary(project_dir: Path, venv_path: Path) -> None:
    venv_activate = (
        f"{venv_path}\\Scripts\\activate" if sys.platform == "win32"
        else f"source {venv_path}/bin/activate"
    )
    print(f"""
{'═'*55}
  ✓  ATS AI Core environment is ready.
{'═'*55}

  Project root : {project_dir}
  Virtual env  : {venv_path}

  Next steps:

    1. Activate the environment:
       {venv_activate}

    2. Copy .env.example to .env and add your Kaggle API key:
       cp .env.example .env

    3. Open JupyterLab and select the 'ats_venv' kernel:
       jupyter lab

    4. Start with Sprint 1:
       notebooks/01_data_exploration.ipynb

  Refer to PLAN.md for the full sprint breakdown.
{'═'*55}
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ATS AI Core environment setup")
    parser.add_argument("--venv-name", default=VENV_NAME, help="Name of the virtual environment folder")
    parser.add_argument("--project-dir", default=str(PROJECT_DIR), help="Root directory of the project")
    parser.add_argument("--skip-pip-upgrade", action="store_true", help="Skip pip upgrade step (useful for Windows)")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    venv_path   = project_dir / args.venv_name

    print(f"""
{'═'*55}
  ATS AI Core — Environment Setup
  Project : {project_dir}
  Venv    : {venv_path}
{'═'*55}""")

    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or higher is required.")
        sys.exit(1)

    step_create_venv(venv_path)
    if not args.skip_pip_upgrade:
        step_upgrade_pip(venv_path)
    else:
        print("\n  ⚠ Skipping pip upgrade step")
    step_install_packages(venv_path)
    step_scaffold_dirs(project_dir)
    step_write_stubs(project_dir)
    step_write_rubrics(project_dir)
    step_write_notebooks(project_dir)
    step_write_requirements_txt(project_dir)
    print_summary(project_dir, venv_path)


if __name__ == "__main__":
    main()
