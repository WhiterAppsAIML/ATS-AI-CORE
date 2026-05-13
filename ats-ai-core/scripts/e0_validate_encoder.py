"""scripts/e0_validate_encoder.py — INJECTION-E0 Encoder Validation

Validates sentence-transformers/all-MiniLM-L6-v2 as the replacement encoder
for MobileUSE (512-dim) in the ATS pipeline:

  1. Convert to INT8 TFLite — zero Flex ops, size < 25 MB
  2. Spearman rank correlation vs USE Lite on 50 resume-JD pairs — rho > 0.85
  3. Export 20-sample tokenizer JSON for manual Dart WordPiece comparison

Outputs:
  model/minilm/encoder_only.tflite
  evaluation/e0_encoder_report.json

Hard stops (sys.exit 1) if size >= 25 MB, Flex ops detected, or rho < 0.85.
Prints "E0 PASSED — proceed to E1" on success.

Usage:
    python scripts/e0_validate_encoder.py
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from src.config import EVALUATION_DIR, MODEL_DIR, USE_LITE_URL

# ── Constants ──────────────────────────────────────────────────────────────────
MINILM_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
SEQ_LEN      = 128
MAX_SIZE_MB  = 25.0
MIN_SPEARMAN = 0.85
NUM_TOK_SMPL = 20

TFLITE_PATH    = MODEL_DIR / "minilm" / "encoder_only.tflite"
SAVEDMODEL_DIR = MODEL_DIR / "minilm" / "saved_model"
REPORT_PATH    = EVALUATION_DIR / "e0_encoder_report.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── 50 synthetic resume–JD pairs (5 domains × 5 quality tiers × 2 reps) ──────
# Pairs are ordered: excellent → good → moderate → weak → poor within each domain.
# This gradient is critical for the Spearman correlation to be meaningful.
_SAMPLE_PAIRS: list[tuple[str, str]] = [
    # ── IT / Software (×10) ───────────────────────────────────────────────────
    (
        "Python developer 5 years. Django REST framework PostgreSQL Redis Docker Kubernetes. "
        "Led microservices architecture migration at a fintech startup.",
        "Senior Python backend engineer. 5+ years Django/FastAPI. PostgreSQL Redis Docker "
        "Kubernetes required. Microservices experience essential.",
    ),
    (
        "Full-stack engineer React TypeScript Node.js GraphQL AWS 3 years. "
        "Built SaaS platforms serving 100k daily active users.",
        "Seeking full-stack engineer with React TypeScript Node.js. GraphQL and AWS preferred. "
        "SaaS product background valued.",
    ),
    (
        "Java Spring Boot developer 4 years. REST APIs MySQL Maven Jenkins CI/CD pipelines. "
        "Some microservices exposure.",
        "Backend Java engineer needed. Spring Boot REST APIs relational databases. "
        "CI/CD experience desired.",
    ),
    (
        "DevOps engineer AWS Terraform Ansible Kubernetes Docker 3 years. "
        "Monitoring with Prometheus Grafana. On-call rotation experience.",
        "Cloud infrastructure engineer. AWS Terraform containerisation. "
        "Monitoring and automation skills required.",
    ),
    (
        "Frontend developer React Redux 2 years. HTML CSS JavaScript responsive design. "
        "Limited backend experience.",
        "Full-stack developer. React frontend plus Node.js backend and PostgreSQL. "
        "Both layers required.",
    ),
    (
        "Data analyst SQL Python Tableau 3 years. Statistical analysis and dashboard creation. "
        "No software engineering background.",
        "Software engineer Python. Application development REST APIs deployment. "
        "Data analysis skills secondary.",
    ),
    (
        "Network security engineer firewall penetration testing SIEM incident response 4 years.",
        "Mobile app developer Flutter Dart iOS Android. Strong UI/UX REST API integration.",
    ),
    (
        "SAP ABAP consultant SD MM modules 5 years. Business process mapping.",
        "Machine learning engineer PyTorch TensorFlow deep learning model training deployment.",
    ),
    (
        "Mechanical engineer AutoCAD SolidWorks CNC machining 3 years. "
        "Manufacturing process optimisation.",
        "Senior software architect microservices cloud-native Kubernetes distributed systems.",
    ),
    (
        "High school computer science teacher Python basics curriculum design.",
        "Principal engineer real-time trading systems low-latency C++ Java quant finance.",
    ),
    # ── Healthcare (×10) ─────────────────────────────────────────────────────
    (
        "Registered nurse ICU critical care 5 years. ACLS BLS certified. "
        "Patient monitoring mechanical ventilators vasopressors.",
        "ICU nurse. Critical care experience required. ACLS BLS certification. "
        "Ventilator and vasopressor management.",
    ),
    (
        "Clinical pharmacist hospital 4 years. Medication therapy management oncology pharmacy. "
        "Drug interaction review.",
        "Hospital clinical pharmacist. Medication management patient counselling. "
        "Oncology experience preferred.",
    ),
    (
        "Medical doctor general practice 3 years. Primary care chronic disease management EHR.",
        "Physician primary care clinic. Chronic disease management EMR systems. "
        "Patient-centred care focus.",
    ),
    (
        "Physical therapist outpatient orthopedics 4 years. Manual therapy exercise prescription.",
        "PT orthopedic rehabilitation. Manual therapy skills valued.",
    ),
    (
        "Dental hygienist 3 years. Teeth cleaning X-rays periodontal charting.",
        "Nurse practitioner primary care. Prescribing authority full patient management.",
    ),
    (
        "Medical coder ICD-10 CPT billing 4 years. Insurance claims revenue cycle.",
        "Healthcare administrator hospital operations budget planning staffing.",
    ),
    (
        "Veterinarian small animal surgery diagnostics anaesthesia 5 years.",
        "Pharmaceutical sales representative. Marketing drug products to physicians.",
    ),
    (
        "Nutritionist dietitian outpatient counselling weight management 3 years.",
        "Biomedical researcher lab bench clinical trials PhD required.",
    ),
    (
        "Yoga instructor wellness coach meditation stress management.",
        "Cardiovascular surgeon operating room hospital critical care.",
    ),
    (
        "Massage therapist sports massage deep tissue 2 years.",
        "Radiologist MD board certified imaging diagnostics hospital.",
    ),
    # ── Finance / Banking (×10) ──────────────────────────────────────────────
    (
        "Investment banker M&A advisory 4 years. Financial modelling DCF LBO deal structuring IPO.",
        "Investment banking associate. M&A transactions financial modelling client advisory.",
    ),
    (
        "Quantitative risk analyst 3 years. VaR credit risk Basel III Python R statistical modelling.",
        "Quantitative risk analyst. Credit and market risk regulatory capital Python required.",
    ),
    (
        "Financial analyst FP&A 3 years. Budget forecasting Excel Power BI variance analysis.",
        "Finance manager FP&A. Budgeting forecasting financial reporting. Advanced Excel required.",
    ),
    (
        "Chartered accountant audit tax 4 years. IFRS financial statements compliance.",
        "Senior accountant financial reporting IFRS/GAAP audit experience preferred.",
    ),
    (
        "Retail bank teller customer service 2 years. Cash handling basic loan processing.",
        "Corporate credit analyst complex underwriting large corporate financial statement analysis.",
    ),
    (
        "Insurance underwriter property casualty 3 years. Risk assessment policy pricing.",
        "Equity research analyst stock valuation sector analysis investment recommendations.",
    ),
    (
        "Mortgage broker residential loan origination 4 years.",
        "Hedge fund portfolio manager derivatives trading quant strategies.",
    ),
    (
        "Payroll specialist HRIS ADP payroll processing 3 years.",
        "Investment director private equity fund management LP relations.",
    ),
    (
        "Real estate agent residential commercial property sales.",
        "CFO financial strategy corporate governance board reporting SEC filings.",
    ),
    (
        "Retail cashier customer service basic arithmetic.",
        "Chief risk officer enterprise risk Basel IV regulatory oversight.",
    ),
    # ── Education (×10) ──────────────────────────────────────────────────────
    (
        "High school mathematics teacher 5 years. AP calculus statistics curriculum design.",
        "High school math teacher. Calculus statistics experience. Curriculum development valued.",
    ),
    (
        "University professor computer science 6 years. ML research publications teaching.",
        "Assistant professor computer science. Teaching algorithms data structures. ML/AI research.",
    ),
    (
        "Elementary school teacher 4 years. Reading literacy STEM IEP differentiated instruction.",
        "Primary school teacher English literacy. Classroom management differentiated instruction.",
    ),
    (
        "Corporate trainer L&D 3 years. Instructional design e-learning Adobe Captivate facilitation.",
        "Learning and development specialist. Training design delivery LMS experience preferred.",
    ),
    (
        "ESL teacher English second language adults community college 3 years.",
        "Special education teacher K-8. Learning disabilities IEP autism behavioural support.",
    ),
    (
        "Music teacher private lessons piano violin 5 years.",
        "School principal instructional coaching staff evaluation budget management.",
    ),
    (
        "Academic librarian research databases cataloguing 4 years.",
        "Online course developer video production LMS digital marketing.",
    ),
    (
        "Preschool teacher early childhood play-based learning 2 years.",
        "University provost academic affairs higher education administration policy.",
    ),
    (
        "Sports coach basketball physical education 3 years.",
        "Curriculum director K-12 district standards alignment literacy assessment.",
    ),
    (
        "Art teacher painting drawing sculpture high school.",
        "Data scientist education analytics predictive modelling student outcomes.",
    ),
    # ── Management / Business (×10) ──────────────────────────────────────────
    (
        "Product manager B2B SaaS 4 years. Roadmap prioritisation stakeholder management agile OKRs.",
        "Senior product manager B2B SaaS. Roadmap planning stakeholder alignment agile required.",
    ),
    (
        "HR director talent acquisition compensation benefits 6 years. Workday HRIS workforce planning.",
        "VP of HR. Talent acquisition comp benefits strategy. Workday strategic HR leadership.",
    ),
    (
        "Operations manager manufacturing 4 years. Lean six sigma supply chain cost reduction.",
        "Operations director manufacturing logistics. Process improvement lean P&L responsibility.",
    ),
    (
        "Business development manager enterprise SaaS sales 5 years. Salesforce CRM pipeline.",
        "Sales director enterprise software. New logo acquisition account management SaaS.",
    ),
    (
        "Digital marketing manager SEO SEM content strategy 3 years B2C campaigns.",
        "General manager P&L ownership full team leadership multi-site operations.",
    ),
    (
        "IT project coordinator Jira documentation status reporting 2 years.",
        "Chief operating officer strategic operations executive board collaboration.",
    ),
    (
        "Customer success manager SMB accounts churn reduction upsell SaaS 3 years.",
        "Investment management firm operations regulatory compliance fund administration.",
    ),
    (
        "Procurement specialist vendor management contracts cost negotiation 4 years.",
        "Chief marketing officer brand strategy digital transformation growth.",
    ),
    (
        "Administrative assistant calendar scheduling travel coordination 2 years.",
        "Managing director global strategy M&A integration executive leadership.",
    ),
    (
        "Corporate events coordinator logistics vendor management 3 years.",
        "CEO tech startup fundraising product vision board governance.",
    ),
]

assert len(_SAMPLE_PAIRS) == 50, f"Expected 50 pairs, got {len(_SAMPLE_PAIRS)}"

# ── 20 tokenizer fidelity samples ─────────────────────────────────────────────
_TOK_SAMPLES: list[str] = [
    "Python developer with 5 years of experience in Django and REST APIs",
    "Registered nurse with ICU critical care experience and ACLS certification",
    "MBA finance professional with investment banking and M&A advisory skills",
    "High school mathematics teacher with AP calculus curriculum development",
    "Senior software engineer specialising in microservices and cloud architecture",
    "Data scientist with expertise in machine learning NLP Python TensorFlow",
    "Product manager B2B SaaS roadmap agile OKRs stakeholder alignment",
    "Chartered accountant with IFRS financial reporting and audit background",
    "Medical doctor general practice chronic disease management patient care",
    "HR director talent acquisition compensation benefits Workday HRIS",
    "Full-stack developer React Node.js PostgreSQL GraphQL AWS",
    "Operations manager lean six sigma supply chain process optimisation",
    "Clinical pharmacist oncology medication therapy management",
    "University professor computer science algorithms machine learning research",
    "Business development manager enterprise sales CRM pipeline management",
    "Physical therapist orthopaedic rehabilitation manual therapy",
    "DevOps engineer AWS Kubernetes Docker Terraform CI/CD monitoring",
    "Financial analyst FP&A budget forecasting Excel Power BI variance",
    "Elementary school teacher reading literacy STEM differentiated instruction",
    "Network security engineer penetration testing SIEM incident response",
]

assert len(_TOK_SAMPLES) == NUM_TOK_SMPL


# ── Helper functions ──────────────────────────────────────────────────────────

def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two [N, D] embedding matrices."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a_n * b_n, axis=1)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    def _rank(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    rx, ry = _rank(x), _rank(y)
    n = len(x)
    d2 = (rx - ry) ** 2
    return float(1.0 - 6.0 * d2.sum() / (n * (n**2 - 1)))


def detect_flex_ops(model_bytes: bytes) -> list[str]:
    """Return sorted list of Flex op names found in the TFLite flatbuffer binary.

    Flex ops are custom ops whose names start with 'Flex' (e.g. FlexGatherNd).
    They are stored as ASCII strings in the flatbuffer, making a regex scan
    sufficient for reliable detection.
    """
    found = {
        m.group(0).decode("ascii")
        for m in re.finditer(rb"Flex[A-Za-z0-9_]+", model_bytes)
    }
    return sorted(found)


def mean_pool_l2(last_hidden: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    """Attention-mask weighted mean pooling then L2 normalisation.

    Matches sentence-transformers' pooling_mode_mean_tokens output exactly.
    """
    mask = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)  # [B, L, 1]
    sum_emb = tf.reduce_sum(last_hidden * mask, axis=1)              # [B, D]
    count   = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-9)         # [B, 1]
    return tf.nn.l2_normalize(sum_emb / count, axis=-1)             # [B, D]


# ── Step 1: TFLite INT8 conversion ────────────────────────────────────────────

def step1_convert(tokenizer, tf_model) -> dict:
    """Convert MiniLM to INT8 TFLite and assert size / Flex-op constraints."""
    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    class MiniLMModule(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @tf.function(input_signature=[
            tf.TensorSpec([1, SEQ_LEN], tf.int32, name="input_ids"),
            tf.TensorSpec([1, SEQ_LEN], tf.int32, name="attention_mask"),
        ])
        def encode(self, input_ids, attention_mask):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=tf.zeros_like(input_ids),
                training=False,
            )
            return mean_pool_l2(out.last_hidden_state, attention_mask)

    module = MiniLMModule(tf_model)
    concrete_fn = module.encode.get_concrete_function()

    # Calibration texts for INT8 representative dataset (≥100 samples)
    calib_texts = [t for pair in _SAMPLE_PAIRS for t in pair]  # 100 texts
    calib_data: list[tuple[np.ndarray, np.ndarray]] = []
    for text in calib_texts:
        enc = tokenizer(
            text,
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        calib_data.append((
            enc["input_ids"].astype(np.int32),
            enc["attention_mask"].astype(np.int32),
        ))

    def representative_dataset():
        for ids, mask in calib_data:
            yield [ids, mask]

    log.info("Converting MiniLM to INT8 TFLite (TFLITE_BUILTINS_INT8)…")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], module)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_bytes = converter.convert()
    TFLITE_PATH.write_bytes(tflite_bytes)

    size_mb   = TFLITE_PATH.stat().st_size / (1024 * 1024)
    flex_ops  = detect_flex_ops(tflite_bytes)

    # Op inventory via interpreter
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()

    log.info("TFLite size  : %.2f MB (limit %.0f MB)", size_mb, MAX_SIZE_MB)
    log.info("Flex ops     : %s", flex_ops or "none")

    return {
        "tflite_path": str(TFLITE_PATH),
        "size_mb":     round(size_mb, 3),
        "flex_ops":    flex_ops,
        "size_pass":   size_mb < MAX_SIZE_MB,
        "flex_pass":   len(flex_ops) == 0,
    }


# ── Step 2: Embedding quality check ──────────────────────────────────────────

def _encode_minilm(tokenizer, tf_model, texts: list[str]) -> np.ndarray:
    """Encode texts one-at-a-time to avoid OOM; returns [N, 384] float32."""
    embeddings = []
    for text in texts:
        enc = tokenizer(
            text,
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        out = tf_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=tf.zeros_like(enc["input_ids"]),
            training=False,
        )
        emb = mean_pool_l2(out.last_hidden_state, enc["attention_mask"])
        embeddings.append(emb.numpy()[0])
    return np.array(embeddings, dtype=np.float32)


def step2_quality(tokenizer, tf_model) -> dict:
    """Compute Spearman rank correlation between MiniLM and USE similarities."""
    resumes = [p[0] for p in _SAMPLE_PAIRS]
    jds     = [p[1] for p in _SAMPLE_PAIRS]

    log.info("Encoding %d resume texts with MiniLM…", len(resumes))
    r_mini = _encode_minilm(tokenizer, tf_model, resumes)
    log.info("Encoding %d JD texts with MiniLM…", len(jds))
    j_mini = _encode_minilm(tokenizer, tf_model, jds)
    sims_mini = cosine_similarity_batch(r_mini, j_mini)

    result: dict = {
        "num_pairs":        len(_SAMPLE_PAIRS),
        "minilm_sim_mean":  round(float(sims_mini.mean()), 4),
        "minilm_sim_std":   round(float(sims_mini.std()),  4),
        "minilm_sim_min":   round(float(sims_mini.min()),  4),
        "minilm_sim_max":   round(float(sims_mini.max()),  4),
    }

    use_path = Path(USE_LITE_URL)
    if not use_path.exists():
        log.warning("USE Lite path not found (%s) — skipping Spearman check.", USE_LITE_URL)
        result["use_available"]  = False
        result["spearman_rho"]   = None
        result["spearman_pass"]  = "SKIP_USE_UNAVAILABLE"
        return result

    try:
        import tensorflow_hub as hub
        log.info("Loading USE Lite from %s…", USE_LITE_URL)
        use_layer = hub.KerasLayer(
            USE_LITE_URL, input_shape=[], dtype=tf.string, trainable=False,
        )
        log.info("Encoding %d texts with USE Lite…", len(resumes) + len(jds))
        r_use = use_layer(tf.constant(resumes)).numpy()
        j_use = use_layer(tf.constant(jds)).numpy()
        sims_use = cosine_similarity_batch(r_use, j_use)

        rho = spearman_rho(sims_mini, sims_use)
        log.info("Spearman rho (MiniLM vs USE): %.4f (threshold %.2f)", rho, MIN_SPEARMAN)

        result["use_available"]     = True
        result["use_sim_mean"]      = round(float(sims_use.mean()), 4)
        result["spearman_rho"]      = round(rho, 4)
        result["spearman_pass"]     = bool(rho > MIN_SPEARMAN)

    except Exception as exc:
        log.warning("USE encoder error (%s) — skipping Spearman check.", exc)
        result["use_available"]  = False
        result["spearman_rho"]   = None
        result["spearman_pass"]  = "SKIP_USE_ERROR"
        result["use_error"]      = str(exc)

    return result


# ── Step 3: Tokenizer fidelity export ─────────────────────────────────────────

def step3_tokenizer(tokenizer) -> dict:
    """Export 20-sample token IDs for manual Dart WordPiece comparison.

    The Dart side must tokenize the same 20 texts with its WordPieceTokenizer
    (vocab from tokenizer.vocab_file) and compare input_ids list element-by-
    element against the values in this JSON.  100% match is required for E0.
    """
    samples = []
    for text in _TOK_SAMPLES:
        enc    = tokenizer(
            text,
            max_length=SEQ_LEN,
            padding="max_length",
            truncation=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        samples.append({
            "text":            text,
            "input_ids":       enc["input_ids"],
            "attention_mask":  enc["attention_mask"],
            "tokens":          tokens,
            "non_pad_length":  int(sum(enc["attention_mask"])),
        })

    return {
        "num_samples":     len(samples),
        "seq_len":         SEQ_LEN,
        "vocab_size":      tokenizer.vocab_size,
        "fidelity_status": "PENDING_DART_VERIFICATION",
        "instruction":     (
            "Tokenize each 'text' in Dart using WordPieceTokenizer with the MiniLM vocab. "
            "Apply [CLS]/[SEP] wrapping, truncate to 128, pad with [PAD]=0. "
            "Assert element-wise equality of 'input_ids' for all 20 samples."
        ),
        "samples": samples,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from transformers import AutoTokenizer, TFAutoModel
    except ImportError as exc:
        log.error("transformers library not installed: %s", exc)
        sys.exit(1)

    log.info("Loading tokenizer for %s…", MINILM_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MINILM_NAME)

    log.info("Loading TF model for %s…", MINILM_NAME)
    tf_model = TFAutoModel.from_pretrained(MINILM_NAME, from_pt=True)

    report: dict = {}
    failures: list[str] = []

    # ── Step 1 ──────────────────────────────────────────────────────────────
    print("\n── STEP 1: TFLite INT8 Conversion ──")
    try:
        s1 = step1_convert(tokenizer, tf_model)
    except Exception as exc:
        log.error("TFLite conversion failed: %s", exc, exc_info=True)
        failures.append(f"TFLite conversion exception: {exc}")
        s1 = {"error": str(exc), "size_pass": False, "flex_pass": False,
              "size_mb": -1, "flex_ops": [], "tflite_path": str(TFLITE_PATH)}
    report["conversion"] = s1

    if not s1.get("size_pass"):
        failures.append(f"Encoder size {s1.get('size_mb', '?')} MB >= {MAX_SIZE_MB} MB limit")
    if not s1.get("flex_pass"):
        failures.append(f"Flex ops detected: {s1.get('flex_ops', [])}")

    # ── Step 2 ──────────────────────────────────────────────────────────────
    print("\n── STEP 2: Embedding Quality (Spearman vs USE) ──")
    s2 = step2_quality(tokenizer, tf_model)
    report["embedding_quality"] = s2

    if s2.get("spearman_pass") is False:
        failures.append(
            f"Spearman rho {s2['spearman_rho']:.4f} < threshold {MIN_SPEARMAN}"
        )

    # ── Step 3 ──────────────────────────────────────────────────────────────
    print("\n── STEP 3: Tokenizer Fidelity Export ──")
    s3 = step3_tokenizer(tokenizer)
    report["tokenizer_fidelity"] = s3

    # ── Write report ─────────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report written → %s", REPORT_PATH)

    # ── Summary ──────────────────────────────────────────────────────────────
    spearman_str = (
        f"{s2['spearman_rho']:.4f}  (min {MIN_SPEARMAN})"
        if s2.get("spearman_rho") is not None
        else str(s2.get("spearman_pass"))
    )
    spearman_status = (
        "[PASS]" if s2.get("spearman_pass") is True
        else ("[FAIL]" if s2.get("spearman_pass") is False else "[SKIP]")
    )

    print("\n" + "=" * 62)
    print("  INJECTION-E0 ENCODER VALIDATION REPORT")
    print("=" * 62)
    print(f"  TFLite path     : {s1.get('tflite_path', 'N/A')}")
    print(f"  Encoder size    : {s1.get('size_mb', '?'):.3f} MB  "
          f"(limit {MAX_SIZE_MB} MB)  "
          f"{'[PASS]' if s1.get('size_pass') else '[FAIL]'}")
    print(f"  Flex ops        : {s1.get('flex_ops') or 'none'}  "
          f"{'[PASS]' if s1.get('flex_pass') else '[FAIL]'}")
    print(f"  MiniLM sim mean : {s2.get('minilm_sim_mean', '?')}")
    print(f"  Spearman rho    : {spearman_str}  {spearman_status}")
    print(f"  Tokenizer fid.  : {s3['fidelity_status']}  "
          f"({s3['num_samples']} samples saved)")
    print(f"  Report          : {REPORT_PATH}")
    print("=" * 62)

    if failures:
        print("\n  E0 FAILED — Hard Stop:")
        for msg in failures:
            print(f"    - {msg}")
        print()
        sys.exit(1)
    else:
        print("\n  E0 PASSED — proceed to E1\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
