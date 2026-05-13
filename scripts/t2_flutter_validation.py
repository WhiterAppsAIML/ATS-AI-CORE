"""INJECTION-T2 — Flutter Integration Verification

Generates 20-sample test vectors from curated resume–JD pairs (all 7 domains,
4 fresher profiles), benchmarks TFLite CPU latency, and writes IO_SCHEMA.md
with the MiniLM tokenizer contract for the Flutter integration team.

Hard stops:
  - TFLite model not found at expected path
  - Mean latency >= 500 ms

Outputs:
  evaluation/t2_test_vectors.json
  evaluation/t2_integration_report.json
  model/tflite/IO_SCHEMA.md (created / updated)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import NoReturn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

_TFLiteInterpreter = tf.lite.Interpreter  # type: ignore[attr-defined]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent   # ats/
TFLITE_PATH = ROOT_DIR / "model" / "tflite" / "ats_unified_minilm_int8.tflite"
VOCAB_PATH  = ROOT_DIR / "model" / "tflite" / "vocab.txt"
META_PATH   = ROOT_DIR / "model" / "tflite" / "tokenizer_config.json"
EVAL_DIR    = ROOT_DIR / "evaluation"
TFLITE_DIR  = ROOT_DIR / "model" / "tflite"

VECTORS_PATH = EVAL_DIR / "t2_test_vectors.json"
REPORT_PATH  = EVAL_DIR / "t2_integration_report.json"
SCHEMA_PATH  = TFLITE_DIR / "IO_SCHEMA.md"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN       = 128
LATENCY_RUNS      = 100
LATENCY_WARMUP    = 5
LATENCY_LIMIT_MS  = 500.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("T2")

_BANNER = "=" * 62


def _hard_stop(msg: str) -> NoReturn:
    log.error("HARD STOP — %s", msg)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 20 DIVERSE RESUME–JD PAIRS
# 7 domains covered; pairs 1, 4, 7, 10 are fresher profiles (4 total)
# ═══════════════════════════════════════════════════════════════════════════════

TEST_PAIRS: list[dict] = [
    # ── Domain 0: IT / Software (3 pairs, 1 fresher) ─────────────────────────
    {
        "id": 1, "domain": 0, "domain_name": "IT / Software", "is_fresher": True,
        "resume_text": (
            "Computer Science graduate with strong foundation in Python and Java. "
            "Developed web application using React and Node.js as final year project. "
            "Familiar with REST APIs, Git version control, and agile methodology. "
            "Completed internship at software startup building e-commerce features."
        ),
        "jd_text": (
            "Junior Software Engineer. Requirements: proficiency in Python or Java, "
            "understanding of REST APIs and web frameworks, Git version control, "
            "ability to work in agile team environment, CS degree or equivalent."
        ),
    },
    {
        "id": 2, "domain": 0, "domain_name": "IT / Software", "is_fresher": False,
        "resume_text": (
            "Senior Python developer with 6 years experience building scalable microservices. "
            "Expert in FastAPI, Docker, Kubernetes, and AWS cloud infrastructure. "
            "Optimized PostgreSQL and Redis databases reducing query time by 40%. "
            "Led backend team of 5 engineers at fintech company."
        ),
        "jd_text": (
            "Senior Python Developer. Must have FastAPI or Django, Docker and Kubernetes, "
            "AWS cloud infrastructure, PostgreSQL database optimization, "
            "5 plus years Python experience, team leadership skills required."
        ),
    },
    {
        "id": 3, "domain": 0, "domain_name": "IT / Software", "is_fresher": False,
        "resume_text": (
            "Full stack developer with 4 years React and Node.js experience. "
            "Built and maintained SaaS platforms serving 50000 daily active users. "
            "Implemented CI/CD pipelines using GitHub Actions and Jenkins. "
            "Strong skills in MongoDB, TypeScript, and GraphQL."
        ),
        "jd_text": (
            "Full Stack Engineer. React, Node.js, TypeScript, MongoDB, GraphQL. "
            "Experience with CI/CD pipelines and cloud deployments required. "
            "Strong understanding of software architecture and performance optimization."
        ),
    },
    # ── Domain 1: Non-IT / Management (3 pairs, 1 fresher) ───────────────────
    {
        "id": 4, "domain": 1, "domain_name": "Non-IT / Management", "is_fresher": True,
        "resume_text": (
            "MBA graduate specializing in operations and supply chain management. "
            "Completed internship at manufacturing company managing vendor relationships "
            "and inventory optimization. Strong analytical and communication skills. "
            "Proficient in Excel, PowerBI, and project management tools."
        ),
        "jd_text": (
            "Management Trainee Operations. MBA graduate preferred. "
            "Vendor management, inventory control, data analysis using Excel or PowerBI, "
            "strong communication skills, willingness to travel."
        ),
    },
    {
        "id": 5, "domain": 1, "domain_name": "Non-IT / Management", "is_fresher": False,
        "resume_text": (
            "HR Director with 12 years experience in talent acquisition and employee development. "
            "Managed recruitment for 300 positions annually across 5 business units. "
            "Designed compensation structures and performance management systems. "
            "Expert in HRIS platforms including Workday and SAP SuccessFactors."
        ),
        "jd_text": (
            "HR Director. Minimum 10 years HR experience, talent acquisition strategy, "
            "compensation design, performance management, HRIS systems Workday preferred, "
            "business partner experience across multiple functions."
        ),
    },
    {
        "id": 6, "domain": 1, "domain_name": "Non-IT / Management", "is_fresher": False,
        "resume_text": (
            "Operations Manager with 8 years managing logistics and supply chain. "
            "Led team of 60 warehouse staff and reduced operational costs by 25% through "
            "process optimization and lean manufacturing principles. "
            "Experience with ERP systems and demand forecasting."
        ),
        "jd_text": (
            "Operations Manager. Supply chain and logistics management, team leadership 50 plus staff, "
            "lean manufacturing or Six Sigma certification preferred, ERP systems, "
            "budget management and cost reduction track record required."
        ),
    },
    # ── Domain 2: Design / Creative (3 pairs, 1 fresher) ─────────────────────
    {
        "id": 7, "domain": 2, "domain_name": "Design / Creative", "is_fresher": True,
        "resume_text": (
            "Visual Communication graduate with strong portfolio in brand identity and UI design. "
            "Proficient in Adobe Illustrator, Photoshop, and Figma. "
            "Completed freelance branding projects for 5 local businesses. "
            "Understanding of UX research methods and accessibility standards."
        ),
        "jd_text": (
            "Junior Graphic Designer. Adobe Creative Suite proficiency required. "
            "Figma experience, UI UX understanding, portfolio demonstrating brand identity work, "
            "ability to work with marketing team on campaign materials."
        ),
    },
    {
        "id": 8, "domain": 2, "domain_name": "Design / Creative", "is_fresher": False,
        "resume_text": (
            "Senior UX Designer with 7 years experience in product design for fintech. "
            "Led design system for mobile banking app with 2 million users. "
            "Expert in user research, usability testing, and interaction design. "
            "Managed junior designers and collaborated with cross-functional product teams."
        ),
        "jd_text": (
            "Senior UX Designer. 5 plus years product design experience, design system ownership, "
            "user research and usability testing expertise, Figma proficiency, "
            "fintech or banking domain experience preferred, team leadership."
        ),
    },
    {
        "id": 9, "domain": 2, "domain_name": "Design / Creative", "is_fresher": False,
        "resume_text": (
            "Creative Director with 11 years at top advertising agencies. "
            "Managed creative team of 18 designers and copywriters. "
            "Won 5 national and 2 international advertising awards. "
            "Expert in brand strategy, integrated campaigns, and client presentations."
        ),
        "jd_text": (
            "Creative Director. 10 plus years advertising or marketing experience, "
            "team management, award-winning portfolio, brand strategy expertise, "
            "client-facing presentation skills, integrated campaign management."
        ),
    },
    # ── Domain 3: Healthcare (3 pairs, 1 fresher) ─────────────────────────────
    {
        "id": 10, "domain": 3, "domain_name": "Healthcare", "is_fresher": True,
        "resume_text": (
            "MBBS graduate completing internship at tertiary care hospital. "
            "Strong clinical assessment skills in internal medicine and emergency care. "
            "Completed research project on hypertension management outcomes. "
            "Basic life support and ACLS certified."
        ),
        "jd_text": (
            "Junior Medical Officer. MBBS degree required, completion of internship, "
            "clinical assessment skills, BLS and ACLS certification, "
            "willingness to work rotational shifts in hospital setting."
        ),
    },
    {
        "id": 11, "domain": 3, "domain_name": "Healthcare", "is_fresher": False,
        "resume_text": (
            "ICU Registered Nurse with 9 years critical care experience. "
            "Expert in mechanical ventilator management, hemodynamic monitoring, "
            "and post-cardiac surgery care. Trained 25 junior nurses. "
            "CCRN certification and NIH stroke scale certified."
        ),
        "jd_text": (
            "Senior ICU Nurse. Minimum 5 years critical care nursing experience, "
            "ventilator management competency, CCRN certification preferred, "
            "hemodynamic monitoring expertise, charge nurse experience valued."
        ),
    },
    {
        "id": 12, "domain": 3, "domain_name": "Healthcare", "is_fresher": False,
        "resume_text": (
            "Clinical pharmacist with 10 years hospital pharmacy experience. "
            "Expert in antimicrobial stewardship, pharmacokinetics, and drug interactions. "
            "Led pharmacy department of 15 staff. Implemented medication reconciliation "
            "program reducing adverse drug events by 30%."
        ),
        "jd_text": (
            "Clinical Pharmacist Manager. PharmD required, hospital pharmacy management experience, "
            "antimicrobial stewardship program knowledge, staff management, "
            "medication safety initiatives, clinical rounding with medical teams."
        ),
    },
    # ── Domain 4: Finance / Banking (2 pairs) ─────────────────────────────────
    {
        "id": 13, "domain": 4, "domain_name": "Finance / Banking", "is_fresher": False,
        "resume_text": (
            "Investment analyst with 5 years at bulge bracket investment bank. "
            "CFA Level 3 candidate. Expert in DCF, LBO, and precedent transactions analysis. "
            "Covered technology and healthcare sectors. Bloomberg Terminal proficient. "
            "Executed 12 M&A advisory mandates totalling 3.5 billion dollars."
        ),
        "jd_text": (
            "Investment Banking Analyst. CFA qualification preferred, "
            "financial modeling expertise including DCF and LBO, "
            "Bloomberg Terminal, sector coverage experience, M&A advisory background."
        ),
    },
    {
        "id": 14, "domain": 4, "domain_name": "Finance / Banking", "is_fresher": False,
        "resume_text": (
            "Retail bank branch manager with 9 years commercial banking experience. "
            "Managed portfolio of 600 SME and corporate clients worth 120 million. "
            "Expert in credit assessment, risk management, and relationship banking. "
            "Consistently exceeded sales targets by 20% annually."
        ),
        "jd_text": (
            "Branch Manager Commercial Banking. Minimum 6 years banking experience, "
            "credit analysis and risk management skills, relationship management, "
            "sales target achievement, team leadership, CFA or MBA preferred."
        ),
    },
    # ── Domain 5: Legal (3 pairs) ─────────────────────────────────────────────
    {
        "id": 15, "domain": 5, "domain_name": "Legal", "is_fresher": False,
        "resume_text": (
            "Corporate M&A associate with 8 years at Magic Circle law firm. "
            "Specializes in cross-border mergers, acquisitions, and private equity transactions. "
            "Led legal due diligence on 20 deals worth 4 billion dollars. "
            "LLM from Cambridge, admitted to Bar in England and Wales."
        ),
        "jd_text": (
            "Corporate Associate M&A. 5 to 8 years M&A transaction experience at top-tier firm, "
            "cross-border deal exposure, private equity transactions, due diligence management, "
            "LLM preferred, strong academic background required."
        ),
    },
    {
        "id": 16, "domain": 5, "domain_name": "Legal", "is_fresher": False,
        "resume_text": (
            "Compliance manager with 11 years in financial services regulatory compliance. "
            "Expert in Basel III capital requirements, MiFID II, GDPR, and AML KYC frameworks. "
            "Managed FCA audit processes and regulatory reporting. "
            "Law degree from LSE and CAMS certified."
        ),
        "jd_text": (
            "Head of Compliance Financial Services. Law degree required, CAMS certification, "
            "regulatory compliance expertise in FCA and PRA framework, AML KYC program management, "
            "audit preparation and regulatory liaison, minimum 8 years experience."
        ),
    },
    {
        "id": 17, "domain": 5, "domain_name": "Legal", "is_fresher": False,
        "resume_text": (
            "IP attorney specializing in technology patents and trademark disputes. "
            "Filed 120 patent applications across semiconductor, AI, and biotech sectors. "
            "7 years USPTO prosecution experience, 4 PTAB proceedings. "
            "JD with technical background in electrical engineering."
        ),
        "jd_text": (
            "IP Attorney Patent Prosecution. USPTO registration required, "
            "technology patent prosecution experience in semiconductor or software, "
            "PTAB proceedings experience, JD with technical degree, "
            "client counseling and trademark portfolio management."
        ),
    },
    # ── Domain 6: Education (3 pairs) ─────────────────────────────────────────
    {
        "id": 18, "domain": 6, "domain_name": "Education", "is_fresher": False,
        "resume_text": (
            "Secondary school mathematics teacher with 11 years experience. "
            "Improved student pass rates from 68% to 93% over 5 years. "
            "Developed digital mathematics curriculum for IB and A-Level programmes. "
            "Head of Mathematics department, mentored 8 junior teachers."
        ),
        "jd_text": (
            "Head of Mathematics Secondary School. Teaching certification required, "
            "minimum 7 years secondary education experience, IB or A-Level curriculum expertise, "
            "department leadership experience, digital learning platform skills."
        ),
    },
    {
        "id": 19, "domain": 6, "domain_name": "Education", "is_fresher": False,
        "resume_text": (
            "Associate Professor of Computer Science at research university. "
            "PhD from Stanford, specialization in machine learning and distributed systems. "
            "Published 28 peer-reviewed papers with 1200 citations. "
            "Teaching undergraduate algorithms and graduate deep learning courses."
        ),
        "jd_text": (
            "Assistant or Associate Professor Computer Science. PhD required, "
            "strong publication record in machine learning or systems area, teaching experience "
            "in core CS subjects, grant writing and research supervision capabilities."
        ),
    },
    {
        "id": 20, "domain": 6, "domain_name": "Education", "is_fresher": False,
        "resume_text": (
            "Curriculum development manager with 9 years in teacher training and instructional design. "
            "Designed blended learning programmes for 600 teachers across 40 schools. "
            "Expert in eLearning authoring tools including Articulate 360 and Adobe Captivate. "
            "Certified instructional designer and project management professional."
        ),
        "jd_text": (
            "Curriculum Development Manager. Instructional design expertise required, "
            "eLearning tool proficiency in Articulate or Captivate, teacher training programme design, "
            "project management certification preferred, stakeholder management experience."
        ),
    },
]

# 3 short phrases used to generate Dart tokenizer worked examples
_DART_EXAMPLE_PHRASES = [
    "python developer with aws experience",
    "registered nurse critical care icu",
    "mathematics teacher secondary school",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

def load_tokenizer():
    print(f"\n[1/6] Loading tokenizer …")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _hard_stop("transformers not installed — run: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)
    log.info("Tokenizer loaded: %s  (vocab_size=%d)", MINILM_MODEL_NAME, tokenizer.vocab_size)
    return tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD TFLITE INTERPRETER
# ═══════════════════════════════════════════════════════════════════════════════

def load_interpreter():
    print(f"\n[2/6] Loading TFLite interpreter …")
    if not TFLITE_PATH.exists():
        _hard_stop(f"TFLite model not found: {TFLITE_PATH}")

    model_bytes = TFLITE_PATH.read_bytes()
    interp = _TFLiteInterpreter(model_content=model_bytes)
    interp.allocate_tensors()
    size_mb = len(model_bytes) / (1024 * 1024)
    log.info("TFLite loaded: %s  (%.2f MB)", TFLITE_PATH.name, size_mb)

    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()
    log.info("Inputs  (%d): %s", len(in_details),
             [(d["name"], d["shape"].tolist(), str(np.dtype(d["dtype"]))) for d in in_details])
    log.info("Outputs (%d): %s", len(out_details),
             [(d["name"], d["shape"].tolist()) for d in out_details])
    return interp


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOKENIZE
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_text(tokenizer, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a single text → (input_ids [1,128], attention_mask [1,128]) int32."""
    enc = tokenizer(
        text,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TFLITE INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _set_inputs(
    interp,
    r_ids: np.ndarray,
    r_mask: np.ndarray,
    j_ids: np.ndarray,
    j_mask: np.ndarray,
) -> None:
    """Set TFLite input tensors by matching tensor name substrings."""
    assigned: set[int] = set()
    for d in interp.get_input_details():
        n = d["name"].lower()
        idx = d["index"]
        if "resume_input_ids" in n and idx not in assigned:
            interp.set_tensor(idx, r_ids); assigned.add(idx)
        elif "resume_attention_mask" in n and idx not in assigned:
            interp.set_tensor(idx, r_mask); assigned.add(idx)
        elif "jd_input_ids" in n and idx not in assigned:
            interp.set_tensor(idx, j_ids); assigned.add(idx)
        elif "jd_attention_mask" in n and idx not in assigned:
            interp.set_tensor(idx, j_mask); assigned.add(idx)

    # Positional fallback if name matching missed any
    in_details = interp.get_input_details()
    if len(assigned) < 4 and len(in_details) == 4:
        order = [r_ids, r_mask, j_ids, j_mask]
        for i, d in enumerate(in_details):
            if d["index"] not in assigned:
                interp.set_tensor(d["index"], order[i])


def _get_outputs(
    interp,
) -> tuple[float, list[float], list[float]]:
    """Return (ats_raw 0-1, domain_probs[7], rsg_probs[46]) from TFLite outputs."""
    ats_raw: float | None = None
    dom_probs: list[float] | None = None
    rsg_probs: list[float] | None = None

    for d in interp.get_output_details():
        t = interp.get_tensor(d["index"])
        s = int(t.shape[-1])
        if s == 1:
            ats_raw = float(t.ravel()[0])
        elif s == 7:
            dom_probs = [float(x) for x in t.ravel()]
        elif s == 46:
            rsg_probs = [float(x) for x in t.ravel()]

    if ats_raw is None or dom_probs is None or rsg_probs is None:
        _hard_stop(
            "TFLite output shape mismatch — expected shapes [1,1], [1,7], [1,46]; "
            f"got {[d['shape'].tolist() for d in interp.get_output_details()]}"
        )
    return ats_raw, dom_probs, rsg_probs  # type: ignore[return-value]


def _infer(interp, r_ids, r_mask, j_ids, j_mask):
    _set_inputs(interp, r_ids, r_mask, j_ids, j_mask)
    interp.invoke()
    return _get_outputs(interp)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE 20 TEST VECTORS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_vectors(tokenizer, interp) -> list[dict]:
    print(f"\n[3/6] Generating test vectors ({len(TEST_PAIRS)} pairs) …")
    vectors = []
    for pair in TEST_PAIRS:
        r_ids, r_mask = tokenize_text(tokenizer, pair["resume_text"])
        j_ids, j_mask = tokenize_text(tokenizer, pair["jd_text"])
        ats_raw, dom_probs, rsg_probs = _infer(interp, r_ids, r_mask, j_ids, j_mask)

        dom_argmax = int(np.argmax(dom_probs))
        rsg_argmax = int(np.argmax(rsg_probs))

        vec: dict = {
            "id":          pair["id"],
            "domain":      pair["domain"],
            "domain_name": pair["domain_name"],
            "is_fresher":  pair["is_fresher"],
            "resume_text": pair["resume_text"],
            "jd_text":     pair["jd_text"],
            "tokenization": {
                "resume_input_ids":       r_ids[0].tolist(),
                "resume_attention_mask":  r_mask[0].tolist(),
                "jd_input_ids":           j_ids[0].tolist(),
                "jd_attention_mask":      j_mask[0].tolist(),
            },
            "outputs": {
                "ats_score_raw":        round(ats_raw, 6),
                "ats_score_pct":        round(ats_raw * 100.0, 2),
                "domain_probs":         [round(x, 6) for x in dom_probs],
                "domain_argmax":        dom_argmax,
                "rsg_template_probs":   [round(x, 6) for x in rsg_probs],
                "rsg_template_argmax":  rsg_argmax,
            },
        }
        vectors.append(vec)
        log.info(
            "  [%2d/20] %-26s  ats=%5.1f%%  dom_pred=%d (%s)  rsg=%d",
            pair["id"], pair["domain_name"],
            ats_raw * 100.0, dom_argmax, pair["domain_name"], rsg_argmax,
        )

    return vectors


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_latency(interp, sample_vec: dict) -> dict:
    print(f"\n[4/6] Latency benchmark — {LATENCY_WARMUP} warmup + {LATENCY_RUNS} timed runs …")

    r_ids  = np.array(sample_vec["tokenization"]["resume_input_ids"],  dtype=np.int32).reshape(1, MAX_SEQ_LEN)
    r_mask = np.array(sample_vec["tokenization"]["resume_attention_mask"], dtype=np.int32).reshape(1, MAX_SEQ_LEN)
    j_ids  = np.array(sample_vec["tokenization"]["jd_input_ids"],      dtype=np.int32).reshape(1, MAX_SEQ_LEN)
    j_mask = np.array(sample_vec["tokenization"]["jd_attention_mask"],  dtype=np.int32).reshape(1, MAX_SEQ_LEN)

    for _ in range(LATENCY_WARMUP):
        _infer(interp, r_ids, r_mask, j_ids, j_mask)

    times_ms: list[float] = []
    for _ in range(LATENCY_RUNS):
        _set_inputs(interp, r_ids, r_mask, j_ids, j_mask)
        t0 = time.perf_counter()
        interp.invoke()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    stats = {
        "device":      "CPU",
        "runs":        LATENCY_RUNS,
        "mean_ms":     round(float(np.mean(arr)),           2),
        "p50_ms":      round(float(np.percentile(arr, 50)), 2),
        "p95_ms":      round(float(np.percentile(arr, 95)), 2),
        "p99_ms":      round(float(np.percentile(arr, 99)), 2),
        "min_ms":      round(float(np.min(arr)),            2),
        "max_ms":      round(float(np.max(arr)),            2),
        "limit_ms":    LATENCY_LIMIT_MS,
        "passed":      bool(float(np.mean(arr)) < LATENCY_LIMIT_MS),
    }
    result = "PASS" if stats["passed"] else "FAIL"
    log.info(
        "Latency — mean=%.1f ms  p50=%.1f ms  p95=%.1f ms  p99=%.1f ms  [%s]",
        stats["mean_ms"], stats["p50_ms"], stats["p95_ms"], stats["p99_ms"], result,
    )
    if not stats["passed"]:
        _hard_stop(
            f"Mean latency {stats['mean_ms']:.1f} ms >= {LATENCY_LIMIT_MS:.0f} ms limit"
        )
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DART WORKED EXAMPLES (3 short phrases → actual token IDs)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_dart_examples(tokenizer) -> list[dict]:
    examples = []
    for phrase in _DART_EXAMPLE_PHRASES:
        enc = tokenizer(
            phrase,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
        )
        input_ids = list(enc["input_ids"])
        attn_mask = list(enc["attention_mask"])
        tokens    = tokenizer.convert_ids_to_tokens(input_ids)
        non_pad   = int(sum(attn_mask))
        examples.append({
            "text":       phrase,
            "input_ids":  input_ids,
            "attn_mask":  attn_mask,
            "tokens":     tokens,
            "non_pad":    non_pad,
        })
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WRITE IO_SCHEMA.md
# ═══════════════════════════════════════════════════════════════════════════════

def write_io_schema(interp, dart_examples: list[dict]) -> None:
    print(f"\n[5/6] Writing IO_SCHEMA.md …")

    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    def _dtype_str(d: dict) -> str:
        return str(np.dtype(d["dtype"])).upper()

    def _shape_str(d: dict) -> str:
        return "[" + ", ".join(str(x) for x in d["shape"].tolist()) + "]"

    # ── Input table ───────────────────────────────────────────────────────────
    in_table_rows = ""
    for d in in_details:
        in_table_rows += f"| `{d['name']}` | `{_shape_str(d)}` | `{_dtype_str(d)}` |\n"

    # ── Output table ──────────────────────────────────────────────────────────
    out_table_rows = ""
    for d in out_details:
        s = int(d["shape"][-1])
        if s == 1:
            postproc = "Multiply raw by **100** → ATS score 0–100"
        elif s == 7:
            postproc = "`argmax` → domain index 0–6"
        elif s == 46:
            postproc = "`argmax` → RSG template index 0–45"
        else:
            postproc = "—"
        out_table_rows += f"| `{d['name']}` | `{_shape_str(d)}` | `{_dtype_str(d)}` | {postproc} |\n"

    # ── Dart worked examples ──────────────────────────────────────────────────
    ex_md = ""
    for i, ex in enumerate(dart_examples, 1):
        real_tokens = " ".join(ex["tokens"][: ex["non_pad"]])
        real_ids    = ex["input_ids"][: ex["non_pad"]]
        ex_md += f"""
#### Example {i}: `"{ex['text']}"`

| Step | Action | Result |
|------|--------|--------|
| 1 | Lowercase | `"{ex['text'].lower()}"` |
| 2 | WordPiece tokens (first {ex['non_pad']}) | `{real_tokens}` |
| 3 | Token IDs (real portion) | `{real_ids}` |
| 4 | Pad to 128 with `[PAD]`=0 | positions {ex['non_pad']}–127 = 0 |
| 5 | Attention mask | 1 for positions 0–{ex['non_pad'] - 1}, 0 for {ex['non_pad']}–127 |

Full `input_ids` (128 values):
```
{json.dumps(ex['input_ids'])}
```

Full `attention_mask` (128 values):
```
{json.dumps(ex['attn_mask'])}
```

"""

    # ── Assemble markdown ─────────────────────────────────────────────────────
    content = f"""# ATS Unified Model — TFLite IO Schema

> **Stage:** INJECTION-T2 Flutter Integration
> **Model:** `ats_unified_minilm_int8.tflite` (22.8 MB, dynamic-range INT8)
> **Encoder:** sentence-transformers/all-MiniLM-L6-v2
> **Updated:** `scripts/t2_flutter_validation.py`

---

## 1. Model File

| Property | Value |
|----------|-------|
| Filename | `ats_unified_minilm_int8.tflite` |
| Size | ~22.8 MB |
| Quantization | Dynamic-range INT8 (weights only; activations float32) |
| Flex ops | **0** — standard `TFLITE_BUILTINS` only |
| Encoder | all-MiniLM-L6-v2 (384-dim, frozen) |
| Vocab size | 30 522 (WordPiece) |
| Max sequence length | 128 tokens |

---

## 2. Input Tensors

Pass **4 tensors**, all `int32`, shape `[1, 128]`.

| Tensor name | Shape | Dtype |
|-------------|-------|-------|
{in_table_rows}
**Notes:**

- All token IDs come from the WordPiece tokenizer (vocab size = 30 522).
- Attention mask: `1` for real tokens (including `[CLS]` and `[SEP]`), `0` for `[PAD]`.
- Batch size is always **1** — the model does not support batched inference.
- `token_type_ids` (all-zeros for single-sentence BERT) are handled **internally**; do **not** add them as external inputs.

---

## 3. Output Tensors

| Tensor name | Shape | Dtype | Post-processing |
|-------------|-------|-------|-----------------|
{out_table_rows}
### Post-processing (Dart)

```dart
// ── ATS Score ──────────────────────────────────────────────
// Output shape [1, 1], dtype float32, range roughly 0.0–1.0
final double atsScoreRaw = atsOutput[0][0];
final double atsScorePct = atsScoreRaw * 100.0;   // e.g. 0.72 → 72.0 %

// ── Domain Classification ───────────────────────────────────
// Output shape [1, 7], dtype float32 (softmax probabilities)
final List<double> domainProbs = domainOutput[0];
final int domainIndex = argmax(domainProbs);       // 0–6
const domainNames = [
  'IT / Software', 'Non-IT / Management', 'Design / Creative',
  'Healthcare', 'Finance / Banking', 'Legal', 'Education',
];
final String domainName = domainNames[domainIndex];

// ── RSG Template Selection ──────────────────────────────────
// Output shape [1, 46], dtype float32 (softmax probabilities)
final List<double> rsgProbs = rsgOutput[0];
final int rsgTemplateIndex = argmax(rsgProbs);     // 0–45
```

---

## 4. Domain Index Mapping

| Index | Domain |
|-------|--------|
| 0 | IT / Software |
| 1 | Non-IT / Management |
| 2 | Design / Creative |
| 3 | Healthcare |
| 4 | Finance / Banking |
| 5 | Legal |
| 6 | Education |

---

## 5. Dart Tokenizer Specification

### Overview

The Flutter app must reproduce **exact** WordPiece tokenization matching
`sentence-transformers/all-MiniLM-L6-v2` (BERT-style, `do_lower_case=true`).
Token IDs must be identical to the Python reference — the TFLite model is
deterministic, so the same IDs always produce the same outputs.

### Special Token IDs

| Token | ID |
|-------|----|
| `[PAD]` | 0 |
| `[UNK]` | 100 |
| `[CLS]` | 101 |
| `[SEP]` | 102 |

### Tokenization Algorithm

```
1. Load vocab.txt from the TFLite metadata (or sidecar file).
   Build a Map<String, int> from token string → integer ID.

2. Lowercase the entire input text (do_lower_case = true).

3. Basic tokenization:
   a. Unicode-normalize (NFD) and strip combining marks (accents).
   b. Insert spaces around every CJK character and ASCII punctuation.
   c. Split on whitespace to obtain a list of "words".

4. For each word, apply WordPiece tokenization:
   a. If the whole word is in the vocab → single token.
   b. Otherwise greedily find the longest prefix in vocab;
      prepend "##" to the remaining suffix and recurse.
   c. If no valid segmentation exists → emit [UNK] (ID 100).

5. Prepend [CLS] (ID 101) to the token list.
   Append [SEP] (ID 102) to the token list.

6. Truncate if total length > {MAX_SEQ_LEN}:
   Keep the first {MAX_SEQ_LEN - 2} content tokens, then add [CLS] and [SEP].
   (Total is always exactly {MAX_SEQ_LEN}.)

7. Pad to exactly {MAX_SEQ_LEN} with [PAD] (ID 0).

8. Construct attention_mask:
   - 1 for every position up to and including [SEP]
   - 0 for every [PAD] position

9. Return Int32List of length {MAX_SEQ_LEN} for input_ids.
   Return Int32List of length {MAX_SEQ_LEN} for attention_mask.
```

### Recommended Flutter Package

```yaml
dependencies:
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.4.1  # provides BertTokenizer
```

```dart
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class AtsTokenizer {{
  final BertTokenizer _tokenizer;
  static const int maxSeqLen = {MAX_SEQ_LEN};

  AtsTokenizer(String vocabPath)
      : _tokenizer = BertTokenizer.fromFile(vocabPath, doLowerCase: true);

  /// Tokenize [text] → (input_ids, attention_mask) each as Int32List[{MAX_SEQ_LEN}].
  List<Int32List> tokenize(String text) {{
    final tokens  = _tokenizer.tokenize(text);
    final ids     = _tokenizer.convertTokensToIds(tokens);
    final content = ids.take(maxSeqLen - 2).toList();
    final full    = [101, ...content, 102];          // [CLS] … [SEP]
    final padded  = Int32List(maxSeqLen);
    final mask    = Int32List(maxSeqLen);
    for (int i = 0; i < full.length; i++) {{
      padded[i] = full[i];
      mask[i]   = 1;
    }}
    return [padded, mask];
  }}
}}
```

### Worked Examples
{ex_md}
---

## 6. Flutter Integration Checklist

```
[ ] Bundle ats_unified_minilm_int8.tflite and vocab.txt as assets
[ ] Load TFLite interpreter in an Isolate (CPU only — no GPU delegate required)
[ ] Instantiate AtsTokenizer with bundled vocab.txt path
[ ] Tokenize resume text  → resumeIds  (Int32List[128])
                           resumeMask (Int32List[128])
[ ] Tokenize JD text      → jdIds     (Int32List[128])
                           jdMask    (Int32List[128])
[ ] Set input tensors in order:
      resume_input_ids, resume_attention_mask,
      jd_input_ids,     jd_attention_mask
[ ] Call interpreter.invoke()
[ ] Read ats_score output [1,1]  → multiply by 100
[ ] Read domain_probs output [1,7]  → argmax
[ ] Read rsg_template output [1,46] → argmax
[ ] Validate against t2_test_vectors.json:
      ATS score within 2 pts of reference value
      Domain argmax matches exactly
      RSG argmax matches exactly
```

---

*Generated by INJECTION-T2 — `scripts/t2_flutter_validation.py`*
"""

    SCHEMA_PATH.write_text(content, encoding="utf-8")
    log.info("IO_SCHEMA.md written: %s", SCHEMA_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. WRITE OUTPUT FILES
# ═══════════════════════════════════════════════════════════════════════════════

def write_test_vectors(vectors: list[dict], tokenizer) -> None:
    payload = {
        "metadata": {
            "stage":      "T2",
            "tokenizer":  MINILM_MODEL_NAME,
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": MAX_SEQ_LEN,
            "do_lower_case": True,
            "model_file": TFLITE_PATH.name,
            "total_pairs": len(vectors),
            "domains_covered": sorted({v["domain"] for v in vectors}),
            "fresher_count": sum(1 for v in vectors if v["is_fresher"]),
        },
        "vectors": vectors,
    }
    VECTORS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Test vectors written: %s  (%d vectors)", VECTORS_PATH.name, len(vectors))


def write_report(vectors: list[dict], latency: dict) -> None:
    domain_counts: dict[str, int] = {}
    for v in vectors:
        domain_counts[v["domain_name"]] = domain_counts.get(v["domain_name"], 0) + 1

    report = {
        "stage":  "T2",
        "status": "PASSED",
        "model":  str(TFLITE_PATH),
        "test_vectors": {
            "count":           len(vectors),
            "domains_covered": sorted({v["domain"] for v in vectors}),
            "domain_counts":   domain_counts,
            "fresher_count":   sum(1 for v in vectors if v["is_fresher"]),
        },
        "latency": latency,
        "outputs": {
            "test_vectors_path": str(VECTORS_PATH),
            "report_path":       str(REPORT_PATH),
            "io_schema_path":    str(SCHEMA_PATH),
        },
        "checks": {
            "all_domains_covered": len({v["domain"] for v in vectors}) == 7,
            "freshers_gte_4":      sum(1 for v in vectors if v["is_fresher"]) >= 4,
            "latency_pass":        latency["passed"],
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Integration report written: %s", REPORT_PATH.name)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(_BANNER)
    print("  INJECTION-T2 — Flutter Integration Verification")
    print(_BANNER)

    tokenizer = load_tokenizer()
    interp    = load_interpreter()
    vectors   = generate_test_vectors(tokenizer, interp)
    latency   = benchmark_latency(interp, vectors[0])

    print(f"\n[5/6] Writing output files …")
    dart_examples = _build_dart_examples(tokenizer)
    write_io_schema(interp, dart_examples)
    write_test_vectors(vectors, tokenizer)
    write_report(vectors, latency)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_checks = (
        len({v["domain"] for v in vectors}) == 7
        and sum(1 for v in vectors if v["is_fresher"]) >= 4
        and latency["passed"]
    )

    print(f"\n[6/6] Verification summary")
    print(_BANNER)
    print(f"  Vectors      : {len(vectors)} pairs across 7 domains")
    print(f"  Fresher pairs: {sum(1 for v in vectors if v['is_fresher'])}")
    print(f"  Latency      : mean={latency['mean_ms']:.1f} ms  "
          f"p50={latency['p50_ms']:.1f} ms  "
          f"p95={latency['p95_ms']:.1f} ms  "
          f"p99={latency['p99_ms']:.1f} ms  [{'PASS' if latency['passed'] else 'FAIL'}]")
    print(f"  Test vectors : {VECTORS_PATH.name}")
    print(f"  Report       : {REPORT_PATH.name}")
    print(f"  IO schema    : {SCHEMA_PATH.name}")
    print(_BANNER)

    if not all_checks:
        log.error("One or more checks FAILED — see report for details")
        sys.exit(1)

    print("\nT2 COMPLETE — model ready for Flutter integration\n")


if __name__ == "__main__":
    main()
