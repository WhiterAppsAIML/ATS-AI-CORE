r"""
Unified Model Real-Time Test GUI
=================================
Tests the unified Keras model with:
  - PDF/TXT resume upload
  - Job description input
  - ATS scoring (score, band, domain, missing keywords, feedback)
  - RSG summary generation (template selection + SlotFiller)

Run from project root:
    cd C:\Users\saini\Desktop\ats
    $env:PYTHONPATH = "C:\Users\saini\Desktop\ats"
    python test_unified_gui.py

Requirements: tkinter (built-in), PyMuPDF for PDF reading
    pip install pymupdf --break-system-packages
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import sys
import json
import traceback
import typing

# Keep TensorFlow startup logs minimal and avoid oneDNN int64 fallback warnings.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ─── PATH SETUP ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ATS_CORE = os.path.join(PROJECT_ROOT, "ats-ai-core")
ATS_CORE_SRC = os.path.join(ATS_CORE, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ATS_CORE)
sys.path.insert(0, ATS_CORE_SRC)

WEIGHTS_PATH = os.path.join(ATS_CORE, "model", "unified_model", "best_unified_weights.h5")
RSG_LABEL_MAP = os.path.join(ATS_CORE, "model", "unified_model", "rsg_label_mapping.json")
RSG_TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "..", "rsg", "RSG-AI-MODULE-main", "data", "templates"
)

DOMAIN_NAMES = [
    "IT / Software", "Non-IT / Management", "Design / Creative",
    "Healthcare", "Finance / Banking", "Legal", "Education"
]

BAND_MAP = [
    (85, 101, "Excellent Match",  "#22c55e"),
    (65,  85, "Good Match",       "#84cc16"),
    (45,  65, "Moderate Match",   "#eab308"),
    (25,  45, "Weak Match",       "#f97316"),
    (0,   25, "Poor Match",       "#ef4444"),
]

# ─── MODEL LOADER ──────────────────────────────────────────────────────────────
_model = None
_label_map = None

def load_model():
    global _model, _label_map
    from unified_engine.unified_model import build_unified_model

    _model = build_unified_model()
    _model.load_weights(WEIGHTS_PATH)

    if os.path.exists(RSG_LABEL_MAP):
        with open(RSG_LABEL_MAP, "r") as f:
            _label_map = json.load(f)
    return _model


# ─── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(resume_text: str, jd_text: str) -> dict:
    import tensorflow as tf
    import numpy as np

    global _model
    if _model is None:
        load_model()

    outputs = _model.predict(
        {"resume_text": tf.constant([resume_text]), "jd_text": tf.constant([jd_text])},
        verbose=0
    )

    # Parse outputs  (order: ats_score, domain_logits, rsg_logits)
    ats_raw      = float(outputs[0][0][0])
    domain_probs = outputs[1][0]
    rsg_probs    = outputs[2][0] if len(outputs) > 2 else None

    ats_score    = round(ats_raw * 100, 2)
    domain_idx   = int(np.argmax(domain_probs))
    domain_name  = DOMAIN_NAMES[domain_idx]
    domain_conf  = round(float(domain_probs[domain_idx]) * 100, 1)

    # Band
    band_label = "Unknown"
    band_color = "#94a3b8"
    for lo, hi, label, color in BAND_MAP:
        if lo <= ats_score < hi:
            band_label = label
            band_color = color
            break

    # Fresher detection (simple heuristic)
    fresher_keywords = ["fresher", "0 years", "no experience", "recent graduate",
                        "entry level", "internship", "pursuing", "b.tech", "b.e.", "bsc"]
    is_fresher = any(k in resume_text.lower() for k in fresher_keywords)

    # Missing keywords (TF-IDF based)
    missing = extract_missing_keywords(resume_text, jd_text)

    # Feedback
    feedback = generate_feedback(ats_score, domain_name, missing, is_fresher)

    # RSG template
    rsg_result = None
    if rsg_probs is not None and _label_map is not None:
        rsg_idx = int(np.argmax(rsg_probs))
        rsg_confidence = round(float(rsg_probs[rsg_idx]) * 100, 1)
        # Reverse label map (value → key)
        idx_to_label = {v: k for k, v in _label_map.items()}
        template_id = idx_to_label.get(rsg_idx, f"template_{rsg_idx}")
        rsg_result = {
            "template_id": template_id,
            "confidence": rsg_confidence,
            "summary": generate_rsg_summary(resume_text, jd_text, domain_name, template_id, is_fresher)
        }

    return {
        "ats_score": ats_score,
        "band": band_label,
        "band_color": band_color,
        "domain_index": domain_idx,
        "domain_name": domain_name,
        "domain_confidence": domain_conf,
        "is_fresher": is_fresher,
        "missing_keywords": missing,
        "feedback": feedback,
        "rsg": rsg_result,
    }


def extract_missing_keywords(resume_text: str, jd_text: str) -> dict:
    """Simple TF-IDF based keyword gap analysis."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        STOPWORDS = {
            "the","and","for","with","this","that","are","you","have","will",
            "our","your","their","they","from","but","not","can","all","we",
            "any","its","been","has","was","who","use","may","more","also",
            "each","such","both","very","into","than","other","must","about",
            "which","when","what","where","how","resume","job","candidate"
        }

        vect = TfidfVectorizer(max_features=60, stop_words="english", ngram_range=(1, 2))
        vect.fit([jd_text])
        jd_terms = set(vect.get_feature_names_out())

        resume_lower = resume_text.lower()
        missing_all = [t for t in jd_terms
                       if t.lower() not in resume_lower
                       and t.lower() not in STOPWORDS
                       and len(t) > 2]

        # Simple hard/soft skill split
        soft_markers = ["communication","leadership","team","collaborate","interpersonal",
                        "problem solving","analytical","management","detail","motivated"]
        hard = [k for k in missing_all if not any(s in k for s in soft_markers)][:8]
        soft = [k for k in missing_all if any(s in k for s in soft_markers)][:4]

        return {"hard_skills": hard, "soft_skills": soft}
    except Exception:
        return {"hard_skills": [], "soft_skills": []}


def generate_feedback(score: float, domain: str, missing: dict, is_fresher: bool) -> list:
    tips = []
    hard = missing.get("hard_skills", [])
    soft = missing.get("soft_skills", [])

    if hard:
        tips.append(f"Add these missing technical skills to your resume: {', '.join(hard[:4])}.")
    if soft:
        tips.append(f"Highlight soft skills such as: {', '.join(soft[:3])}.")
    if score < 45:
        tips.append("Your resume needs significant tailoring for this role. Focus on matching the JD language.")
    elif score < 65:
        tips.append("Good foundation — customize your summary section to mirror the job description.")
    elif score >= 85:
        tips.append("Strong match! Ensure your most relevant experience appears in the top half of your resume.")

    if is_fresher:
        tips.append("As a fresher, emphasize academic projects, internships, and certifications relevant to this domain.")

    if "IT" in domain or "Software" in domain:
        tips.append("Include GitHub links, project URLs, or portfolio links for technical roles.")
    elif "Finance" in domain:
        tips.append("Quantify achievements with numbers — % growth, $ managed, cost savings.")
    elif "Healthcare" in domain:
        tips.append("List certifications and licenses prominently near the top of your resume.")

    return tips[:5]


def generate_rsg_summary(resume_text: str, jd_text: str, domain: str,
                          template_id: str, is_fresher: bool) -> str:
    """SlotFiller-style summary generator using resume content."""
    # Extract name (first non-empty line heuristic)
    lines = [l.strip() for l in resume_text.split('\n') if l.strip()]
    name = lines[0] if lines else "The candidate"

    # Extract experience years
    import re
    exp_match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s*)?(experience|exp)', resume_text, re.I)
    years = exp_match.group(1) if exp_match else None

    # Build summary
    if is_fresher or years is None:
        exp_phrase = "a motivated fresher"
        exp_clause = "with a strong academic foundation"
    else:
        exp_phrase = f"an experienced professional with {years}+ years"
        exp_clause = "of hands-on industry experience"

    summary = (
        f"{name} is {exp_phrase} {exp_clause} in the {domain} domain. "
        f"Demonstrates strong alignment with the target role requirements, "
        f"combining technical expertise with practical problem-solving skills. "
    )

    if is_fresher:
        summary += (
            "Eager to contribute to a dynamic team environment while continuing "
            "to develop professional skills through real-world application."
        )
    else:
        summary += (
            "Proven track record of delivering results in fast-paced environments "
            "with a focus on quality and continuous improvement."
        )

    return summary


# ─── PDF READER ────────────────────────────────────────────────────────────────
def read_pdf(path: str) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        return "[PyMuPDF not installed. Run: pip install pymupdf]\nPaste resume text manually."
    except Exception as e:
        return f"[Error reading PDF: {e}]"


# ─── GUI ───────────────────────────────────────────────────────────────────────
class UnifiedModelGUI:
    status_lbl: typing.Any
    resume_box: typing.Any
    jd_box: typing.Any
    run_btn: typing.Any
    score_frame: typing.Any
    score_val: typing.Any
    band_lbl: typing.Any
    meta_lbl: typing.Any
    notebook: typing.Any
    kw_box: typing.Any
    fb_box: typing.Any
    rsg_box: typing.Any
    raw_box: typing.Any

    def __init__(self, root):
        self.root = root
        self.root.title("Unified Model — Real-Time Test Interface")
        self.root.geometry("1280x820")
        self.root.configure(bg="#0f1117")
        self.root.resizable(True, True)

        self._model_loaded = False
        self._loading = False

        self._build_ui()
        self._async_load_model()

    # ── UI BUILD ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg="#0f1117")
        hdr.pack(fill="x", padx=20, pady=(16, 4))

        tk.Label(hdr, text="⚡ Unified Model Test Interface",
                 font=("Courier New", 18, "bold"),
                 fg="#38bdf8", bg="#0f1117").pack(side="left")

        self.status_lbl = tk.Label(hdr, text="● Loading model…",
                                   font=("Courier New", 11),
                                   fg="#fbbf24", bg="#0f1117")
        self.status_lbl.pack(side="right")

        # Main paned layout
        paned = tk.PanedWindow(self.root, orient="horizontal",
                               bg="#0f1117", sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=16, pady=8)

        # ── LEFT: Inputs ──────────────────────────────────────────────────────
        left = tk.Frame(paned, bg="#0f1117")
        paned.add(left, minsize=480)

        # Resume section
        res_hdr = tk.Frame(left, bg="#0f1117")
        res_hdr.pack(fill="x", pady=(0, 4))
        tk.Label(res_hdr, text="📄 RESUME",
                 font=("Courier New", 12, "bold"),
                 fg="#94a3b8", bg="#0f1117").pack(side="left")
        tk.Button(res_hdr, text="Upload PDF",
                  command=self._upload_pdf,
                  font=("Courier New", 10),
                  bg="#1e293b", fg="#38bdf8",
                  relief="flat", cursor="hand2",
                  padx=10, pady=3).pack(side="right")

        self.resume_box = scrolledtext.ScrolledText(
            left, height=16, font=("Courier New", 10),
            bg="#1e293b", fg="#e2e8f0", insertbackground="#38bdf8",
            relief="flat", wrap="word", padx=8, pady=8
        )
        self.resume_box.pack(fill="both", expand=True)
        self.resume_box.insert("1.0", "Paste your resume text here, or upload a PDF above…")
        self.resume_box.bind("<FocusIn>", self._clear_placeholder_resume)

        # JD section
        tk.Label(left, text="💼 JOB DESCRIPTION",
                 font=("Courier New", 12, "bold"),
                 fg="#94a3b8", bg="#0f1117").pack(anchor="w", pady=(12, 4))

        self.jd_box = scrolledtext.ScrolledText(
            left, height=12, font=("Courier New", 10),
            bg="#1e293b", fg="#e2e8f0", insertbackground="#38bdf8",
            relief="flat", wrap="word", padx=8, pady=8
        )
        self.jd_box.pack(fill="both", expand=True)
        self.jd_box.insert("1.0", "Paste the job description here…")
        self.jd_box.bind("<FocusIn>", self._clear_placeholder_jd)

        # Run button
        self.run_btn = tk.Button(
            left, text="▶  RUN ANALYSIS",
            command=self._run_analysis,
            font=("Courier New", 13, "bold"),
            bg="#0284c7", fg="white",
            relief="flat", cursor="hand2",
            padx=16, pady=10, state="disabled"
        )
        self.run_btn.pack(fill="x", pady=(12, 0))

        # ── RIGHT: Results ────────────────────────────────────────────────────
        right = tk.Frame(paned, bg="#0f1117")
        paned.add(right, minsize=480)

        tk.Label(right, text="📊 RESULTS",
                 font=("Courier New", 12, "bold"),
                 fg="#94a3b8", bg="#0f1117").pack(anchor="w", pady=(0, 6))

        # Score card
        self.score_frame = tk.Frame(right, bg="#1e293b",
                                     highlightbackground="#334155",
                                     highlightthickness=1)
        self.score_frame.pack(fill="x", pady=(0, 10))

        score_inner = tk.Frame(self.score_frame, bg="#1e293b")
        score_inner.pack(padx=16, pady=12)

        self.score_val = tk.Label(score_inner, text="—",
                                   font=("Courier New", 48, "bold"),
                                   fg="#38bdf8", bg="#1e293b")
        self.score_val.pack()

        self.band_lbl = tk.Label(score_inner, text="Awaiting analysis…",
                                  font=("Courier New", 13),
                                  fg="#64748b", bg="#1e293b")
        self.band_lbl.pack()

        self.meta_lbl = tk.Label(score_inner, text="",
                                  font=("Courier New", 10),
                                  fg="#94a3b8", bg="#1e293b")
        self.meta_lbl.pack(pady=(4, 0))

        # Results tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook", background="#0f1117", borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background="#1e293b", foreground="#94a3b8",
                        font=("Courier New", 10, "bold"),
                        padding=[12, 6])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", "#0284c7")],
                  foreground=[("selected", "white")])

        self.notebook = ttk.Notebook(right, style="Dark.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        # Tab 1 — Keywords
        kw_frame = tk.Frame(self.notebook, bg="#0f1117")
        self.notebook.add(kw_frame, text="Missing Keywords")
        self.kw_box = scrolledtext.ScrolledText(
            kw_frame, font=("Courier New", 11),
            bg="#1e293b", fg="#e2e8f0",
            relief="flat", wrap="word", padx=10, pady=10, state="disabled"
        )
        self.kw_box.pack(fill="both", expand=True)

        # Tab 2 — Feedback
        fb_frame = tk.Frame(self.notebook, bg="#0f1117")
        self.notebook.add(fb_frame, text="Feedback")
        self.fb_box = scrolledtext.ScrolledText(
            fb_frame, font=("Courier New", 11),
            bg="#1e293b", fg="#e2e8f0",
            relief="flat", wrap="word", padx=10, pady=10, state="disabled"
        )
        self.fb_box.pack(fill="both", expand=True)

        # Tab 3 — RSG Summary
        rsg_frame = tk.Frame(self.notebook, bg="#0f1117")
        self.notebook.add(rsg_frame, text="RSG Summary")
        self.rsg_box = scrolledtext.ScrolledText(
            rsg_frame, font=("Courier New", 11),
            bg="#1e293b", fg="#e2e8f0",
            relief="flat", wrap="word", padx=10, pady=10, state="disabled"
        )
        self.rsg_box.pack(fill="both", expand=True)

        # Tab 4 — Raw JSON
        raw_frame = tk.Frame(self.notebook, bg="#0f1117")
        self.notebook.add(raw_frame, text="Raw Output")
        self.raw_box = scrolledtext.ScrolledText(
            raw_frame, font=("Courier New", 10),
            bg="#1e293b", fg="#64748b",
            relief="flat", wrap="word", padx=10, pady=10, state="disabled"
        )
        self.raw_box.pack(fill="both", expand=True)

    # ── MODEL LOADING ─────────────────────────────────────────────────────────
    def _async_load_model(self):
        def _load():
            try:
                load_model()
                self._model_loaded = True
                self.root.after(0, self._on_model_ready)
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda msg=err_msg: self._on_model_error(msg))
        threading.Thread(target=_load, daemon=True).start()

    def _on_model_ready(self):
        self.status_lbl.config(text="● Model ready", fg="#22c55e")
        self.run_btn.config(state="normal")

    def _on_model_error(self, err):
        self.status_lbl.config(text="● Model load failed", fg="#ef4444")
        messagebox.showerror("Model Load Error",
            f"Could not load unified model.\n\nError: {err}\n\n"
            f"Check that weights exist at:\n{WEIGHTS_PATH}")

    # ── INTERACTIONS ──────────────────────────────────────────────────────────
    def _upload_pdf(self):
        path = filedialog.askopenfilename(
            title="Select Resume PDF",
            filetypes=[("PDF files", "*.pdf"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        if path.endswith(".pdf"):
            text = read_pdf(path)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        self.resume_box.delete("1.0", "end")
        self.resume_box.insert("1.0", text)
        self.status_lbl.config(text=f"● Loaded: {os.path.basename(path)}", fg="#38bdf8")

    def _clear_placeholder_resume(self, _):
        if self.resume_box.get("1.0", "end-1c").startswith("Paste your resume"):
            self.resume_box.delete("1.0", "end")

    def _clear_placeholder_jd(self, _):
        if self.jd_box.get("1.0", "end-1c").startswith("Paste the job"):
            self.jd_box.delete("1.0", "end")

    def _run_analysis(self):
        if self._loading:
            return
        resume = self.resume_box.get("1.0", "end-1c").strip()
        jd = self.jd_box.get("1.0", "end-1c").strip()

        if len(resume) < 50:
            messagebox.showwarning("Input Required", "Please enter resume text (min 50 characters).")
            return
        if len(jd) < 30:
            messagebox.showwarning("Input Required", "Please enter job description (min 30 characters).")
            return

        self._loading = True
        self.run_btn.config(state="disabled", text="⏳  Analysing…")
        self.status_lbl.config(text="● Running inference…", fg="#fbbf24")

        def _infer():
            try:
                result = run_inference(resume, jd)
                self.root.after(0, lambda r=result: self._display_results(r))
            except Exception as e:
                err_msg = str(e)
                tb = traceback.format_exc()
                self.root.after(0, lambda m=err_msg, t=tb: self._on_infer_error(m, t))

        threading.Thread(target=_infer, daemon=True).start()

    def _display_results(self, r: dict):
        self._loading = False
        self.run_btn.config(state="normal", text="▶  RUN ANALYSIS")
        self.status_lbl.config(text="● Analysis complete", fg="#22c55e")

        # Score
        score = r["ats_score"]
        self.score_val.config(text=f"{score:.1f}", fg=r["band_color"])
        self.band_lbl.config(text=r["band"], fg=r["band_color"])
        fresher_tag = "  [FRESHER]" if r["is_fresher"] else ""
        self.meta_lbl.config(
            text=f"Domain: {r['domain_name']}  ({r['domain_confidence']}% confidence){fresher_tag}"
        )

        # Keywords tab
        self._write_box(self.kw_box, self._format_keywords(r["missing_keywords"]))

        # Feedback tab
        self._write_box(self.fb_box, self._format_feedback(r["feedback"]))

        # RSG tab
        self._write_box(self.rsg_box, self._format_rsg(r.get("rsg")))

        # Raw tab
        self._write_box(self.raw_box, json.dumps(r, indent=2))

        # Switch to most relevant tab
        self.notebook.select(0)

    def _on_infer_error(self, err: str, tb: str):
        self._loading = False
        self.run_btn.config(state="normal", text="▶  RUN ANALYSIS")
        self.status_lbl.config(text="● Inference error", fg="#ef4444")
        messagebox.showerror("Inference Error", f"{err}\n\nTraceback:\n{tb[:800]}")

    # ── FORMATTERS ────────────────────────────────────────────────────────────
    def _format_keywords(self, kw: dict) -> str:
        hard = kw.get("hard_skills", [])
        soft = kw.get("soft_skills", [])
        lines = ["── MISSING HARD SKILLS ──────────────────\n"]
        if hard:
            for k in hard:
                lines.append(f"  ✗  {k}")
        else:
            lines.append("  ✓  No critical hard skills missing")
        lines += ["\n\n── MISSING SOFT SKILLS ──────────────────\n"]
        if soft:
            for k in soft:
                lines.append(f"  ✗  {k}")
        else:
            lines.append("  ✓  No soft skills gaps detected")
        return "\n".join(lines)

    def _format_feedback(self, feedback: list) -> str:
        if not feedback:
            return "No feedback generated."
        lines = ["── ACTIONABLE RECOMMENDATIONS ───────────\n"]
        for i, tip in enumerate(feedback, 1):
            lines.append(f"  {i}.  {tip}\n")
        return "\n".join(lines)

    def _format_rsg(self, rsg) -> str:
        if rsg is None:
            return "RSG head not available or label map not found.\nCheck: model\\unified_model\\rsg_label_mapping.json"
        lines = [
            "── RSG TEMPLATE SELECTED ────────────────\n",
            f"  Template ID : {rsg['template_id']}",
            f"  Confidence  : {rsg['confidence']}%\n",
            "── GENERATED PROFESSIONAL SUMMARY ──────\n",
            f"  {rsg['summary']}"
        ]
        return "\n".join(lines)

    @staticmethod
    def _write_box(box: scrolledtext.ScrolledText, text: str):
        box.config(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.config(state="disabled")


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Unified Model Real-Time Test GUI")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Weights      : {WEIGHTS_PATH}")
    print(f"  Label map    : {RSG_LABEL_MAP}")
    print("=" * 60)
    print("  Starting GUI… (model loads in background)")
    print()

    if not os.path.exists(WEIGHTS_PATH):
        print(f"  WARNING: Weights not found at {WEIGHTS_PATH}")
        print("  Ensure you are running from Desktop\\ats")

    root = tk.Tk()
    app = UnifiedModelGUI(root)
    root.mainloop()
