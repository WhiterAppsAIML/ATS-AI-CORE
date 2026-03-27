"""
Realtime ATS Model Tester GUI
Run this file to open a window where you can paste a Resume and a Job Description (JD)
and get the ATS Model's predicted Score and Domain in realtime.
"""

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

# Set Keras legacy mode to allow loading the model
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Ensure the script can import from src
sys.path.append(str(Path(__file__).parent.resolve()))

from src.ats_engine.inference import run_ats_inference, _load_model

class ATSTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ATS Model Realtime Tester")
        self.root.geometry("800x600")
        
        self.model_ready = False
        
        self._build_ui()
        
        # Load model in background via inference module's cached loader
        self.status_var.set("Loading model... Please wait")
        self.predict_btn["state"] = tk.DISABLED
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # PanedWindow for split text areas
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Resume Frame
        resume_frame = ttk.LabelFrame(paned, text="Resume", padding="5")
        
        self.resume_text = tk.Text(resume_frame, wrap=tk.WORD, width=40)
        
        res_ctrl = ttk.Frame(resume_frame)
        res_ctrl.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(res_ctrl, text="Paste or upload text:").pack(side=tk.LEFT)
        ttk.Button(res_ctrl, text="Upload File", command=lambda: self.load_file(self.resume_text)).pack(side=tk.RIGHT)
        
        self.resume_text.pack(fill=tk.BOTH, expand=True)
        paned.add(resume_frame, weight=1)

        # JD Frame
        jd_frame = ttk.LabelFrame(paned, text="Job Description (JD)", padding="5")
        
        self.jd_text = tk.Text(jd_frame, wrap=tk.WORD, width=40)
        
        jd_ctrl = ttk.Frame(jd_frame)
        jd_ctrl.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(jd_ctrl, text="Paste or upload text:").pack(side=tk.LEFT)
        ttk.Button(jd_ctrl, text="Upload File", command=lambda: self.load_file(self.jd_text)).pack(side=tk.RIGHT)
        
        self.jd_text.pack(fill=tk.BOTH, expand=True)
        paned.add(jd_frame, weight=1)

        # Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)

        self.predict_btn = ttk.Button(control_frame, text="Predict Score", command=self.predict, style="Accent.TButton")
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(control_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Waiting to load model...")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Results area
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        
        # Top row: score, band, domain
        top_row = ttk.Frame(result_frame)
        top_row.pack(fill=tk.X)

        self.score_var = tk.StringVar(value="Score: -- / 100")
        self.domain_var = tk.StringVar(value="Domain: --")
        self.band_var = tk.StringVar(value="Band: --")
        self.fresher_var = tk.StringVar(value="")
        
        ttk.Label(top_row, textvariable=self.score_var, font=("Segoe UI", 14, "bold")).pack(anchor=tk.W)
        ttk.Label(top_row, textvariable=self.band_var, font=("Segoe UI", 12)).pack(anchor=tk.W)
        ttk.Label(top_row, textvariable=self.domain_var, font=("Segoe UI", 12)).pack(anchor=tk.W)
        ttk.Label(top_row, textvariable=self.fresher_var, font=("Segoe UI", 10, "italic")).pack(anchor=tk.W)

        # Feedback section
        ttk.Separator(result_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(result_frame, text="Feedback:", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        self.feedback_var = tk.StringVar(value="")
        ttk.Label(result_frame, textvariable=self.feedback_var, font=("Segoe UI", 10), wraplength=750, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 5))

        # Missing keywords section
        ttk.Label(result_frame, text="Missing Keywords:", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        self.keywords_var = tk.StringVar(value="")
        ttk.Label(result_frame, textvariable=self.keywords_var, font=("Segoe UI", 10), wraplength=750, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

    def _load_model(self):
        try:
            _load_model()  # triggers the lru_cache singleton in inference.py
            self.model_ready = True
            self.root.after(0, self._model_loaded)
        except Exception as e:
            self.root.after(0, self._model_load_error, str(e))

    def _model_loaded(self):
        self.status_var.set("Model Loaded. Ready.")
        self.predict_btn["state"] = tk.NORMAL

    def _model_load_error(self, err_msg):
        self.status_var.set("Error loading model!")
        messagebox.showerror("Model Load Error", f"Could not load the ATS model.\n\n{err_msg}")

    def clear(self):
        self.resume_text.delete("1.0", tk.END)
        self.jd_text.delete("1.0", tk.END)
        self.score_var.set("Score: -- / 100")
        self.domain_var.set("Domain: --")
        self.band_var.set("Band: --")
        self.fresher_var.set("")
        self.feedback_var.set("")
        self.keywords_var.set("")

    def load_file(self, text_widget):
        filepath = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("Text and PDF Files", "*.txt *.pdf"), ("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            return
            
        try:
            content = ""
            if filepath.lower().endswith(".pdf"):
                import pypdf
                reader = pypdf.PdfReader(filepath)
                content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            else:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Fallback for Windows files
                    with open(filepath, "r", encoding="cp1252") as f:
                        content = f.read()
            
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", content)
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file:\n{e}")

    def predict(self):
        if not self.model_ready:
            return
            
        resume = self.resume_text.get("1.0", tk.END).strip()
        jd = self.jd_text.get("1.0", tk.END).strip()
        
        if len(resume) < 10 or len(jd) < 10:
            messagebox.showwarning("Input Error", "Please provide a valid Resume and Job Description (at least 10 chars).")
            return
            
        self.status_var.set("Predicting...")
        self.predict_btn["state"] = tk.DISABLED
        self.root.update_idletasks()
        
        try:
            result = run_ats_inference(resume, jd)
            
            self.score_var.set(f"Score: {result['ats_score']:.1f} / 100")
            self.domain_var.set(f"Domain: {result['domain_name']} (idx {result['domain_index']})")
            self.band_var.set(f"Band: {result['score_band']}")
            self.fresher_var.set("(Fresher profile detected)" if result["is_fresher"] else "")

            # Display feedback
            feedback_lines = "\n".join(f"  \u2022 {fb}" for fb in result["feedback"])
            self.feedback_var.set(feedback_lines if feedback_lines else "No feedback available.")

            # Display missing keywords
            kw = result["missing_keywords"]
            kw_parts = []
            if kw["hard_skills"]:
                kw_parts.append(f"Hard Skills: {', '.join(kw['hard_skills'])}")
            if kw["soft_skills"]:
                kw_parts.append(f"Soft Skills: {', '.join(kw['soft_skills'])}")
            if kw["other"]:
                kw_parts.append(f"Other: {', '.join(kw['other'])}")
            self.keywords_var.set("\n".join(kw_parts) if kw_parts else "No missing keywords found.")

            self.status_var.set("Done.")
            
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))
            self.status_var.set("Error during inference")
        finally:
            self.predict_btn["state"] = tk.NORMAL

if __name__ == "__main__":
    root = tk.Tk()
    # Optional styling
    try:
        root.tk.call('tk', 'scaling', 1.5)
    except tk.TclError:
        pass
    app = ATSTesterApp(root)
    root.mainloop()
