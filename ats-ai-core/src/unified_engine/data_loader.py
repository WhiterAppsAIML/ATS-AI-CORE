import pandas as pd
import numpy as np
from pathlib import Path

def load_ats_data(csv_path: str, limit: int | None = None):
    """Returns (resume_texts, jd_texts, ats_scores_0to1, domain_labels)"""
    df = pd.read_csv(csv_path).dropna()
    if limit is not None:
        df = df.sample(min(limit, len(df)), random_state=42)
    score_col = "score" if "score" in df.columns else "ats_score"
    domain_col = "domain_index" if "domain_index" in df.columns else "domain_label"
    missing = [c for c in [score_col, domain_col, "resume_text", "jd_text"] if c not in df.columns]
    if missing:
        raise ValueError(f"ATS CSV missing columns: {missing}; found {list(df.columns)}")
    resume_texts  = df["resume_text"].astype(str).values
    jd_texts      = df["jd_text"].astype(str).values
    ats_scores    = (df[score_col].astype(float) / 100.0).values
    domain_labels = df[domain_col].astype(int).values
    return resume_texts, jd_texts, ats_scores, domain_labels

def load_rsg_data(csv_path: str):
    """Returns (profile_texts, template_indices_original_ids)"""
    df = pd.read_csv(csv_path).dropna()
    profile_texts    = df["profile_text"].astype(str).values
    template_indices = df["template_index"].astype(int).values
    return profile_texts, template_indices
