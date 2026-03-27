"""
T-14 · evaluation/ats_eval.py

Evaluation script for the trained ATS model. Computes MAE, RMSE,
score-band accuracy, and domain classification F1. Generates a
predicted-vs-actual scatter plot. Prints pass/fail against targets.

Usage:
    python evaluation/ats_eval.py
"""

import logging
import os
import sys
from pathlib import Path

# Must be set BEFORE importing tensorflow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Ensure project root is on sys.path so `src.*` is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    ATS_MODEL_DIR,
    EVALUATION_DIR,
    LABELED_DIR,
    TARGET_DOMAIN_F1,
    TARGET_MAE,
    get_score_band,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error on 0–100 scale.

    Args:
        y_true: Ground-truth scores (0–100).
        y_pred: Predicted scores (0–100).

    Returns:
        MAE float.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error on 0–100 scale.

    Args:
        y_true: Ground-truth scores (0–100).
        y_pred: Predicted scores (0–100).

    Returns:
        RMSE float.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_band_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions landing in the correct score band.

    Args:
        y_true: Ground-truth scores (0–100).
        y_pred: Predicted scores (0–100).

    Returns:
        Band accuracy in [0.0, 1.0].
    """
    true_bands = [get_score_band(float(s)) for s in y_true]
    pred_bands = [get_score_band(float(s)) for s in y_pred]
    correct = sum(t == p for t, p in zip(true_bands, pred_bands))
    return correct / len(true_bands)


def compute_domain_f1(
    y_true_domain: np.ndarray,
    y_pred_domain: np.ndarray,
) -> float:
    """Macro-averaged F1 score for domain classification.

    Args:
        y_true_domain: Ground-truth domain indices.
        y_pred_domain: Predicted domain indices.

    Returns:
        Macro F1 float.
    """
    from sklearn.metrics import f1_score  # noqa: PLC0415
    return float(f1_score(y_true_domain, y_pred_domain, average="macro", zero_division=0))


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on test data and return predictions.

    Args:
        model: Loaded Keras model.
        test_df: DataFrame with resume_text and jd_text columns.
        batch_size: Inference batch size.

    Returns:
        Tuple (score_preds_0_100, domain_preds) as numpy arrays.
    """
    resume_texts = test_df["resume_text"].astype(str).tolist()
    jd_texts     = test_df["jd_text"].astype(str).tolist()

    score_preds  = []
    domain_preds = []

    for i in range(0, len(resume_texts), batch_size):
        batch_r = np.array(resume_texts[i: i + batch_size])
        batch_j = np.array(jd_texts[i: i + batch_size])
        preds = model.predict(
            {"resume_text": batch_r, "jd_text": batch_j},
            verbose=0,
        )
        score_out = preds[0]           # (batch, 1) sigmoid
        domain_out = preds[1]          # (batch, 7) softmax probs
        score_preds.append(score_out.flatten())
        domain_preds.append(np.argmax(domain_out, axis=1))

    scores  = np.concatenate(score_preds) * 100.0   # scale to 0–100
    domains = np.concatenate(domain_preds)
    return scores, domains


# ── Plot helper ───────────────────────────────────────────────────────────────

def save_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save a predicted-vs-actual scatter plot.

    Args:
        y_true: Ground-truth scores (0–100).
        y_pred: Predicted scores (0–100).
        output_path: Path to save the PNG file.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("matplotlib not installed - skipping scatter plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.35, s=10, color="#2E75B6")
    ax.plot([0, 100], [0, 100], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("True ATS Score")
    ax.set_ylabel("Predicted ATS Score")
    ax.set_title("ATS Model - Predicted vs Actual Scores")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Scatter plot saved to %s", output_path)


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_ats_model(
    model_path: Path = ATS_MODEL_DIR / "final_model_weights.h5",
    test_csv: Path = LABELED_DIR / "test_split.csv",
) -> dict[str, float]:
    """Evaluate the saved ATS model on a held-out test set.

    Args:
        model_path: Path to saved Keras model weights.
        test_csv: Path to test split CSV with resume_text, jd_text,
            score (0–100), domain_index columns.

    Returns:
        Dict with keys: mae, rmse, band_accuracy, domain_f1.
    """
    from src.ats_engine.model import build_ats_model
    logger.info("Loading model weights from %s", model_path)
    model = build_ats_model()
    model.load_weights(str(model_path))

    logger.info("Loading test data from %s", test_csv)
    test_df = pd.read_csv(test_csv)
    test_df = test_df.dropna(subset=["resume_text", "jd_text", "score"])
    test_df = test_df[test_df["domain_index"] >= 0]

    y_true_scores  = test_df["score"].clip(0, 100).values
    y_true_domains = test_df["domain_index"].astype(int).values

    logger.info("Running inference on %d test samples ...", len(test_df))
    y_pred_scores, y_pred_domains = run_inference(model, test_df)

    mae           = compute_mae(y_true_scores, y_pred_scores)
    rmse          = compute_rmse(y_true_scores, y_pred_scores)
    band_acc      = compute_band_accuracy(y_true_scores, y_pred_scores)
    domain_f1     = compute_domain_f1(y_true_domains, y_pred_domains)

    results = {
        "mae":           round(mae, 4),
        "rmse":          round(rmse, 4),
        "band_accuracy": round(band_acc, 4),
        "domain_f1":     round(domain_f1, 4),
    }

    # ── Per-domain F1 ────────────────────────────────────────────────────────
    from sklearn.metrics import f1_score as _f1_score, classification_report
    from src.config import DOMAIN_LABELS

    per_domain_f1 = {}
    for idx, name in sorted(DOMAIN_LABELS.items()):
        mask = y_true_domains == idx
        if mask.sum() > 0:
            f1_val = float(_f1_score(
                (y_true_domains[mask] == idx).astype(int),
                (y_pred_domains[mask] == idx).astype(int),
                average="binary", zero_division=0,
            ))
            per_domain_f1[name] = round(f1_val, 4)
        else:
            per_domain_f1[name] = None

    # Better per-domain: use the per-class F1 from classification_report
    unique_labels = sorted(set(y_true_domains) | set(y_pred_domains))
    target_names = [DOMAIN_LABELS.get(i, f"Domain {i}") for i in unique_labels]
    cls_report = classification_report(
        y_true_domains, y_pred_domains,
        labels=unique_labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    per_domain_f1 = {}
    for idx in sorted(DOMAIN_LABELS.keys()):
        name = DOMAIN_LABELS[idx]
        if name in cls_report:
            per_domain_f1[name] = round(cls_report[name]["f1-score"], 4)
        else:
            per_domain_f1[name] = None

    results["per_domain_f1"] = per_domain_f1

    # ── Fresher fairness (detect via resume text patterns) ──────────────────
    import re
    _fresher_re = re.compile(
        r"fresher|fresh\s*graduate|entry[- ]level|intern(?:ship)?|"
        r"0\s*years?\s*(?:of\s*)?(?:experience|professional)|"
        r"final[- ]year\s*student|looking for entry|recent(?:ly)?\s*graduat|"
        r"no\s*(?:prior\s*)?experience|trainee|articleship|pupillage",
        re.IGNORECASE,
    )
    fresher_mask = test_df["resume_text"].astype(str).apply(
        lambda t: bool(_fresher_re.search(t))
    )
    experienced_mean = None
    fresher_mean = None
    if fresher_mask.sum() > 0 and (~fresher_mask).sum() > 0:
        experienced_mean = float(y_pred_scores[~fresher_mask.values].mean())
        fresher_mean = float(y_pred_scores[fresher_mask.values].mean())

    # ── Print report ─────────────────────────────────────────────────────────
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 55)
    print("  ATS MODEL EVALUATION REPORT (Post Synthetic Data)")
    print("=" * 55)
    print(f"  Samples evaluated : {len(test_df)}")
    print(f"  Score MAE         : {mae:.2f}  (target < {TARGET_MAE})  "
          f"{'[PASS]' if mae < TARGET_MAE else '[FAIL]'}")
    print(f"  Score RMSE        : {rmse:.2f}")
    print(f"  Band Accuracy     : {band_acc:.2%}")
    print(f"  Domain F1 (macro) : {domain_f1:.4f}  (target > {TARGET_DOMAIN_F1})  "
          f"{'[PASS]' if domain_f1 > TARGET_DOMAIN_F1 else '[FAIL]'}")

    print("\n  PER-DOMAIN F1")
    print("  " + "-" * 40)
    for name, f1_val in per_domain_f1.items():
        marker = ""
        if name == "Legal":
            marker = "  <- was 0.68"
        elif name == "Education":
            marker = "  <- was 0.62"
        if f1_val is not None:
            print(f"    {name:22s}: {f1_val:.4f}{marker}")
        else:
            print(f"    {name:22s}: N/A{marker}")

    # Previous baseline
    prev_f1 = 0.7791
    prev_mae = 2.33
    prev_legal_f1 = 0.68
    prev_edu_f1 = 0.62

    print("\n  DELTA FROM PREVIOUS RUN")
    print("  " + "-" * 40)
    print(f"    Domain F1 change  : {prev_f1:.4f} -> {domain_f1:.4f}  "
          f"({'+' if domain_f1 >= prev_f1 else ''}{domain_f1 - prev_f1:.4f})")
    if per_domain_f1.get("Legal") is not None:
        lf1 = per_domain_f1["Legal"]
        print(f"    Legal F1 change   : {prev_legal_f1:.2f} -> {lf1:.4f}  "
              f"({'+' if lf1 >= prev_legal_f1 else ''}{lf1 - prev_legal_f1:.4f})")
    if per_domain_f1.get("Education") is not None:
        ef1 = per_domain_f1["Education"]
        print(f"    Education F1 change: {prev_edu_f1:.2f} -> {ef1:.4f}  "
              f"({'+' if ef1 >= prev_edu_f1 else ''}{ef1 - prev_edu_f1:.4f})")
    print(f"    Score MAE change  : {prev_mae:.2f} -> {mae:.2f}  "
          f"({'+' if mae >= prev_mae else ''}{mae - prev_mae:.2f})"
          f"  {'[REGRESSION]' if mae > 5.0 else '[OK]'}")

    if experienced_mean is not None and fresher_mean is not None:
        gap = abs(experienced_mean - fresher_mean)
        print(f"\n  FRESHER FAIRNESS")
        print("  " + "-" * 40)
        print(f"    Experienced Score : {experienced_mean:.2f}")
        print(f"    Fresher Score     : {fresher_mean:.2f}")
        print(f"    Gap               : {gap:.1f} pts  "
              f"{'[PASS]' if gap <= 20 else '[FAIL]'} (target <= 20)")

    # Overall status
    passed = domain_f1 > TARGET_DOMAIN_F1 and mae < TARGET_MAE
    print(f"\n  OVERALL STATUS")
    print("  " + "-" * 40)
    if passed:
        print("    [X] READY - Domain F1 > 0.85 and MAE < 8.0. Awaiting Sai's review.")
    else:
        failures = []
        if domain_f1 <= TARGET_DOMAIN_F1:
            failures.append(f"Domain F1={domain_f1:.4f} <= {TARGET_DOMAIN_F1}")
        if mae >= TARGET_MAE:
            failures.append(f"MAE={mae:.2f} >= {TARGET_MAE}")
        print(f"    [ ] NOT READY - {'; '.join(failures)}. Awaiting Sai's instructions.")

    print("=" * 55 + "\n")

    # ── Save scatter plot ────────────────────────────────────────────────────
    plot_path = EVALUATION_DIR / "plots" / "predicted_vs_actual.png"
    save_scatter_plot(y_true_scores, y_pred_scores, plot_path)

    # ── Save report CSV ───────────────────────────────────────────────────────
    report_path = EVALUATION_DIR / "eval_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(report_path, index=False)
    logger.info("Eval report saved to %s", report_path)

    return results


if __name__ == "__main__":
    evaluate_ats_model()
