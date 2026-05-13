from __future__ import annotations

"""Utility script to load and run the unified model from best_unified_weights.h5.

Run from repository root:
    python tools/work_with_best_unified_weights.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _bootstrap_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    ats_core = repo_root / "ats-ai-core"

    # The unified model expects legacy tf.keras behavior with TF Hub KerasLayer.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TFHUB_CACHE_DIR", str(repo_root / "tfhub_cache"))

    # Make ats-ai-core importable from this script location.
    sys.path.insert(0, str(ats_core))
    sys.path.insert(0, str(ats_core / "src"))

    return repo_root, ats_core


def _parse_args(ats_core: Path) -> argparse.Namespace:
    default_weights = ats_core / "model" / "unified_model" / "best_unified_weights.h5"
    default_mapping = ats_core / "model" / "unified_model" / "rsg_label_mapping.json"

    parser = argparse.ArgumentParser(
        description="Load unified model weights and run one inference pass.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=default_weights,
        help="Path to best_unified_weights.h5",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=default_mapping,
        help="Path to rsg_label_mapping.json",
    )
    parser.add_argument(
        "--resume-text",
        default="Software engineer with Python, SQL, and API development experience.",
        help="Resume text input",
    )
    parser.add_argument(
        "--jd-text",
        default="Looking for a backend Python developer with SQL and API design experience.",
        help="Job description text input",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print JSON output in indented form",
    )
    return parser.parse_args()


def _load_label_map(mapping_path: Path) -> dict[int, str]:
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # Stored as {template_id: class_index}; convert for prediction lookup.
    return {int(class_index): template_id for template_id, class_index in raw.items()}


def main() -> int:
    repo_root, ats_core = _bootstrap_paths()
    args = _parse_args(ats_core)

    if not args.weights.exists():
        print(f"ERROR: weights file not found: {args.weights}")
        return 1

    import numpy as np
    import tensorflow as tf
    from src.unified_engine.unified_model import build_unified_model

    domain_names = [
        "IT/Software",
        "Non-IT/Management",
        "Design/Creative",
        "Healthcare",
        "Finance/Banking",
        "Legal",
        "Education",
    ]

    model = build_unified_model()
    model.load_weights(str(args.weights))

    outputs = model(
        {
            "resume_text": tf.constant([args.resume_text]),
            "jd_text": tf.constant([args.jd_text]),
        },
        training=False,
    )

    ats_score = float(outputs[0].numpy()[0][0]) * 100.0
    domain_probs = outputs[1].numpy()[0]
    rsg_probs = outputs[2].numpy()[0]

    domain_idx = int(np.argmax(domain_probs))
    rsg_idx = int(np.argmax(rsg_probs))

    idx_to_template = _load_label_map(args.mapping)
    top_rsg_template = idx_to_template.get(rsg_idx, f"template_{rsg_idx}")

    result: dict[str, Any] = {
        "repo_root": str(repo_root),
        "weights": str(args.weights),
        "mapping": str(args.mapping),
        "model_name": model.name,
        "outputs": [output.name for output in model.outputs],
        "ats_score": round(ats_score, 2),
        "domain": {
            "index": domain_idx,
            "name": domain_names[domain_idx],
            "confidence": round(float(domain_probs[domain_idx]) * 100.0, 2),
        },
        "rsg": {
            "index": rsg_idx,
            "template_id": top_rsg_template,
            "confidence": round(float(rsg_probs[rsg_idx]) * 100.0, 2),
        },
    }

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
