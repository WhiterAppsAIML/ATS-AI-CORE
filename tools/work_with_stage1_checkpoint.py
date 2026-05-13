from __future__ import annotations

"""Utility script to load and run the unified model from Stage 1 checkpoint.

Run from repository root:
    python tools/work_with_stage1_checkpoint.py --pretty

This script also tolerates the common typo:
    stage1_checkpoint.weights.h  -> stage1_checkpoint.weights.h5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DOMAIN_NAMES = [
    "IT/Software",
    "Non-IT/Management",
    "Design/Creative",
    "Healthcare",
    "Finance/Banking",
    "Legal",
    "Education",
]


def _bootstrap_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    ats_core = repo_root / "ats-ai-core"

    # The unified model expects legacy tf.keras behavior with TF Hub KerasLayer.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    if "TFHUB_CACHE_DIR" not in os.environ:
        repo_cache = repo_root / "tfhub_cache"
        core_cache = ats_core / "tfhub_cache"

        # Prefer a pre-populated cache to avoid network dependency.
        if (core_cache / "063d866c06683311b44b4992fd46003be952409c" / "saved_model.pb").exists():
            os.environ["TFHUB_CACHE_DIR"] = str(core_cache)
        else:
            os.environ["TFHUB_CACHE_DIR"] = str(repo_cache)

    # Make ats-ai-core importable from this script location.
    sys.path.insert(0, str(ats_core))
    sys.path.insert(0, str(ats_core / "src"))

    return repo_root, ats_core


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Stage 1 checkpoint weights and run unified model inference.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("ats-ai-core/model/unified_model/stage1_checkpoint.weights.h5"),
        help="Path to Stage 1 weights (.weights.h5).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("ats-ai-core/model/unified_model/rsg_label_mapping.json"),
        help="Path to RSG label mapping JSON.",
    )
    parser.add_argument(
        "--resume-text",
        default="Software engineer with Python, SQL, and API development experience.",
        help="Resume text input.",
    )
    parser.add_argument(
        "--jd-text",
        default="Looking for a backend Python developer with SQL and API design experience.",
        help="Job description text input.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print JSON output in indented form.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive prompt after first inference pass.",
    )
    return parser.parse_args()


def _resolve_path(path_value: Path, repo_root: Path, ats_core: Path) -> Path:
    """Resolve a possibly-relative path against common project roots."""
    candidates: list[Path] = []

    if path_value.is_absolute():
        candidates.append(path_value)
    else:
        candidates.extend(
            [
                repo_root / path_value,
                ats_core / path_value,
                path_value,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Tolerate typo: .weights.h -> .weights.h5
    for candidate in candidates:
        text = str(candidate)
        if text.endswith(".weights.h"):
            fixed = Path(text + "5")
            if fixed.exists():
                print(
                    f"WARNING: '{candidate}' not found. Using '{fixed}' instead.",
                    file=sys.stderr,
                )
                return fixed

    return candidates[0]


def _load_label_map(mapping_path: Path) -> dict[int, str]:
    if not mapping_path.exists():
        return {}

    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "idx_to_id" in raw and isinstance(raw["idx_to_id"], dict):
        return {int(idx): str(template_id) for idx, template_id in raw["idx_to_id"].items()}

    if isinstance(raw, dict) and "id_to_idx" in raw and isinstance(raw["id_to_idx"], dict):
        return {int(class_idx): str(template_id) for template_id, class_idx in raw["id_to_idx"].items()}

    return {}


def _find_local_mobileuse_model(repo_root: Path, ats_core: Path) -> Path | None:
    # TF Hub stores the USE mobile module under this deterministic hash.
    module_hash = "063d866c06683311b44b4992fd46003be952409c"
    candidates = [
        ats_core / "tfhub_cache" / module_hash,
        repo_root / "tfhub_cache" / module_hash,
    ]

    for candidate in candidates:
        if (candidate / "saved_model.pb").exists():
            return candidate
    return None


def _load_weights_robust(model: Any, primary_path: Path, ats_core: Path) -> tuple[Path, str]:
    """Load weights with practical fallback strategy across file variants."""
    candidates = [primary_path]
    alt_path = ats_core / "model" / "unified_model" / primary_path.name
    if alt_path.exists() and alt_path != primary_path:
        candidates.append(alt_path)

    errors: list[str] = []

    for candidate in candidates:
        try:
            model.load_weights(str(candidate))
            return candidate, "strict"
        except Exception as exc:  # pragma: no cover - runtime dependent
            errors.append(f"strict load failed for {candidate}: {exc}")

    for candidate in candidates:
        try:
            model.load_weights(str(candidate), by_name=True, skip_mismatch=True)
            print(
                f"WARNING: Loaded with partial match from '{candidate}' (skip_mismatch=True).",
                file=sys.stderr,
            )
            return candidate, "partial"
        except Exception as exc:  # pragma: no cover - runtime dependent
            errors.append(f"partial load failed for {candidate}: {exc}")

    raise RuntimeError("\n".join(errors))


def _predict(model: Any, resume_text: str, jd_text: str, idx_to_template: dict[int, str]) -> dict[str, Any]:
    import numpy as np
    import tensorflow as tf

    outputs = model(
        {
            "resume_text": tf.constant([resume_text]),
            "jd_text": tf.constant([jd_text]),
        },
        training=False,
    )

    ats_score = float(outputs[0].numpy()[0][0]) * 100.0
    domain_probs = outputs[1].numpy()[0]
    rsg_probs = outputs[2].numpy()[0]

    domain_idx = int(np.argmax(domain_probs))
    rsg_idx = int(np.argmax(rsg_probs))
    template_id = idx_to_template.get(rsg_idx, f"template_{rsg_idx}")

    return {
        "ats_score": round(ats_score, 2),
        "domain": {
            "index": domain_idx,
            "name": DOMAIN_NAMES[domain_idx],
            "confidence": round(float(domain_probs[domain_idx]) * 100.0, 2),
        },
        "rsg": {
            "index": rsg_idx,
            "template_id": template_id,
            "confidence": round(float(rsg_probs[rsg_idx]) * 100.0, 2),
        },
    }


def _print_json(payload: dict[str, Any], pretty: bool) -> None:
    if pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))


def main() -> int:
    repo_root, ats_core = _bootstrap_paths()
    args = _parse_args()

    weights_path = _resolve_path(args.weights, repo_root, ats_core)
    mapping_path = _resolve_path(args.mapping, repo_root, ats_core)

    if not weights_path.exists():
        print(f"ERROR: Stage 1 weights file not found: {weights_path}")
        print("Hint: expected model/unified_model/stage1_checkpoint.weights.h5")
        return 1

    import tensorflow as tf
    from src.unified_engine import unified_model as unified_model_module

    local_mobileuse_path = _find_local_mobileuse_model(repo_root, ats_core)
    if local_mobileuse_path is not None:
        # Force local model path to avoid online TF Hub resolution.
        unified_model_module.USE_URL = str(local_mobileuse_path)

    try:
        model = unified_model_module.build_unified_model()
        loaded_from, load_mode = _load_weights_robust(model, weights_path, ats_core)
    except OSError as exc:
        msg = str(exc)
        if "does not appear to be a valid module" in msg or "saved_model" in msg:
            print("ERROR: TensorFlow Hub module download/cache appears corrupted.")
            print(f"Details: {msg}")
            print("Fix steps:")
            print("  1) c:/Users/saini/Desktop/ats/final_venv/Scripts/python.exe fix_tfhub_cache.py")
            print("  2) Re-run this script.")
            return 2
        raise
    except Exception as exc:
        print("ERROR: Could not load Stage 1 checkpoint with current architecture.")
        print(f"Details: {exc}")
        return 3
    idx_to_template = _load_label_map(mapping_path)

    first_result = _predict(model, args.resume_text, args.jd_text, idx_to_template)
    payload = {
        "repo_root": str(repo_root),
        "weights": str(loaded_from),
        "load_mode": load_mode,
        "mapping": str(mapping_path),
        "model_name": model.name,
        "outputs": [output.name for output in model.outputs],
        "result": first_result,
    }
    _print_json(payload, args.pretty)

    if args.interactive:
        print("\nInteractive mode started. Press Enter on resume text to exit.")
        while True:
            resume_text = input("\nResume text: ").strip()
            if not resume_text:
                break
            jd_text = input("Job description text: ").strip()
            if not jd_text:
                print("Job description text cannot be empty.")
                continue

            result = _predict(model, resume_text, jd_text, idx_to_template)
            _print_json({"result": result}, args.pretty)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
