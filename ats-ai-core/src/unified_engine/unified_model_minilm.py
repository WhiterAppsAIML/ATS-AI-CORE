"""ats-ai-core/src/unified_engine/unified_model_minilm.py

Unified ATS + Domain + RSG model with MiniLM-L6-v2 encoder (INJECTION-E1).

Inputs — 4 × int32 [1, 128]:
  resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask

Outputs:
  ats_score    [1, 1]   sigmoid
  domain_probs [1, 7]   softmax   (from jd_emb)
  rsg_template [1, 46]  softmax   (from resume_emb)

Run as __main__ to execute E1 validation:
  python -m src.unified_engine.unified_model_minilm
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_FILE      = Path(__file__).resolve()
_CORE_DIR  = _FILE.parents[2]   # ats-ai-core/
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from src.config import (
    EVALUATION_DIR,
    MODEL_DIR,
    MINILM_MODEL_NAME,
    MINILM_EMBEDDING_DIM,
    MINILM_MAX_SEQ_LEN,
    RSG_NUM_CLASSES,
)

SEQ_LEN     = MINILM_MAX_SEQ_LEN    # 128
EMB_DIM     = MINILM_EMBEDDING_DIM  # 384
NUM_DOMAINS = 7
NUM_RSG     = RSG_NUM_CLASSES       # 46

TFLITE_PATH = MODEL_DIR / "unified_model" / "unified_minilm_int8.tflite"
REPORT_PATH = EVALUATION_DIR / "e1_architecture_validation.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Pooling helper ────────────────────────────────────────────────────────────

def mean_pool_l2(last_hidden: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    """Attention-mask weighted mean pooling + L2 normalise → [B, D]."""
    mask    = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
    sum_emb = tf.reduce_sum(last_hidden * mask, axis=1)
    count   = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-9)
    return tf.nn.l2_normalize(sum_emb / count, axis=-1)


# ── Custom Keras encoder layer ─────────────────────────────────────────────────

class MiniLMEncoderLayer(tf.keras.layers.Layer):
    """Frozen MiniLM-L6-v2 encoder: ([input_ids, attention_mask]) → [B, 384]."""

    def __init__(self, bert_model, **kwargs):
        kwargs.setdefault("trainable", False)
        super().__init__(**kwargs)
        self._bert = bert_model
        self._bert.trainable = False

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        out = self._bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=tf.zeros_like(input_ids),
            training=False,
        )
        return mean_pool_l2(out.last_hidden_state, attention_mask)

    def get_config(self):
        return super().get_config()


# ── Keras model (for summary and training) ────────────────────────────────────

def build_unified_minilm_model(bert_model) -> tf.keras.Model:
    """Return unified Keras model with frozen MiniLM encoder and 3 output heads.

    Args:
        bert_model: HuggingFace TFBertModel instance (will be frozen).

    Returns:
        tf.keras.Model with inputs [resume_input_ids, resume_attention_mask,
        jd_input_ids, jd_attention_mask] and outputs [ats_score, domain_probs,
        rsg_template].
    """
    resume_ids  = tf.keras.Input((SEQ_LEN,), dtype=tf.int32, name="resume_input_ids")
    resume_mask = tf.keras.Input((SEQ_LEN,), dtype=tf.int32, name="resume_attention_mask")
    jd_ids      = tf.keras.Input((SEQ_LEN,), dtype=tf.int32, name="jd_input_ids")
    jd_mask     = tf.keras.Input((SEQ_LEN,), dtype=tf.int32, name="jd_attention_mask")

    encoder    = MiniLMEncoderLayer(bert_model, name="minilm_encoder")
    resume_emb = encoder([resume_ids,  resume_mask])   # [B, 384]
    jd_emb     = encoder([jd_ids,      jd_mask])       # [B, 384]

    # Feature concat: 384 + 384 + 1 + 1 = 770
    cosine_sim = tf.keras.layers.Dot(axes=1, normalize=True,  name="cosine_sim")([resume_emb, jd_emb])
    dot_prod   = tf.keras.layers.Dot(axes=1, normalize=False, name="dot_prod")  ([resume_emb, jd_emb])
    ats_feat   = tf.keras.layers.Concatenate(name="ats_features")([resume_emb, jd_emb, cosine_sim, dot_prod])

    # HEAD 1 — ATS Score  (770 → 256 → 64 → 1)
    x1        = tf.keras.layers.Dense(256, activation="relu",  name="ats_dense1")(ats_feat)
    x1        = tf.keras.layers.Dropout(0.3,                   name="ats_drop1")(x1)
    x1        = tf.keras.layers.Dense(64,  activation="relu",  name="ats_dense2")(x1)
    x1        = tf.keras.layers.Dropout(0.2,                   name="ats_drop2")(x1)
    ats_score = tf.keras.layers.Dense(1, activation="sigmoid", name="ats_score")(x1)

    # HEAD 2 — Domain  (jd_emb 384 → 256 → 128 → 7)
    x2           = tf.keras.layers.Dense(256, activation="relu",  name="dom_dense1")(jd_emb)
    x2           = tf.keras.layers.Dropout(0.3,                   name="dom_drop1")(x2)
    x2           = tf.keras.layers.Dense(128, activation="relu",  name="dom_dense2")(x2)
    x2           = tf.keras.layers.Dropout(0.2,                   name="dom_drop2")(x2)
    domain_probs = tf.keras.layers.Dense(7, activation="softmax", name="domain_probs")(x2)

    # HEAD 3 — RSG  (resume_emb 384 → 512→BN → 256→BN → 128→BN → 46)
    x3           = tf.keras.layers.Dense(512, activation="relu", name="rsg_dense1")(resume_emb)
    x3           = tf.keras.layers.BatchNormalization(           name="rsg_bn1")(x3)
    x3           = tf.keras.layers.Dropout(0.4,                  name="rsg_drop1")(x3)
    x3           = tf.keras.layers.Dense(256, activation="relu", name="rsg_dense2")(x3)
    x3           = tf.keras.layers.BatchNormalization(           name="rsg_bn2")(x3)
    x3           = tf.keras.layers.Dropout(0.3,                  name="rsg_drop2")(x3)
    x3           = tf.keras.layers.Dense(128, activation="relu", name="rsg_dense3")(x3)
    x3           = tf.keras.layers.BatchNormalization(           name="rsg_bn3")(x3)
    x3           = tf.keras.layers.Dropout(0.3,                  name="rsg_drop3")(x3)
    rsg_template = tf.keras.layers.Dense(46, activation="softmax", name="rsg_template")(x3)

    return tf.keras.Model(
        inputs=[resume_ids, resume_mask, jd_ids, jd_mask],
        outputs=[ats_score, domain_probs, rsg_template],
        name="unified_ats_minilm",
    )


# ── INT8 TFLite conversion ─────────────────────────────────────────────────────

class _InferenceModule(tf.Module):
    """Wraps the Keras model as a tf.Module for TFLite concrete-function export."""

    def __init__(self, keras_model: tf.keras.Model):
        super().__init__()
        self.model = keras_model

    @tf.function(input_signature=[
        tf.TensorSpec([1, SEQ_LEN], tf.int32, name="resume_input_ids"),
        tf.TensorSpec([1, SEQ_LEN], tf.int32, name="resume_attention_mask"),
        tf.TensorSpec([1, SEQ_LEN], tf.int32, name="jd_input_ids"),
        tf.TensorSpec([1, SEQ_LEN], tf.int32, name="jd_attention_mask"),
    ])
    def infer(self, resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask):
        ats, dom, rsg = self.model(
            [resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask],
            training=False,
        )
        return ats, dom, rsg


def _detect_flex_ops(model_bytes: bytes) -> list[str]:
    return sorted({
        m.group(0).decode("ascii")
        for m in re.finditer(rb"Flex[A-Za-z0-9_]+", model_bytes)
    })


def _get_op_inventory(model_bytes: bytes) -> list[str]:
    try:
        interp = tf.lite.Interpreter(model_content=model_bytes)
        interp.allocate_tensors()
        if hasattr(interp, "_get_ops_details"):
            return sorted({op["op_name"] for op in interp._get_ops_details()})
    except Exception:
        pass
    return []


def convert_to_int8_tflite(keras_model: tf.keras.Model, num_calib: int = 200) -> bytes:
    """Convert unified Keras model to INT8 TFLite; return raw bytes."""
    module      = _InferenceModule(keras_model)
    concrete_fn = module.infer.get_concrete_function()

    rng = np.random.default_rng(42)

    def representative_dataset():
        for _ in range(num_calib):
            r_ids  = rng.integers(1, 30522, size=(1, SEQ_LEN)).astype(np.int32)
            r_mask = np.ones((1, SEQ_LEN), dtype=np.int32)
            j_ids  = rng.integers(1, 30522, size=(1, SEQ_LEN)).astype(np.int32)
            j_mask = np.ones((1, SEQ_LEN), dtype=np.int32)
            yield [r_ids, r_mask, j_ids, j_mask]

    log.info("Converting to INT8 TFLite…")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], module)
    converter.optimizations                  = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset         = representative_dataset
    converter.target_spec.supported_ops      = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    return converter.convert()


# ── E1 validation entry point ─────────────────────────────────────────────────

def _run_e1_validation() -> None:
    try:
        from transformers import TFAutoModel
    except ImportError as exc:
        log.error("transformers library not installed: %s", exc)
        sys.exit(1)

    log.info("Loading TF model for %s…", MINILM_MODEL_NAME)
    bert_model = TFAutoModel.from_pretrained(MINILM_MODEL_NAME, from_pt=True)
    bert_model.trainable = False

    # ── Step 1: Keras model summary ──────────────────────────────────────────
    print("\n-- STEP 1: Keras model summary --")
    model = build_unified_minilm_model(bert_model)
    model.summary()

    trainable_params     = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable_params = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))

    # -- Step 2: INT8 TFLite dry-run -----------------------------------------
    print("\n-- STEP 2: INT8 TFLite dry-run conversion --")
    failures: list[str] = []

    try:
        tflite_bytes = convert_to_int8_tflite(model)
    except Exception as exc:
        log.error("TFLite conversion failed: %s", exc, exc_info=True)
        failures.append(f"Conversion exception: {exc}")
        _write_report(
            {"summary": {}, "conversion": {"error": str(exc)}, "validation_passed": False}
        )
        _print_result(failures)
        sys.exit(1)

    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_bytes)
    size_mb  = TFLITE_PATH.stat().st_size / (1024 * 1024)
    flex_ops = _detect_flex_ops(tflite_bytes)
    ops      = _get_op_inventory(tflite_bytes)

    log.info("TFLite size : %.2f MB", size_mb)
    log.info("Flex ops    : %s", flex_ops or "none")

    # ── Output shape check ───────────────────────────────────────────────────
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    out_shapes = sorted([[int(x) for x in d["shape"]] for d in interp.get_output_details()])
    expected   = sorted([[1, 1], [1, 7], [1, 46]])
    shapes_ok  = out_shapes == expected

    log.info("Output shapes : %s  %s", out_shapes, "[PASS]" if shapes_ok else "[FAIL]")

    if flex_ops:
        failures.append(f"Flex ops detected: {flex_ops}")
    if not shapes_ok:
        failures.append(f"Output shapes {out_shapes} != expected {expected}")

    # ── Report ───────────────────────────────────────────────────────────────
    report = {
        "model_summary": {
            "name":               model.name,
            "trainable_params":   trainable_params,
            "non_trainable_params": non_trainable_params,
            "total_params":       trainable_params + non_trainable_params,
        },
        "conversion": {
            "tflite_path":    str(TFLITE_PATH),
            "size_mb":        round(size_mb, 3),
            "flex_ops":       flex_ops,
            "flex_pass":      len(flex_ops) == 0,
            "output_shapes":  out_shapes,
            "shapes_pass":    shapes_ok,
        },
        "op_inventory":       ops,
        "validation_passed":  len(failures) == 0,
    }
    _write_report(report)
    _print_result(failures, report)

    if failures:
        sys.exit(1)


def _write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report written -> %s", REPORT_PATH)


def _print_result(failures: list[str], report: dict | None = None) -> None:
    print("\n" + "=" * 62)
    print("  INJECTION-E1 ARCHITECTURE VALIDATION REPORT")
    print("=" * 62)
    if report:
        s = report.get("model_summary", {})
        c = report.get("conversion", {})
        print(f"  Model           : {s.get('name', 'N/A')}")
        print(f"  Trainable params: {s.get('trainable_params', '?'):,}")
        print(f"  TFLite size     : {c.get('size_mb', '?')} MB")
        print(f"  Flex ops        : {c.get('flex_ops') or 'none'}  "
              f"{'[PASS]' if c.get('flex_pass') else '[FAIL]'}")
        print(f"  Output shapes   : {c.get('output_shapes', '?')}  "
              f"{'[PASS]' if c.get('shapes_pass') else '[FAIL]'}")
        print(f"  Report          : {REPORT_PATH}")
    print("=" * 62)
    if failures:
        print("\n  E1 FAILED — Hard Stop:")
        for msg in failures:
            print(f"    - {msg}")
        print()
    else:
        print("\n  E1 PASSED - proceed to R0\n")


if __name__ == "__main__":
    _run_e1_validation()
