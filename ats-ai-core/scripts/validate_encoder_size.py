"""
validate_encoder_size.py — Pre-training sanity check.

Loads the USE Lite encoder from config.USE_LITE_URL, wraps it in a
minimal Keras model, converts to TFLite with Float16 quantisation,
and asserts the output file is < 30 MB.

Usage:
    python scripts/validate_encoder_size.py
"""

import os
import sys
import tempfile

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import tensorflow_hub as hub

# Ensure project root is on sys.path so `src.config` is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import USE_LITE_URL

MAX_SIZE_MB = 30.0


def main() -> None:
    print(f"Encoder URL : {USE_LITE_URL}")
    print(f"Max allowed : {MAX_SIZE_MB} MB")
    print()

    # ── Build a minimal model around the encoder ─────────────────────────
    text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    encoder = hub.KerasLayer(
        USE_LITE_URL,
        input_shape=[],
        dtype=tf.string,
        trainable=False,
        name="mobile_use_encoder",
    )
    embeddings = encoder(text_input)
    output = tf.keras.layers.Dense(1, name="dummy_output")(embeddings)
    model = tf.keras.Model(inputs=text_input, outputs=output)
    model.summary()

    # ── Convert to TFLite with Float16 quantisation ──────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    # ── Write to a temp file and measure size ────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        f.write(tflite_model)
        tflite_path = f.name

    size_bytes = os.path.getsize(tflite_path)
    size_mb = size_bytes / (1024 * 1024)

    print()
    print(f"TFLite file : {tflite_path}")
    print(f"Size        : {size_mb:.2f} MB")
    print()

    # ── Clean up ─────────────────────────────────────────────────────────
    os.remove(tflite_path)

    # ── Pass / Fail ──────────────────────────────────────────────────────
    if size_mb < MAX_SIZE_MB:
        print(f"✅ PASS — {size_mb:.2f} MB < {MAX_SIZE_MB} MB")
        sys.exit(0)
    else:
        print(f"❌ FAIL — {size_mb:.2f} MB >= {MAX_SIZE_MB} MB")
        sys.exit(1)


if __name__ == "__main__":
    main()
