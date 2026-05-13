import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Force tf.keras → tf-keras (Keras 2)
                                           # Required: hub.KerasLayer is incompatible
                                           # with Keras 3 KerasTensors in Functional API

# Set TFHUB_CACHE_DIR to project-local cache if not already set
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # → ats/
os.environ.setdefault("TFHUB_CACHE_DIR", str(_PROJECT_ROOT / "tfhub_cache"))

import tensorflow as tf
import tensorflow_hub as hub

from src.config import EMBEDDING_DIM, RSG_NUM_CLASSES, USE_LITE_URL

# ---------------------------------------------------------------------------
# MobileUSE — universal-sentence-encoder-mobile
#   Input:  raw tf.string
#   Output: 512-dim float32 embeddings
#   Size:   ~30-40 MB
# ---------------------------------------------------------------------------
USE_URL = USE_LITE_URL
USE_OUTPUT_DIM = EMBEDDING_DIM


def build_unified_model():
    # --- Inputs ---
    resume_input = tf.keras.Input(shape=(), dtype=tf.string, name="resume_text")
    jd_input     = tf.keras.Input(shape=(), dtype=tf.string, name="jd_text")

    # --- Shared MobileUSE encoder (frozen) ---
    encoder_layer = hub.KerasLayer(USE_URL, input_shape=[], dtype=tf.string,
                                   trainable=False, name="mobile_use_encoder")
    resume_emb = encoder_layer(resume_input)
    jd_emb     = encoder_layer(jd_input)

    # --- Feature engineering (identical to production ATS model) ---
    cosine_sim   = tf.keras.layers.Dot(axes=1, normalize=True,
                       name="cosine_sim")([resume_emb, jd_emb])
    dot_prod     = tf.keras.layers.Dot(axes=1, normalize=False,
                       name="dot_prod")([resume_emb, jd_emb])
    ats_features = tf.keras.layers.Concatenate(
                       name="ats_features")([resume_emb, jd_emb, cosine_sim, dot_prod])

    # --- HEAD 1: ATS Score (unchanged from production model) ---
    # Input to similarity head: 512 + 512 + 1 + 1 = 1026
    x1 = tf.keras.layers.Dense(256, activation="relu",  name="ats_dense1")(ats_features)
    x1 = tf.keras.layers.Dropout(0.3,                   name="ats_drop1")(x1)
    x1 = tf.keras.layers.Dense(64,  activation="relu",  name="ats_dense2")(x1)
    x1 = tf.keras.layers.Dropout(0.2,                   name="ats_drop2")(x1)
    ats_output = tf.keras.layers.Dense(1, activation="sigmoid",
                     name="ats_score")(x1)

    # --- HEAD 2: Domain Classifier (unchanged from production model) ---
    # Input to domain head: 512
    x2 = tf.keras.layers.Dense(256, activation="relu",  name="dom_dense1")(jd_emb)
    x2 = tf.keras.layers.Dropout(0.3,                   name="dom_drop1")(x2)
    x2 = tf.keras.layers.Dense(128, activation="relu",  name="dom_dense2")(x2)
    x2 = tf.keras.layers.Dropout(0.2,                   name="dom_drop2")(x2)
    domain_output = tf.keras.layers.Dense(7, activation="softmax",
                        name="domain_probs")(x2)

    # --- HEAD 3: RSG Template Classifier ---
    # Input to RSG head: 512
    # Architecture matches retrained summary_model.keras exactly
    x3 = tf.keras.layers.Dense(512, activation="relu",  name="rsg_dense1")(resume_emb)
    x3 = tf.keras.layers.BatchNormalization(             name="rsg_bn1")(x3)
    x3 = tf.keras.layers.Dropout(0.4,                   name="rsg_drop1")(x3)
    x3 = tf.keras.layers.Dense(256, activation="relu",  name="rsg_dense2")(x3)
    x3 = tf.keras.layers.BatchNormalization(             name="rsg_bn2")(x3)
    x3 = tf.keras.layers.Dropout(0.3,                   name="rsg_drop2")(x3)
    x3 = tf.keras.layers.Dense(128, activation="relu",  name="rsg_dense3")(x3)
    x3 = tf.keras.layers.BatchNormalization(             name="rsg_bn3")(x3)
    x3 = tf.keras.layers.Dropout(0.3,                   name="rsg_drop3")(x3)
    rsg_output = tf.keras.layers.Dense(RSG_NUM_CLASSES, activation="softmax",
                     name="rsg_template")(x3)

    model = tf.keras.Model(
        inputs=[resume_input, jd_input],
        outputs=[ats_output, domain_output, rsg_output],
        name="unified_ats_rsg_model"
    )
    return model

if __name__ == '__main__':
    model = build_unified_model()
    model.summary()
    print('BUILD OK — MobileUSE swap complete')
