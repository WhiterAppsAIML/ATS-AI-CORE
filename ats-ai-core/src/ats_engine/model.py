"""
T-10 · src/ats_engine/model.py

Builds the Keras ATS scoring model.

Architecture:
  - Input: two string tensors (resume_text, jd_text)
  - Shared encoder: USE Lite (frozen by default)
  - Similarity head: cosine similarity → dense → sigmoid → ats_score
  - Domain classifier head: dense → softmax (7 classes) → domain_logits

Output contract (SACRED — never change without updating IO_SCHEMA.md):
  Output 0: ats_score   float32 [1]  score in [0.0, 1.0]
  Output 1: domain_logits float32 [7] domain class probabilities
"""

import logging
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import tensorflow_hub as hub

from src.config import (
    EMBEDDING_DIM, NUM_DOMAINS, USE_LITE_URL,
    LEARNING_RATE, SCORE_LOSS_WEIGHT, DOMAIN_LOSS_WEIGHT,
)

logger = logging.getLogger(__name__)


class _USEEncoderLayer(tf.keras.layers.Layer):
    """Wraps hub.KerasLayer so it works reliably with Keras Functional API."""

    def __init__(self, hub_url, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self._hub_layer = hub.KerasLayer(
            hub_url, input_shape=[], dtype=tf.string, trainable=trainable,
        )

    def call(self, inputs):
        return self._hub_layer(inputs)


def _cosine_similarity_layer(vec_a: tf.Tensor, vec_b: tf.Tensor) -> tf.Tensor:
    """Compute normalised cosine similarity between paired embedding vectors.

    Args:
        vec_a: Tensor of shape (batch, EMBEDDING_DIM).
        vec_b: Tensor of shape (batch, EMBEDDING_DIM).

    Returns:
        Tensor of shape (batch, 1) with values in [-1, 1].
    """
    vec_a = tf.math.l2_normalize(vec_a, axis=1)
    vec_b = tf.math.l2_normalize(vec_b, axis=1)
    cosine = tf.reduce_sum(vec_a * vec_b, axis=1, keepdims=True)
    return cosine


def build_ats_model(
    hub_url: str = USE_LITE_URL,
    frozen_encoder: bool = True,
) -> tf.keras.Model:
    """Build and compile the dual-head ATS Keras model.

    The model shares a single USE Lite encoder for both resume and JD
    inputs, then splits into a similarity head (score) and a domain
    classification head.

    Args:
        hub_url: TF Hub URL for USE Lite.
        frozen_encoder: If True, encoder weights are non-trainable.

    Returns:
        Compiled :class:`tf.keras.Model` with two outputs:
        ``ats_score`` and ``domain_logits``.
    """
    # ── Inputs ───────────────────────────────────────────────────────────────
    resume_input = tf.keras.Input(shape=(), dtype=tf.string, name="resume_text")
    jd_input     = tf.keras.Input(shape=(), dtype=tf.string, name="jd_text")

    # ── Shared encoder ────────────────────────────────────────────────────────
    encoder = _USEEncoderLayer(
        hub_url,
        trainable=not frozen_encoder,
        name="use_lite_encoder",
    )
    resume_embedding = encoder(resume_input)  # (batch, 512)
    jd_embedding     = encoder(jd_input)      # (batch, 512)

    # ── Similarity head ───────────────────────────────────────────────────────
    # Concatenate embeddings + cosine feature
    cosine_feature = tf.keras.layers.Lambda(
        lambda vecs: _cosine_similarity_layer(vecs[0], vecs[1]),
        name="cosine_sim",
    )([resume_embedding, jd_embedding])  # (batch, 1)
    dot_feature    = tf.keras.layers.Dot(axes=1, normalize=True)(
        [resume_embedding, jd_embedding]
    )  # (batch, 1)

    sim_concat = tf.keras.layers.Concatenate(name="similarity_features")(
        [resume_embedding, jd_embedding, cosine_feature, dot_feature]
    )  # (batch, 1026)

    sim_dense1 = tf.keras.layers.Dense(256, activation="relu", name="sim_dense1")(sim_concat)
    sim_drop1  = tf.keras.layers.Dropout(0.3, name="sim_drop1")(sim_dense1)
    sim_dense2 = tf.keras.layers.Dense(64, activation="relu", name="sim_dense2")(sim_drop1)
    sim_drop2  = tf.keras.layers.Dropout(0.2, name="sim_drop2")(sim_dense2)

    # Score output: sigmoid → [0.0, 1.0]
    ats_score = tf.keras.layers.Dense(1, activation="sigmoid", name="ats_score")(sim_drop2)

    # ── Domain classifier head ────────────────────────────────────────────────
    # Operates on JD embedding only (domain is a property of the job, not the candidate)
    dom_dense1  = tf.keras.layers.Dense(256, activation="relu", name="dom_dense1")(jd_embedding)
    dom_drop1   = tf.keras.layers.Dropout(0.3, name="dom_drop1")(dom_dense1)
    dom_dense2  = tf.keras.layers.Dense(128, activation="relu", name="dom_dense2")(dom_drop1)
    dom_drop2   = tf.keras.layers.Dropout(0.2, name="dom_drop2")(dom_dense2)

    # Logits output: 7-class softmax
    domain_logits = tf.keras.layers.Dense(
        NUM_DOMAINS, activation="softmax", name="domain_logits"
    )(dom_drop2)

    # ── Model assembly ────────────────────────────────────────────────────────
    model = tf.keras.Model(
        inputs=[resume_input, jd_input],
        outputs=[ats_score, domain_logits],
        name="ats_scoring_model",
    )

    # ── Compilation ───────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "ats_score":     "mean_absolute_error",
            "domain_logits": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "ats_score":     SCORE_LOSS_WEIGHT,
            "domain_logits": DOMAIN_LOSS_WEIGHT,
        },
        metrics={
            "ats_score":     [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "domain_logits": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        },
    )

    param_count = model.count_params()
    logger.info(
        "ATS model built. Total params: %s (trainable: %s)",
        f"{param_count:,}",
        f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}",
    )
    if param_count > 5_000_000:
        logger.warning(
            "Model has %d parameters — consider reducing dense layer sizes "
            "to stay within the 5M target for TFLite size constraints.",
            param_count,
        )

    return model
