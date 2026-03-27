"""
T-09 · src/encoding/use_lite_encoder.py

Wraps the Universal Sentence Encoder Lite (USE Lite) from TF Hub.
Provides a clean interface for single-text encoding, batch encoding,
and cosine similarity computation used throughout the ATS pipeline.

The encoder is loaded once and frozen by default (transfer learning).
Call encoder.unfreeze() only if MAE exceeds 10.0 after full training.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from src.config import EMBEDDING_DIM, USE_LITE_URL

logger = logging.getLogger(__name__)


class USELiteEncoder:
    """Wrapper around USE Lite for ATS pipeline encoding tasks.

    Attributes:
        embed: The loaded TF Hub KerasLayer.
        frozen: Whether the encoder weights are frozen.
    """

    def __init__(
        self,
        hub_url: str = USE_LITE_URL,
        frozen: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Load USE Lite from TF Hub.

        Args:
            hub_url: TF Hub URL for USE Lite.
            frozen: If True, encoder weights are non-trainable.
            cache_dir: Optional local path to cache the downloaded model.
        """
        if cache_dir:
            import os
            os.environ["TFHUB_CACHE_DIR"] = str(cache_dir)

        logger.info("Loading USE Lite from %s (frozen=%s)", hub_url, frozen)
        self.embed: hub.KerasLayer = hub.KerasLayer(
            hub_url,
            input_shape=[],
            dtype=tf.string,
            trainable=not frozen,
            name="use_lite_encoder",
        )
        self.frozen = frozen
        logger.info("USE Lite loaded. Embedding dim: %d", EMBEDDING_DIM)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a 512-dimensional embedding.

        Args:
            text: Input text string.

        Returns:
            Numpy array of shape (512,).
        """
        tensor = tf.constant([text])
        embedding: tf.Tensor = self.embed(tensor)
        return embedding.numpy()[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of text strings efficiently.

        Args:
            texts: List of text strings.

        Returns:
            Numpy array of shape (len(texts), 512).
        """
        tensor = tf.constant(texts)
        embeddings: tf.Tensor = self.embed(tensor)
        return embeddings.numpy()

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text_a: First text string.
            text_b: Second text string.

        Returns:
            Cosine similarity in range [-1.0, 1.0].
        """
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)
        return float(_cosine_similarity(vec_a, vec_b))

    def similarity_batch(
        self,
        resume_texts: list[str],
        jd_texts: list[str],
    ) -> np.ndarray:
        """Compute pairwise cosine similarities for aligned lists.

        Args:
            resume_texts: List of resume text strings.
            jd_texts: List of JD text strings (same length as resume_texts).

        Returns:
            1-D numpy array of cosine similarity scores.
        """
        if len(resume_texts) != len(jd_texts):
            raise ValueError(
                f"resume_texts length ({len(resume_texts)}) must equal "
                f"jd_texts length ({len(jd_texts)})"
            )
        r_vecs = self.encode_batch(resume_texts)
        j_vecs = self.encode_batch(jd_texts)
        # Row-wise cosine similarity
        norms_r = np.linalg.norm(r_vecs, axis=1, keepdims=True) + 1e-9
        norms_j = np.linalg.norm(j_vecs, axis=1, keepdims=True) + 1e-9
        r_norm = r_vecs / norms_r
        j_norm = j_vecs / norms_j
        return np.sum(r_norm * j_norm, axis=1)

    def unfreeze(self) -> None:
        """Unfreeze encoder weights for fine-tuning.

        Only call this if score MAE exceeds 10.0 after full training with
        frozen encoder, per RULES.md frozen-encoder-first rule.
        """
        self.embed.trainable = True
        self.frozen = False
        logger.warning(
            "USE Lite encoder UNFROZEN. This should only be done if MAE > 10.0."
        )

    def freeze(self) -> None:
        """Re-freeze encoder weights."""
        self.embed.trainable = False
        self.frozen = True
        logger.info("USE Lite encoder re-frozen.")

    def as_keras_layer(self) -> hub.KerasLayer:
        """Return the underlying KerasLayer for use inside a Keras model.

        Returns:
            The hub.KerasLayer instance.
        """
        return self.embed


# ── Standalone cosine similarity ─────────────────────────────────────────────

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity scalar.
    """
    norm_a = np.linalg.norm(vec_a) + 1e-9
    norm_b = np.linalg.norm(vec_b) + 1e-9
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ── Module-level singleton (lazy) ────────────────────────────────────────────
_encoder_instance: Optional[USELiteEncoder] = None


def get_encoder(frozen: bool = True) -> USELiteEncoder:
    """Return a module-level singleton encoder instance.

    Loads the encoder once and reuses it for all subsequent calls
    within the same process, avoiding repeated TF Hub downloads.

    Args:
        frozen: Passed to :class:`USELiteEncoder` on first init only.

    Returns:
        The singleton :class:`USELiteEncoder` instance.
    """
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = USELiteEncoder(frozen=frozen)
    return _encoder_instance
