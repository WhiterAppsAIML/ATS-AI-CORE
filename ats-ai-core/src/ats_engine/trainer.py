"""
T-12 · src/ats_engine/trainer.py

Training loop for the ATS scoring model. Handles dataset loading,
train/val/test splitting (split BEFORE any feature extraction),
multi-task loss, early stopping, and model saving.
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    ATS_MODEL_DIR,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAINING_PAIRS_CSV,
    VALIDATION_SPLIT,
)

logger = logging.getLogger(__name__)


# ── Dataset preparation ───────────────────────────────────────────────────────

def load_training_data(
    csv_path: Path = TRAINING_PAIRS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training pairs and split into train / val / test sets.

    IMPORTANT: Split is done BEFORE any TF-IDF or feature extraction
    to prevent data leakage (per RULES.md).

    Args:
        csv_path: Path to training_pairs.csv.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info("Loading training pairs from %s", csv_path)
    df = pd.read_csv(csv_path)

    required = {"resume_text", "jd_text", "score", "domain_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training_pairs.csv missing columns: {missing}")

    # Drop rows with null score or text
    df = df.dropna(subset=["resume_text", "jd_text", "score"])
    df = df[df["domain_index"] >= 0]

    # Normalise score to [0, 1]
    df["score_norm"] = df["score"].clip(0, 100) / 100.0

    # Shuffle before splitting
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(df)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VALIDATION_SPLIT)

    test_df  = df.iloc[:n_test]
    val_df   = df.iloc[n_test: n_test + n_val]
    train_df = df.iloc[n_test + n_val:]

    logger.info(
        "Split: train=%d  val=%d  test=%d  (total=%d)",
        len(train_df), len(val_df), len(test_df), n,
    )
    return train_df, val_df, test_df


def _to_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    domain_class_weights: dict[int, float] | None = None,
) -> tf.data.Dataset:
    """Convert a DataFrame of pairs into a batched tf.data.Dataset.

    Args:
        df: DataFrame with resume_text, jd_text, score_norm, domain_index.
        batch_size: Number of samples per batch.
        shuffle: If True, shuffle the dataset before batching.
        domain_class_weights: Optional mapping domain_index → weight.
            When provided, per-sample weights are included so Keras
            applies class balancing via sample_weight (required for
            multi-output models where class_weight is unsupported).

    Returns:
        A batched :class:`tf.data.Dataset` yielding
        (inputs, targets) or (inputs, targets, sample_weights).
    """
    resumes  = df["resume_text"].astype(str).tolist()
    jds      = df["jd_text"].astype(str).tolist()
    scores   = df["score_norm"].astype(np.float32).tolist()
    domains  = df["domain_index"].astype(np.int32).tolist()

    inputs = {"resume_text": resumes, "jd_text": jds}
    targets = {"ats_score": scores, "domain_logits": domains}

    if domain_class_weights is not None:
        # Build per-sample weight arrays for each output head.
        # ats_score: uniform weight 1.0
        # domain_logits: class-balanced weight from domain_class_weights
        score_sw = [1.0] * len(df)
        domain_sw = [domain_class_weights.get(d, 1.0) for d in domains]
        dataset = tf.data.Dataset.from_tensor_slices((
            inputs, targets,
            {"ats_score": score_sw, "domain_logits": domain_sw},
        ))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(df), 10_000), seed=RANDOM_SEED)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _build_callbacks(model_dir: Path) -> list[tf.keras.callbacks.Callback]:
    """Build standard training callbacks.

    Args:
        model_dir: Directory to save the best model checkpoint.

    Returns:
        List of Keras callbacks.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_dir / "best_model_weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        str(model_dir / "training_log.csv"),
        append=True,
    )
    return [early_stop, checkpoint, reduce_lr, csv_logger]


# ── Main training function ────────────────────────────────────────────────────

def compute_domain_class_weights(domain_labels: list[int]) -> dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    from src.config import NUM_DOMAINS, DOMAIN_CLASS_WEIGHTS
    classes = np.unique(domain_labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(domain_labels))
    cw = dict(zip(classes.tolist(), weights.tolist()))
    # Keras requires consecutive keys 0..N-1; fill missing domains with weight 1.0
    for i in range(NUM_DOMAINS):
        if i not in cw:
            cw[i] = 1.0
    # Apply manual per-class weight overrides from config
    for idx in cw:
        cw[idx] *= DOMAIN_CLASS_WEIGHTS.get(idx, 1.0)
    for idx, w in sorted(cw.items()):
        logger.info("Domain %d class weight: %.4f", idx, w)
    return cw


def train(
    model: tf.keras.Model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    model_dir: Path = ATS_MODEL_DIR,
) -> tf.keras.callbacks.History:
    """Train the ATS model with multi-task loss.

    Args:
        model: Compiled Keras model from :func:`src.ats_engine.model.build_ats_model`.
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        batch_size: Samples per gradient update.
        epochs: Maximum training epochs (early stopping may cut short).
        model_dir: Directory to save checkpoints and logs.

    Returns:
        Keras :class:`tf.keras.callbacks.History` object.
    """
    logger.info(
        "Starting training: epochs=%d  batch_size=%d  train=%d  val=%d",
        epochs, batch_size, len(train_df), len(val_df),
    )

    domain_labels = train_df["domain_index"].astype(int).tolist()
    class_weights = compute_domain_class_weights(domain_labels)

    train_ds = _to_tf_dataset(train_df, batch_size=batch_size, shuffle=True,
                              domain_class_weights=class_weights)
    val_ds   = _to_tf_dataset(val_df,   batch_size=batch_size, shuffle=False)

    callbacks = _build_callbacks(model_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final weights (best weights already restored by EarlyStopping)
    final_path = model_dir / "final_model_weights.h5"
    model.save_weights(str(final_path))
    logger.info("Final model weights saved to %s", final_path)

    _log_training_summary(history)
    return history


def _log_training_summary(history: tf.keras.callbacks.History) -> None:
    """Print a concise training summary to the logger.

    Args:
        history: Keras History object returned by model.fit.
    """
    best_epoch = int(np.argmin(history.history.get("val_loss", [0])))
    best_mae   = history.history.get("val_ats_score_mae", [None])[best_epoch]
    best_acc   = history.history.get("val_domain_logits_acc", [None])[best_epoch]

    logger.info(
        "Training complete. Best epoch: %d | val MAE: %.4f (×100 = %.2f) | "
        "val domain acc: %.4f",
        best_epoch + 1,
        best_mae or 0,
        (best_mae or 0) * 100,
        best_acc or 0,
    )

    from src.config import TARGET_MAE, TARGET_DOMAIN_F1
    target_mae = TARGET_MAE / 100.0   # scale to 0.0-1.0 scale
    target_acc = TARGET_DOMAIN_F1
    if best_mae and best_mae > target_mae:
        logger.warning(
            "MAE %.4f exceeds target %.4f. Consider unfreezing the encoder "
            "(see RULES.md frozen-encoder-first rule).",
            best_mae, target_mae,
        )
    if best_acc and best_acc < target_acc:
        logger.warning(
            "Domain accuracy %.4f is below target %.4f.",
            best_acc, target_acc,
        )
