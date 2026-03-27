"""
train_domain_classifier.py — Trains the Domain Categorizer model.
Uses a FastText-style architecture (Embedding + GlobalAveragePooling) 
for high TFLite compatibility and low latency.
"""

import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import PROCESSED_DIR, ATS_MODEL_DIR, DOMAIN_LABELS, RANDOM_SEED, BATCH_SIZE, EPOCHS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 512
EMBED_DIM = 64

def build_model(num_classes):
    """Builds a TFLite-friendly sequence classification model."""
    model = models.Sequential([
        layers.Input(shape=(1,), dtype=tf.string),
        layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode='int',
            output_sequence_length=MAX_LEN
        ),
        layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    logger.info("Loading processed domain data...")
    data_path = PROCESSED_DIR / "domain_training_data.csv"
    if not data_path.exists():
        logger.error("Training data not found at %s. Run prepare_domain_data.py first.", data_path)
        return

    df = pd.read_csv(data_path)
    logger.info("Loaded %d samples.", len(df))

    X = df['clean_text'].astype(str).tolist()
    y = df['domain_idx'].values

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y
    )
    
    # Convert to numpy for TF
    X_train = np.array(X_train)
    X_val = np.array(X_val)

    num_classes = len(DOMAIN_LABELS)
    logger.info("Building model for %d classes...", num_classes)
    model = build_model(num_classes)
    
    # Adapting the TextVectorization layer to the training text
    logger.info("Adapting TextVectorization (building vocabulary)...")
    vectorizer = model.layers[0]
    vectorizer.adapt(X_train)
    
    # Train
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # Save model
    ATS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ATS_MODEL_DIR / "domain_classifier_v1"
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Basic Evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info("Validation Accuracy: %.4f", val_acc)

if __name__ == "__main__":
    main()
