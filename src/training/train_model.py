"""
Training pipeline for Chest X-ray disease detection model
"""

import tensorflow as tf

from src.data.dataset_loader import load_dataset
from src.models.build_model import build_model


CSV_PATH = "data/raw/nih_sample/sample_labels.csv"
IMAGE_DIR = "data/raw/nih_sample/images"

BATCH_SIZE = 32
EPOCHS = 10


def train():

    print("Loading dataset...")

    train_ds, val_ds = load_dataset(
        CSV_PATH,
        IMAGE_DIR,
        batch_size=BATCH_SIZE
    )

    print("Building model...")

    model = build_model()

    print("Compiling model...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    print("Starting training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print("Saving model...")

    model.save("saved_models/chest_xray_model.keras")

    print("Training complete!")

    return history


if __name__ == "__main__":
    train()