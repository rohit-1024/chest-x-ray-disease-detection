"""
Dataset loading pipeline for NIH Chest X-ray project
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.preprocessing.label_encoder import encode_labels
from src.preprocessing.image_loader import load_image


def load_dataset(csv_path, image_dir, batch_size=32):
    """
    Create TensorFlow training and validation datasets.
    """

    df = pd.read_csv(csv_path)

    # encode labels
    df["encoded_labels"] = df["Finding Labels"].apply(encode_labels)

    # add image path
    df["image_path"] = df["Image Index"].apply(
        lambda x: f"{image_dir}/{x}"
    )

    # split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_labels = np.array(train_df["encoded_labels"].tolist())
    val_labels = np.array(val_df["encoded_labels"].tolist())

    # create tf datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_df["image_path"].values, train_labels)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_df["image_path"].values, val_labels)
    )

    train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset