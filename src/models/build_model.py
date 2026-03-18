"""
Model architecture for Chest X-ray disease detection
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import DenseNet121


NUM_CLASSES = 14
IMG_SIZE = (224, 224, 3)


def build_model():

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE
    )

    base_model.trainable = False

    x = base_model.output

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="sigmoid"
    )(x)

    model = models.Model(
        inputs=base_model.input,
        outputs=outputs
    )

    return model