"""
Image loading and preprocessing utilities
for NIH Chest X-ray dataset.
"""

import tensorflow as tf


# Image configuration
IMG_SIZE = (224, 224)


def load_image(image_path, label):
    """
    Load and preprocess a chest X-ray image.

    Steps:
    1. Read image from disk
    2. Decode PNG
    3. Resize to 224x224
    4. Normalize pixel values (0–1)

    Returns:
        image tensor, label tensor
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(image, channels=3)

    image = tf.image.resize(image, IMG_SIZE)

    image = image / 255.0

    return image, label