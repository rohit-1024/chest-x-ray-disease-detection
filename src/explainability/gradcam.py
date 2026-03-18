import tensorflow as tf
import numpy as np
import cv2


def get_last_conv_layer(model):
    """
    Automatically find the last convolution layer in the model.
    This avoids hardcoding layer names like 'conv5_block16_concat'.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate GradCAM heatmap.
    """

    # Auto-detect last conv layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model([img_array])

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    # Gradient of class output w.r.t feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Weight feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap)

    heatmap /= max_val

    return heatmap.numpy()


def overlay_heatmap(heatmap, image, alpha=0.4):
    """
    Overlay GradCAM heatmap on original image.
    """

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img