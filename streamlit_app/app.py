import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from src.preprocessing.label_encoder import DISEASES
from src.explainability.gradcam import make_gradcam_heatmap, overlay_heatmap

MODEL_PATH = "saved_models/chest_xray_model.keras"


st.set_page_config(page_title="Chest X-ray Disease Detection")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()


st.title("Chest X-ray Disease Detection System")

st.markdown("""
This application uses a **Deep Learning DenseNet121 model**
to detect **14 thoracic diseases** from chest X-ray images.
""")


uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    st.image(image, caption="Uploaded Image", width="stretch")

    img_resized = cv2.resize(image, (224,224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    predictions = model.predict(img_input)[0]

    st.subheader("Disease Probabilities")

    for disease, prob in zip(DISEASES, predictions):
        st.write(f"{disease}: {prob:.3f}")

    heatmap = make_gradcam_heatmap(
        img_input,
        model,
        last_conv_layer_name="conv5_block16_concat"
    )

    overlay = overlay_heatmap(heatmap, img_resized)

    st.subheader("GradCAM Visualization")

    st.image(overlay, caption="Model Attention Heatmap")