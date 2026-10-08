"""Pepper leaf disease detector - Streamlit app.

Upload a photo of a bell pepper leaf and the CNN trained in the notebook
predicts whether it is healthy or has bacterial spot.
"""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 256
MODEL_FILES = ["plant_disease_model.keras", "plant_disease_model-1.keras", "plant_disease_model.h5"]

# Same order as the dataset folders (alphabetical), which is how Keras numbered the classes.
CLASS_NAMES = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"]
LABELS = {
    "Pepper__bell___Bacterial_spot": "Bacterial spot (diseased)",
    "Pepper__bell___healthy": "Healthy",
}

st.set_page_config(page_title="Pepper Leaf Disease Detector", page_icon="🌿")


@st.cache_resource
def load_model():
    """Load the trained model once and reuse it for every visitor."""
    folder = Path(__file__).parent
    for name in MODEL_FILES:
        if (folder / name).exists():
            return tf.keras.models.load_model(folder / name, compile=False)
    return None


def predict(model, image):
    """Return one probability per class for a PIL image."""
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    # Pixels stay in 0-255: the model rescales them itself in its first layer.
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    return model.predict(batch, verbose=0)[0]


st.title("🌿 Pepper Leaf Disease Detector")
st.write(
    "Upload a photo of a **bell pepper leaf**. A convolutional neural network "
    "will tell you whether the leaf looks healthy or shows bacterial spot."
)

model = load_model()
if model is None:
    st.error("Model file not found. Put `plant_disease_model.keras` in the same folder as `app.py`.")
    st.stop()

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Waiting for an image.")
else:
    image = Image.open(uploaded)
    probabilities = predict(model, image)
    best = int(np.argmax(probabilities))
    label = LABELS[CLASS_NAMES[best]]
    confidence = float(probabilities[best])

    left, right = st.columns(2)
    with left:
        st.image(image, caption="Your image", use_container_width=True)
    with right:
        st.subheader("Result")
        if CLASS_NAMES[best].endswith("healthy"):
            st.success(f"{label}")
        else:
            st.error(f"{label}")
        st.metric("Confidence", f"{confidence:.1%}")
        for name, p in zip(CLASS_NAMES, probabilities):
            st.progress(float(p), text=f"{LABELS[name]}: {float(p):.1%}")

st.divider()
st.caption(
    "The model was trained only on bell pepper leaves from the PlantVillage dataset. "
    "It will still give an answer for any other picture, but that answer means nothing. "
    "This is a student project, not agricultural advice."
)
