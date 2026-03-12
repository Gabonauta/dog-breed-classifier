import os
import sys
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "dog_breed_classifier.keras")
CLASS_NAMES_PATH = os.path.join(ROOT_DIR, "models", "class_names.json")
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="Dog Breed Classifier", layout="centered")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error("No se encontró el archivo del modelo.")
        st.stop()

    if not os.path.exists(CLASS_NAMES_PATH):
        st.error("No se encontró el archivo class_names.json.")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🐶 Clasificador de Razas de Perro")
st.write("Sube una imagen de un perro y el modelo predecirá la raza más probable.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_container_width=True)

    model, class_names = load_artifacts()
    img_array = prepare_image(image)

    preds = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:3]

    st.subheader("Top 3 predicciones")
    for idx in top_indices:
        breed_name = class_names[idx].split('-', 1)[-1]
        st.write(f"**{breed_name}** — {preds[idx] * 100:.2f}%")