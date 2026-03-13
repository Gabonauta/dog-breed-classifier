import os
import sys
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
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
    model = tf.keras.models.load_model(MODEL_PATH)
    imagenet_model = MobileNetV2(weights="imagenet", include_top=True)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, imagenet_model, class_names

def is_dog(imagenet_model, img_array):
    img_processed = preprocess_input(np.copy(img_array))
    preds = imagenet_model.predict(img_processed, verbose=0)
    top_class_index = np.argmax(preds[0])
    return 151 <= top_class_index <= 268

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

    model, imagenet_model, class_names = load_artifacts()
    img_array = prepare_image(image)

    if not is_dog(imagenet_model, img_array):
        st.warning("⚠️ La imagen no parece ser un perro. Por favor, sube una imagen de un perro.")
    else:
        preds = model.predict(img_array)[0]
        top_indices = np.argsort(preds)[::-1][:3]

        st.subheader("Top 3 predicciones")
        for idx in top_indices:
            breed_name = class_names[idx].split('-', 1)[-1]
            st.write(f"**{breed_name}** — {preds[idx] * 100:.2f}%")

    st.image(image, caption="Imagen subida", use_container_width=True)

st.divider()
st.subheader("Razas Soportadas")

# Fetch class names from cached artifacts
_, _, class_names = load_artifacts()

# Load translation mappings
sys.path.append(os.path.dirname(__file__))
from translations import BREED_TRANSLATIONS

table_data = []
for full_name in class_names:
    breed_en = full_name.split('-', 1)[-1]
    breed_es = BREED_TRANSLATIONS.get(breed_en, breed_en)
    table_data.append({"Nombre de Raza": breed_en, "Nombre de Raza en Español": breed_es})

st.dataframe(table_data, use_container_width=True, hide_index=True)