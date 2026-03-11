import json
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

MODEL_PATH = "models/dog_breed_classifier.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMAGE_SIZE = (224, 224)

_imagenet_model = None

def is_dog(img_array):
    global _imagenet_model
    if _imagenet_model is None:
        _imagenet_model = MobileNetV2(weights="imagenet", include_top=True)
    
    img_processed = preprocess_input(np.copy(img_array))
    preds = _imagenet_model.predict(img_processed, verbose=0)
    top_class_index = np.argmax(preds[0])
    
    return 151 <= top_class_index <= 268

def load_and_prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main(image_path):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    img_array = load_and_prepare_image(image_path)

    if not is_dog(img_array):
        print("\n⚠️ La imagen no parece ser un perro.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict(img_array)[0]
    top_indices = np.argsort(preds)[::-1][:3]

    print("\nTop 3 predicciones:")
    for idx in top_indices:
        breed_name = class_names[idx].split('-', 1)[-1]
        print(f"{breed_name}: {preds[idx] * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py ruta/a/la/imagen.jpg")
        sys.exit(1)

    main(sys.argv[1])