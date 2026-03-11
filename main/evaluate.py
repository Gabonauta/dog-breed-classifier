import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "data/test"
MODEL_PATH = "models/dog_breed_classifier.keras"
CLASS_NAMES_PATH = "models/class_names.json"
CONF_MATRIX_PATH = "results/confusion_matrix.png"

os.makedirs("results", exist_ok=True)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
).prefetch(tf.data.AUTOTUNE)

model = load_model(MODEL_PATH)

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

print("\n=== Classification Report ===\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH, dpi=150)

print(f"\nMatriz de confusión guardada en: {CONF_MATRIX_PATH}")