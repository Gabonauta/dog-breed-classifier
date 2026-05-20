# Dog Breed Classifier

Clasificador de razas de perro con Deep Learning (TensorFlow/Keras) y Transfer Learning sobre MobileNetV2.

## Demo
- App en Streamlit: [https://dog-breed-classifier-multi.streamlit.app/](https://dog-breed-classifier-multi.streamlit.app/)

## Overview del proyecto
Este repositorio implementa un pipeline completo para clasificación multiclase de razas caninas:
- Descarga y preparación automática del dataset Stanford Dogs.
- Entrenamiento en dos fases (head training + fine-tuning).
- Evaluación con `classification_report` y matriz de confusión.
- Predicción desde consola (Top-3 probabilidades).
- Interfaz web con Streamlit y tabla de traducción EN/ES de razas.

Actualmente el proyecto está configurado para trabajar con **120 razas** (según `models/class_names.json`).

## Stack técnico
- Python
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- KaggleHub
- Streamlit

## Estructura del repositorio
```text
classifier/
├── app/
│   ├── prepare_data.py      # Descarga dataset y split train/val/test
│   ├── train.py             # Entrenamiento del modelo
│   ├── evaluate.py          # Métricas y matriz de confusión
│   ├── predict.py           # Predicción por CLI (Top-3)
│   ├── streamlit_app.py     # Interfaz web
│   └── translations.py      # Traducciones de razas EN -> ES
├── data/                    # Datos procesados (train, val, test)
├── models/
│   ├── dog_breed_classifier.keras
│   └── class_names.json
├── results/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── test_images/
├── requirements.txt
└── README.md
```

## Instalación
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install kagglehub scikit-learn pillow
```

## Ejecución rápida (sin reentrenar)
Si ya tienes `models/dog_breed_classifier.keras` y `models/class_names.json`:

1. Predicción por consola:
```bash
python3 app/predict.py test_images/chiguagua.jpg
```

2. Levantar app web:
```bash
streamlit run app/streamlit_app.py
```

## Flujo completo de entrenamiento
1. Preparar datos (descarga desde KaggleHub y split 70/15/15):
```bash
python3 app/prepare_data.py
```

2. Entrenar modelo:
```bash
python3 app/train.py
```

3. Evaluar en test y generar matriz de confusión:
```bash
python3 app/evaluate.py
```

## Detalles de entrenamiento
- Arquitectura base: `MobileNetV2` (`weights="imagenet"`, `include_top=False`).
- Tamaño de imagen: `224x224`.
- Batch size: `32`.
- Data augmentation: flip horizontal, rotación, zoom y contraste.
- Entrenamiento por etapas:
  - Etapa 1 (head): `EPOCHS_HEAD = 8`
  - Etapa 2 (fine-tuning): `EPOCHS_FINE = 7`
- Seed fija para reproducibilidad (`SEED = 42`).

## Configuración del dataset
En `app/prepare_data.py` puedes ajustar:
- `MAX_BREEDS`: limita el número de razas (por defecto `None`).
- `MAX_IMAGES_PER_BREED`: máximo de imágenes por raza (por defecto `250`).
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`.

## Outputs generados
- Modelo entrenado: `models/dog_breed_classifier.keras`
- Clases: `models/class_names.json`
- Curvas de entrenamiento: `results/training_curves.png`
- Matriz de confusión: `results/confusion_matrix.png`

## Notas
- La primera ejecución descarga pesos de ImageNet y el dataset, por lo que requiere internet.
- `predict.py` y `streamlit_app.py` incluyen un filtro previo `is_dog` para advertir cuando la imagen no parece contener un perro.
