# Dog Breed Classifier

Clasificador de razas de perro usando Deep Learning y Transfer Learning con MobileNetV2.

## DEMO
https://dog-breed-classifier-multi.streamlit.app/

## Características
- Clasificación multiclase de 10 razas
- Entrenamiento con TensorFlow/Keras
- Evaluación con accuracy, precision, recall y F1-score
- Interfaz web con Streamlit

## Estructura
- `train/prepare_data.py`: descarga y prepara el dataset
- `train/train.py`: entrenamiento del modelo
- `train/evaluate.py`: evaluación y matriz de confusión
- `train/predict.py`: predicción desde consola
- `train/web.py`: demo web

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
Preparar los datos:
```bash
python train/prepare_data.py
```

Entrenar el modelo:
```bash
python train/train.py
```

Evaluar el modelo:
```bash
python train/evaluate.py
```

Ejecutar la interfaz web:
```bash
streamlit run train/web.py
```
