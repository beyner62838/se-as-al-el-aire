# ==============================================================================
# VISTAS DEL SISTEMA HÍBRIDO DE RECONOCIMIENTO DE SEÑAS ASL
# "Señas Al Aire"
#
# Arquitectura del sistema:
#   1. CNN MobileNetV2 fine-tuned sobre el dataset grassknoted/asl-alphabet
#      de Kaggle (29 clases: A-Z, del, nothing, space).
#   2. MLP (256→128→64) entrenado sobre landmarks de mano extraídos con
#      MediaPipe Hand Landmarker (21 puntos × 3 coordenadas = 63 features).
#   3. Sistema híbrido: combinación ponderada
#      (78% landmarks + 22% CNN) para la predicción final.
#
# Flujo de predicción en vivo:
#   Frame de cámara → base64 → servidor → MediaPipe detecta mano →
#   extrae landmarks + crop de la mano → MLP predice sobre landmarks →
#   CNN predice sobre crop → combinación ponderada → letra detectada.
# ==============================================================================

from base64 import b64decode, b64encode
from functools import lru_cache
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import random
from urllib.request import urlopen

import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo (sin display)
import matplotlib.pyplot as plt
import numpy as np

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from proyecto.ml.constants import (
    DYNAMIC_CLASS_NAMES,
    DYNAMIC_MODEL_MIN_BYTES,
    DYNAMIC_MODEL_PATH,
)
from proyecto.ml.sequence_inference import predict_sequence


logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES DEL SISTEMA
# ==============================================================================

# --- Dataset ---
DATASET_HANDLE = 'grassknoted/asl-alphabet'  # Identificador en Kaggle
DATASET_DIR = Path(__file__).resolve().parent / 'datasets' / 'asl_alphabet'
TRAIN_DIR = DATASET_DIR / 'asl_alphabet_train' / 'asl_alphabet_train'

# --- Directorio de modelos ---
MODEL_DIR = Path(__file__).resolve().parent / 'models'

# --- Rutas de archivos de modelos ---
IMAGE_MODEL_PATH = MODEL_DIR / 'asl_alphabet_mobilenet_v2.keras'
IMAGE_CLASS_NAMES_PATH = MODEL_DIR / 'asl_alphabet_class_names.json'
LANDMARK_MODEL_PATH = MODEL_DIR / 'asl_landmarks_mlp.keras'
LANDMARK_METADATA_PATH = MODEL_DIR / 'asl_landmarks_metadata.json'
LANDMARK_DATA_CACHE_PATH = MODEL_DIR / 'asl_landmarks_dataset_v1.npz'
HAND_LANDMARKER_MODEL_PATH = MODEL_DIR / 'hand_landmarker.task'

# --- URL de descarga del modelo de detección de manos de MediaPipe ---
HAND_LANDMARKER_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
)

# --- Directorio de caché para MediaPipe/matplotlib en Render ---
# En Render no hay directorio home, por eso se redirige MPLCONFIGDIR.
MEDIAPIPE_CACHE_DIR = Path(__file__).resolve().parent / '.cache' / 'matplotlib'
MEDIAPIPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MEDIAPIPE_CACHE_DIR))

# --- Hiperparámetros de la CNN ---
IMAGE_SIZE = (96, 96)        # Tamaño de entrada de MobileNetV2
BATCH_SIZE = 64               # Tamaño de lote para entrenamiento
BASE_TRAIN_EPOCHS = 3         # Épocas iniciales (cabeza entrenable)
FINE_TUNE_EPOCHS = 1          # Épocas de fine-tuning
VALIDATION_SPLIT = 0.1        # 10% de los datos para validación

# --- Hiperparámetros del sistema híbrido ---
PREDICTION_THRESHOLD = 0.72   # Confianza mínima para aceptar una predicción
HAND_MARGIN_RATIO = 0.30      # Margen extra al recortar la mano
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.45  # Umbral de detección de MediaPipe

# --- Hiperparámetros del MLP de landmarks ---
LANDMARK_FEATURE_SIZE = 63    # 21 landmarks × 3 coordenadas (x, y, z)
LANDMARK_CACHE_VERSION = 1    # Versión del caché del dataset de landmarks
LANDMARK_TRAIN_EPOCHS = 24    # Épocas de entrenamiento del MLP
LANDMARK_BATCH_SIZE = 512     # Tamaño de lote para el MLP

# --- Pesos del sistema híbrido ---
LANDMARK_MODEL_WEIGHT = 0.78  # Peso del clasificador de landmarks
IMAGE_MODEL_WEIGHT = 0.22     # Peso del clasificador de imágenes

# --- Clases especiales ---
LANDMARK_EXCLUDED_CLASSES = {'nothing'}  # 'nothing' no se entrena en landmarks
NO_HAND_LABEL = 'no_hand'     # Etiqueta cuando no se detecta mano

# --- Tamaños mínimos para considerar un modelo válido ---
IMAGE_MODEL_MIN_BYTES = 100000
LANDMARK_MODEL_MIN_BYTES = 50000

# --- Límites del endpoint de cámara ---
MAX_IMAGE_PAYLOAD_BYTES = 2 * 1024 * 1024
MAX_CAMERA_IMAGE_PIXELS = 2_000_000
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 1

# --- Mapeo de acciones para clases especiales ---
SPECIAL_ACTIONS = {
    'del': 'delete',
    'nothing': 'ignore',
    'space': 'space',
}

# --- Etiquetas legibles para mostrar en la interfaz ---
DISPLAY_LABELS = {
    'del': 'BORRAR',
    'no_hand': 'SIN MANO',
    'nothing': 'SIN SENA',
    'space': 'ESPACIO',
}


# ==============================================================================
# VISTAS PRINCIPALES
# ==============================================================================


@ensure_csrf_cookie
def main(request):
    """Página principal. Renderiza el template con el estado actual
    del sistema (dataset, modelos, etc.)."""
    return render(request, 'index.html', build_home_context())


def prediccion(request):
    """Vista de entrenamiento y evaluación.

    Flujo:
      1. Verifica que el entrenamiento esté habilitado.
      2. Descarga el dataset de Kaggle si hace falta.
      3. Descarga el modelo de detección de manos si hace falta.
      4. Carga o entrena el modelo CNN (MobileNetV2).
      5. Carga o entrena el MLP de landmarks.
      6. Evalúa el MLP sobre el conjunto de validación.
      7. Genera ejemplos de predicción para inspección visual.
      8. Genera y guarda las gráficas de resultados (matplotlib).

    Renderiza index.html con los resultados.
    """
    context = build_home_context()
    if request.method != 'POST':
        return render(request, 'index.html', context)

    if not settings.TRAINING_ENABLED:
        context['error_message'] = (
            'En este despliegue el entrenamiento esta desactivado. '
            'La version de Render esta preparada solo para inferencia.'
        )
        return render(request, 'index.html', context)

    try:
        # --- Preparación de datos y modelos ---
        ensure_image_dataset()
        ensure_hand_landmarker_model()
        image_model = get_or_train_image_model()
        landmark_model = get_or_train_landmark_model()
        landmark_dataset = load_landmark_dataset()
        global_class_names = load_class_names()

        # --- Evaluación del modelo de landmarks ---
        eval_result = landmark_model.evaluate(
            landmark_dataset['val_x'],
            landmark_dataset['val_y'],
            verbose=0,
        )
        loss = float(eval_result[0])
        accuracy = float(eval_result[1])

        # --- Generar gráficas de resultados ---
        try:
            graph_paths = generate_evaluation_graphs(
                landmark_model=landmark_model,
                landmark_dataset=landmark_dataset,
                image_model=image_model,
                global_class_names=global_class_names,
            )
            context.update(graph_paths)
        except Exception:
            context['graph_error'] = 'No se pudieron generar las gráficas.'

        # --- Ejemplos de predicción para inspección visual ---
        preview_examples = build_preview_examples(
            landmark_dataset['supported_class_names'],
            sample_size=12,
        )

        image_results = []
        predicted_tokens = []
        actual_tokens = []
        for idx, example in enumerate(preview_examples, start=1):
            predicted_label, confidence, metadata = predict_pil_image(
                example['image'],
                image_model=image_model,
                landmark_model=landmark_model,
                global_class_names=global_class_names,
                supported_class_names=landmark_dataset['supported_class_names'],
            )
            actual_label = example['actual_label']
            image_results.append({
                'id': idx,
                'image': pil_image_to_base64(example['image']),
                'actual': pretty_label(actual_label),
                'predicted': pretty_label(predicted_label),
                'confidence': f'{confidence:.2f}',
                'mode': metadata['prediction_mode'],
            })
            predicted_tokens.append(format_preview_token(predicted_label))
            actual_tokens.append(format_preview_token(actual_label))

        # --- Estadísticas de cobertura de landmarks ---
        train_detected = int(landmark_dataset['train_detected_count'])
        train_total = int(landmark_dataset['train_total_count'])
        val_detected = int(landmark_dataset['val_detected_count'])
        val_total = int(landmark_dataset['val_total_count'])
        train_coverage = train_detected / train_total if train_total else 0.0
        val_coverage = val_detected / val_total if val_total else 0.0

        context.update({
            'results': image_results,
            'accuracy': f'{accuracy:.4f}',
            'loss': f'{loss:.4f}',
            'sample_count': int(landmark_dataset['val_x'].shape[0]),
            'preview_count': len(image_results),
            'train_count': int(landmark_dataset['train_x'].shape[0]),
            'predicted_word': ' '.join(predicted_tokens),
            'actual_word': ' '.join(actual_tokens),
            'status_message': (
                'Modelo hibrido listo. Se uso el clasificador por landmarks de MediaPipe '
                f'y la CNN de apoyo. Cobertura de landmarks: train {train_detected}/{train_total} '
                f'({train_coverage:.1%}) y validacion {val_detected}/{val_total} ({val_coverage:.1%}).'
            ),
        })
    except Exception as exc:
        context['error_message'] = str(exc)

    return render(request, 'index.html', context)

def camera_predict(request):
    """Endpoint API POST para predicción en vivo desde la cámara.

    Recibe una imagen en base64, la procesa con el sistema híbrido
    (MediaPipe + CNN + MLP) y devuelve la letra detectada junto con
    metadatos (confianza, acción, handedness, etc.).

    Args en el body JSON:
        image: string base64 de la imagen (data:image/png;base64,...)

    Returns:
        JsonResponse con la predicción y metadatos.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Se requiere una solicitud POST.'}, status=405)

    if request.content_type != 'application/json':
        return JsonResponse({'error': 'El contenido debe ser JSON.'}, status=415)

    if len(request.body) > MAX_IMAGE_PAYLOAD_BYTES * 2:
        return JsonResponse({'error': 'La imagen excede el tamaño permitido.'}, status=413)

    client_ip = request.META.get('REMOTE_ADDR', 'unknown')
    rate_limit_key = f'camera-predict:{client_ip}'
    if cache.add(rate_limit_key, 1, timeout=RATE_LIMIT_WINDOW_SECONDS):
        request_count = 1
    else:
        request_count = cache.incr(rate_limit_key)
    if request_count > RATE_LIMIT_REQUESTS:
        return JsonResponse(
            {'error': 'Demasiadas solicitudes. Intenta nuevamente en un momento.'},
            status=429,
        )

    try:
        payload = json.loads(request.body.decode('utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('El cuerpo JSON debe ser un objeto.')
        image_data = payload.get('image')
        if not isinstance(image_data, str) or not image_data:
            raise ValueError('No se recibio ninguna imagen para procesar.')

        image_model = get_or_train_image_model()
        landmark_model = get_or_train_landmark_model()
        predicted_label, confidence, metadata = predict_camera_image(
            image_data=image_data,
            image_model=image_model,
            landmark_model=landmark_model,
        )
        display = pretty_label(predicted_label)
        action = resolve_prediction_action(
            predicted_label,
            confidence,
            hand_detected=metadata['hand_detected'],
        )
        token = build_action_token(
            predicted_label,
            confidence,
            hand_detected=metadata['hand_detected'],
        )

        return JsonResponse({
            'predicted': display,
            'raw_label': predicted_label,
            'confidence': f'{confidence:.2f}',
            'action': action,
            'token': token,
            'hand_detected': metadata['hand_detected'],
            'preprocess_mode': metadata['preprocess_mode'],
            'prediction_mode': metadata['prediction_mode'],
            'landmark_label': metadata.get('landmark_label', ''),
            'landmark_confidence': metadata.get('landmark_confidence', 0.0),
            'image_label': metadata.get('image_label', ''),
            'image_confidence': metadata.get('image_confidence', 0.0),
            'handedness': metadata.get('handedness', 'Unknown'),
            'landmarks': metadata.get('landmarks', []),
        })
    except (UnicodeDecodeError, ValueError) as exc:
        return JsonResponse({'error': str(exc)}, status=400)
    except Exception:
        logger.exception('Error inesperado durante la prediccion de camara.')
        return JsonResponse(
            {'error': 'No fue posible procesar la imagen. Intenta nuevamente.'},
            status=500,
        )


def sequence_predict(request):
    """Predice una seña dinamica a partir de landmarks por frame."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Se requiere una solicitud POST.'}, status=405)
    if request.content_type != 'application/json':
        return JsonResponse({'error': 'El contenido debe ser JSON.'}, status=415)

    try:
        payload = json.loads(request.body.decode('utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('El cuerpo JSON debe ser un objeto.')
        frames = payload.get('frames')
        if not isinstance(frames, list) or not frames:
            raise ValueError('Se requiere una lista de frames de landmarks.')
        if not DYNAMIC_MODEL_PATH.exists() or DYNAMIC_MODEL_PATH.stat().st_size < DYNAMIC_MODEL_MIN_BYTES:
            return JsonResponse(
                {'error': 'El modelo de senas dinamicas aun no esta entrenado.'},
                status=503,
            )

        model = get_dynamic_model()
        class_index, confidence = predict_sequence(model, frames)
        if class_index >= len(DYNAMIC_CLASS_NAMES):
            raise ValueError('La salida del modelo no coincide con sus clases.')
        return JsonResponse({
            'predicted': DYNAMIC_CLASS_NAMES[class_index],
            'confidence': f'{confidence:.2f}',
            'class_index': class_index,
        })
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        return JsonResponse({'error': str(exc)}, status=400)
    except Exception:
        logger.exception('Error inesperado durante la prediccion temporal.')
        return JsonResponse(
            {'error': 'No fue posible procesar la secuencia.'},
            status=500,
        )


@lru_cache(maxsize=1)
def get_dynamic_model():
    """Carga una sola vez el modelo temporal guardado en disco."""
    _, tf = import_ml_dependencies()
    try:
        return tf.keras.models.load_model(DYNAMIC_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError('El modelo de senas dinamicas no es valido.') from exc


# ==============================================================================
# FUNCIONES DE CONTEXTO Y ESTADO
# ==============================================================================


def build_home_context():
    """Prepara el diccionario de contexto para el template index.html.
    Indica si el dataset, la CNN y el MLP de landmarks están listos."""
    dataset_ready = dataset_is_ready() if settings.TRAINING_ENABLED else False
    image_model_ready = model_file_looks_ready(IMAGE_MODEL_PATH, IMAGE_MODEL_MIN_BYTES) and IMAGE_CLASS_NAMES_PATH.exists()
    landmark_model_ready = model_file_looks_ready(LANDMARK_MODEL_PATH, LANDMARK_MODEL_MIN_BYTES) and LANDMARK_METADATA_PATH.exists()
    stats = get_dataset_stats() if dataset_ready else None
    return {
        'dataset_ready': dataset_ready,
        'model_ready': image_model_ready and landmark_model_ready,
        'image_model_ready': image_model_ready,
        'landmark_model_ready': landmark_model_ready,
        'dataset_name': DATASET_HANDLE,
        'dataset_classes': stats['classes'] if stats else 0,
        'dataset_total_images': stats['images'] if stats else 0,
        'train_count': stats['train_images'] if stats else 0,
        'sample_count': stats['validation_images'] if stats else 0,
        'prediction_threshold': f'{PREDICTION_THRESHOLD:.2f}',
        'training_enabled': settings.TRAINING_ENABLED,
        'dynamic_model_ready': (
            DYNAMIC_MODEL_PATH.exists()
            and DYNAMIC_MODEL_PATH.stat().st_size >= DYNAMIC_MODEL_MIN_BYTES
        ),
    }


def dataset_is_ready():
    """Verifica si el dataset ASL Alphabet ya está descargado en disco."""
    return TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir())


@lru_cache(maxsize=1)
def get_dataset_stats():
    """Cuenta el número de clases e imágenes del dataset.
    El resultado se cachea con lru_cache para evitar recorrer
    el disco en cada petición."""
    ensure_image_dataset()
    class_dirs = sorted([path for path in TRAIN_DIR.iterdir() if path.is_dir()])
    total_images = 0
    for class_dir in class_dirs:
        total_images += sum(1 for _ in class_dir.glob('*'))

    return {
        'classes': len(class_dirs),
        'images': total_images,
        'train_images': int(total_images * (1 - VALIDATION_SPLIT)),
        'validation_images': int(total_images * VALIDATION_SPLIT),
    }


# ==============================================================================
# IMPORTADORES DE DEPENDENCIAS (diferidos para evitar errores de import)
# ==============================================================================
# Estas funciones importan las librerías pesadas (TensorFlow, MediaPipe, PIL)
# solo cuando se necesitan, no al arrancar Django. Así la página principal
# carga rápido aunque falten dependencias ML.


def import_ml_dependencies():
    """Importa numpy y tensorflow. Lanza RuntimeError si faltan."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            'Faltan dependencias de machine learning. Activa .venv312 e instala requirements.txt.'
        ) from exc

    return np, tf


def import_image_dependencies():
    """Importa Pillow (PIL). Lanza RuntimeError si falta."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            'Falta Pillow. Instala las dependencias del proyecto antes de usar la camara.'
        ) from exc

    return Image, ImageOps


def import_mediapipe_dependencies():
    """Importa MediaPipe. Lanza RuntimeError si falta."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            'Falta MediaPipe. Activa .venv312 e instala requirements.txt para usar landmarks.'
        ) from exc

    return mp


# ==============================================================================
# DESCARGA DE MODELOS Y DATASET
# ==============================================================================


def ensure_hand_landmarker_model():
    """Descarga el modelo HandLandmarker de MediaPipe si no existe en disco.
    El archivo se guarda en models/hand_landmarker.task."""
    if HAND_LANDMARKER_MODEL_PATH.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(HAND_LANDMARKER_MODEL_URL) as response:
            HAND_LANDMARKER_MODEL_PATH.write_bytes(response.read())
    except Exception as exc:
        if HAND_LANDMARKER_MODEL_PATH.exists():
            HAND_LANDMARKER_MODEL_PATH.unlink()
        raise RuntimeError(
            'No fue posible descargar el modelo de deteccion de mano de MediaPipe.'
        ) from exc


def ensure_image_dataset():
    """Descarga el dataset ASL Alphabet de Kaggle (grassknoted/asl-alphabet)
    si no existe en disco. Usa kagglehub como cliente."""
    if dataset_is_ready():
        return

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            'Falta kagglehub para descargar el dataset grande de Kaggle.'
        ) from exc

    try:
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(DATASET_DIR),
            force_download=False,
        )
    except Exception as exc:
        raise RuntimeError(
            'No fue posible descargar el dataset de Kaggle. Verifica tu conexion o vuelve a intentar.'
        ) from exc

    if not dataset_is_ready():
        raise RuntimeError(
            'La descarga termino, pero no se encontro la carpeta esperada del dataset ASL.'
        )


# ==============================================================================
# CONSTRUCCIÓN DE DATASETS DE ENTRENAMIENTO (CNN)
# ==============================================================================


def build_training_datasets(tf):
    """Crea los datasets de entrenamiento y validación para la CNN
    usando image_dataset_from_directory. Las imágenes se cargan desde
    las carpetas del dataset ASL Alphabet."""
    ensure_image_dataset()
    common_kwargs = {
        'directory': str(TRAIN_DIR),
        'seed': 123,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'label_mode': 'categorical',
        'validation_split': VALIDATION_SPLIT,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset='training',
        shuffle=True,
        **common_kwargs,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset='validation',
        shuffle=False,
        **common_kwargs,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def model_file_looks_ready(model_path, min_bytes):
    """Verifica que un archivo de modelo exista y tenga al menos min_bytes.
    Esto evita cargar modelos corruptos o incompletos."""
    return model_path.exists() and model_path.stat().st_size >= min_bytes


def load_saved_keras_model(tf, model_path, min_bytes):
    """Intenta cargar un modelo .keras desde disco.
    Si el archivo no existe, está corrupto o es muy pequeño,
    retorna None. Si está corrupto, elimina el archivo."""
    if not model_file_looks_ready(model_path, min_bytes):
        if model_path.exists():
            try:
                model_path.unlink()
            except OSError:
                pass
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        try:
            model_path.unlink()
        except OSError:
            pass
        return None


# ==============================================================================
# MODELO CNN (MobileNetV2 con Transfer Learning)
# ==============================================================================


@lru_cache(maxsize=1)
def get_or_train_image_model():
    """Carga el modelo CNN guardado en disco, o lo entrena desde cero
    si no existe y el entrenamiento está habilitado.

    Arquitectura:
      - MobileNetV2 pre-entrenado en ImageNet (alpha=0.35) como extractor
      - Capas de aumento de datos (rotación, zoom, contraste)
      - GlobalAveragePooling2D + Dropout + Dense softmax

    Entrenamiento en dos fases:
      1. Cabeza entrenable, extractor congelado (3 épocas)
      2. Fine-tuning: últimas 20 capas del extractor descongeladas (1 época)

    El resultado se cachea con lru_cache para evitar recargar el modelo
    en cada petición HTTP."""
    _, tf = import_ml_dependencies()
    class_names = load_class_names()

    loaded_model = load_saved_keras_model(tf, IMAGE_MODEL_PATH, IMAGE_MODEL_MIN_BYTES)
    if loaded_model is not None:
        return loaded_model

    if not settings.TRAINING_ENABLED:
        raise RuntimeError(
            'El modelo de imagen no esta disponible en producción y el entrenamiento esta desactivado.'
        )

    ensure_image_dataset()

    train_ds, val_ds = build_training_datasets(tf)
    model = build_transfer_model(tf, len(class_names))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=IMAGE_MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
        ),
    ]

    # Fase 1: entrenar la cabeza (extractor congelado)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=BASE_TRAIN_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Fase 2: fine-tuning (descongelar últimas 20 capas del extractor)
    feature_extractor = next(
        (layer for layer in model.layers if layer.name.startswith('mobilenetv2')),
        None,
    )
    if feature_extractor is not None:
        feature_extractor.trainable = True
        for layer in feature_extractor.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=BASE_TRAIN_EPOCHS,
            epochs=BASE_TRAIN_EPOCHS + FINE_TUNE_EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

    save_class_names(class_names)
    return tf.keras.models.load_model(IMAGE_MODEL_PATH)


def build_transfer_model(tf, num_classes):
    """Construye la arquitectura de la CNN con transfer learning.

    Arquitectura:
      - Input → Augmentation (RandomRotation, RandomZoom, RandomContrast)
      → preprocess_input de MobileNetV2 → MobileNetV2 (congelado, alpha=0.35)
      → GlobalAveragePooling2D → Dropout(0.2) → Dense(num_classes, softmax)

    Args:
        tf: módulo tensorflow
        num_classes: número de clases a clasificar (29)

    Returns:
        Modelo de Keras compilado (Adam lr=1e-3, categorical_crossentropy)
    """
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name='augmentation',
    )

    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35,
    )
    feature_extractor.trainable = False
    feature_extractor._name = 'feature_extractor'

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = feature_extractor(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ==============================================================================
# MODELO DE LANDMARKS (MLP)
# ==============================================================================


@lru_cache(maxsize=1)
def get_or_train_landmark_model():
    """Carga el MLP de landmarks guardado en disco, o lo entrena desde cero
    si no existe y el entrenamiento está habilitado.

    El dataset de landmarks se construye extrayendo landmarks con MediaPipe
    de todas las imágenes del dataset ASL, y se cachea en un archivo .npz.
    """
    _, tf = import_ml_dependencies()

    loaded_model = load_saved_keras_model(tf, LANDMARK_MODEL_PATH, LANDMARK_MODEL_MIN_BYTES)
    if loaded_model is not None:
        return loaded_model

    if not settings.TRAINING_ENABLED:
        raise RuntimeError(
            'El modelo de landmarks no esta disponible en producción y el entrenamiento esta desactivado.'
        )

    landmark_dataset = load_landmark_dataset()

    model = build_landmark_model(tf, len(landmark_dataset['supported_class_names']))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_landmark_metadata(landmark_dataset['supported_class_names'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=LANDMARK_MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
        ),
    ]

    model.fit(
        landmark_dataset['train_x'],
        landmark_dataset['train_y'],
        validation_data=(landmark_dataset['val_x'], landmark_dataset['val_y']),
        epochs=LANDMARK_TRAIN_EPOCHS,
        batch_size=LANDMARK_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return tf.keras.models.load_model(LANDMARK_MODEL_PATH)


def build_landmark_model(tf, num_classes):
    """Construye el MLP para clasificación basada en landmarks.

    Arquitectura:
      Input(63) → BatchNorm → Dense(256, ReLU) → Dropout(0.30)
      → Dense(128, ReLU) → Dropout(0.25)
      → Dense(64, ReLU) → Dropout(0.20)
      → Dense(num_classes, softmax)

    Args:
        tf: módulo tensorflow
        num_classes: número de clases soportadas (excluyendo 'nothing')

    Returns:
        Modelo de Keras compilado (Adam lr=1e-3, sparse_categorical_crossentropy)
    """
    inputs = tf.keras.Input(shape=(LANDMARK_FEATURE_SIZE,))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def load_landmark_dataset():
    """Carga el dataset de landmarks desde el caché en disco (.npz).
    Si el caché no existe o la versión cambió, reconstruye el dataset
    extrayendo landmarks de todas las imágenes con MediaPipe."""
    np, _ = import_ml_dependencies()
    current_class_names = load_class_names()
    current_supported = get_supported_landmark_class_names(current_class_names)

    if LANDMARK_DATA_CACHE_PATH.exists():
        with np.load(LANDMARK_DATA_CACHE_PATH, allow_pickle=False) as cached:
            cache_version = int(cached['cache_version'][0])
            cached_supported = list(cached['supported_class_names'].tolist())
            cached_global = list(cached['global_class_names'].tolist())
            if (
                cache_version == LANDMARK_CACHE_VERSION
                and cached_supported == current_supported
                and cached_global == current_class_names
            ):
                return {
                    'train_x': cached['train_x'].astype('float32'),
                    'train_y': cached['train_y'].astype('int32'),
                    'val_x': cached['val_x'].astype('float32'),
                    'val_y': cached['val_y'].astype('int32'),
                    'val_paths': cached['val_paths'].tolist(),
                    'supported_class_names': cached_supported,
                    'global_class_names': cached_global,
                    'train_detected_count': int(cached['train_detected_count'][0]),
                    'train_total_count': int(cached['train_total_count'][0]),
                    'val_detected_count': int(cached['val_detected_count'][0]),
                    'val_total_count': int(cached['val_total_count'][0]),
                }

    return build_landmark_dataset(current_class_names, current_supported)


def build_landmark_dataset(global_class_names, supported_class_names):
    """Construye el dataset de landmarks desde cero.

    Para cada imagen de entrenamiento y validación en el dataset ASL:
      1. Abre la imagen con PIL.
      2. Ejecuta MediaPipe HandLandmarker para detectar la mano.
      3. Si detecta mano, extrae el vector de landmarks (63 features).
      4. Acumula el vector y la clase correspondiente.

    El resultado se guarda en un archivo .npz comprimido para
    acelerar futuras cargas.

    Args:
        global_class_names: todas las clases del dataset (29)
        supported_class_names: clases soportadas por landmarks (excluye 'nothing')

    Returns:
        Diccionario con train_x, train_y, val_x, val_y, val_paths,
        supported_class_names, global_class_names, y contadores de cobertura.
    """
    np, _ = import_ml_dependencies()
    Image, _ = import_image_dependencies()
    ensure_image_dataset()
    ensure_hand_landmarker_model()

    rng = random.Random(123)
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    val_paths = []
    train_detected_count = 0
    train_total_count = 0
    val_detected_count = 0
    val_total_count = 0

    for class_index, class_name in enumerate(supported_class_names):
        class_dir = TRAIN_DIR / class_name
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        )
        rng.shuffle(image_paths)
        split_index = int(len(image_paths) * (1 - VALIDATION_SPLIT))
        split_index = max(1, min(split_index, len(image_paths) - 1))
        train_paths = image_paths[:split_index]
        eval_paths = image_paths[split_index:]

        for image_path in train_paths:
            train_total_count += 1
            image = Image.open(image_path).convert('RGB')
            analysis = analyze_hand_image(image)
            image.close()
            if not analysis['hand_detected']:
                continue

            train_detected_count += 1
            train_x.append(analysis['feature_vector'])
            train_y.append(class_index)

        for image_path in eval_paths:
            val_total_count += 1
            image = Image.open(image_path).convert('RGB')
            analysis = analyze_hand_image(image)
            image.close()
            if not analysis['hand_detected']:
                continue

            val_detected_count += 1
            val_x.append(analysis['feature_vector'])
            val_y.append(class_index)
            val_paths.append(str(image_path))

    if not train_x or not val_x:
        raise RuntimeError(
            'No fue posible construir el dataset de landmarks. MediaPipe no detecto suficientes manos.'
        )

    train_x = np.asarray(train_x, dtype='float32')
    train_y = np.asarray(train_y, dtype='int32')
    val_x = np.asarray(val_x, dtype='float32')
    val_y = np.asarray(val_y, dtype='int32')

    # Guardar caché comprimido para acelerar futuras cargas
    np.savez_compressed(
        LANDMARK_DATA_CACHE_PATH,
        cache_version=np.asarray([LANDMARK_CACHE_VERSION], dtype='int32'),
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        val_paths=np.asarray(val_paths),
        supported_class_names=np.asarray(supported_class_names),
        global_class_names=np.asarray(global_class_names),
        train_detected_count=np.asarray([train_detected_count], dtype='int32'),
        train_total_count=np.asarray([train_total_count], dtype='int32'),
        val_detected_count=np.asarray([val_detected_count], dtype='int32'),
        val_total_count=np.asarray([val_total_count], dtype='int32'),
    )
    save_landmark_metadata(supported_class_names)
    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'val_paths': val_paths,
        'supported_class_names': list(supported_class_names),
        'global_class_names': list(global_class_names),
        'train_detected_count': train_detected_count,
        'train_total_count': train_total_count,
        'val_detected_count': val_detected_count,
        'val_total_count': val_total_count,
    }


def get_supported_landmark_class_names(global_class_names=None):
    """Retorna los nombres de clase soportados por el modelo de landmarks,
    excluyendo aquellas en LANDMARK_EXCLUDED_CLASSES (ej. 'nothing')."""
    if global_class_names is None:
        global_class_names = load_class_names()
    return [class_name for class_name in global_class_names if class_name not in LANDMARK_EXCLUDED_CLASSES]


def load_landmark_metadata():
    """Carga los metadatos del modelo de landmarks desde JSON.
    Si no existen, los genera a partir de las clases del dataset."""
    if LANDMARK_METADATA_PATH.exists():
        return json.loads(LANDMARK_METADATA_PATH.read_text(encoding='utf-8'))

    supported_class_names = get_supported_landmark_class_names()
    metadata = {
        'supported_class_names': supported_class_names,
        'cache_version': LANDMARK_CACHE_VERSION,
    }
    save_landmark_metadata(supported_class_names)
    return metadata


def save_landmark_metadata(supported_class_names):
    """Guarda los metadatos del modelo de landmarks en un archivo JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARK_METADATA_PATH.write_text(
        json.dumps(
            {
                'supported_class_names': list(supported_class_names),
                'cache_version': LANDMARK_CACHE_VERSION,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding='utf-8',
    )


# ==============================================================================
# DETECCIÓN DE MANOS Y EXTRACCIÓN DE LANDMARKS (MediaPipe)
# ==============================================================================


@lru_cache(maxsize=1)
def get_hand_detector():
    """Singleton del detector HandLandmarker de MediaPipe.
    Se cachea con lru_cache para crear la instancia una sola vez."""
    mp = import_mediapipe_dependencies()
    ensure_hand_landmarker_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def predict_camera_image(image_data, image_model, landmark_model):
    """Convierte una imagen en base64 a PIL Image y ejecuta la
    predicción híbrida. Usada por el endpoint camera_predict."""
    Image, _ = import_image_dependencies()
    encoded = image_data.split(',', 1)[1] if ',' in image_data else image_data
    max_encoded_size = (MAX_IMAGE_PAYLOAD_BYTES * 4 // 3) + 4
    if len(encoded) > max_encoded_size:
        raise ValueError('La imagen excede el tamaño permitido.')
    try:
        raw_image = b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError('La imagen enviada no esta codificada en base64 valido.') from exc
    if not raw_image or len(raw_image) > MAX_IMAGE_PAYLOAD_BYTES:
        raise ValueError('La imagen excede el tamaño permitido.')

    try:
        opened_image = Image.open(BytesIO(raw_image))
        if opened_image.width * opened_image.height > MAX_CAMERA_IMAGE_PIXELS:
            opened_image.close()
            raise ValueError('La resolución de la imagen excede el límite permitido.')
        image = opened_image.convert('RGB')
        opened_image.close()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('La imagen enviada no tiene un formato válido.') from exc
    try:
        return predict_pil_image(image, image_model=image_model, landmark_model=landmark_model)
    finally:
        image.close()


def predict_pil_image(
    image,
    image_model=None,
    landmark_model=None,
    global_class_names=None,
    supported_class_names=None,
):
    """Predicción híbrida completa sobre una imagen PIL.

    Flujo:
      1. Analiza la imagen con MediaPipe para detectar mano.
      2. Si no hay mano, retorna 'no_hand'.
      3. Si hay mano, extrae landmarks y crop de la mano.
      4. MLP predice sobre landmarks → probs expandidas a 29 clases.
      5. CNN predice sobre crop (elige entre original y mirrored).
      6. Combina: 78% landmarks + 22% CNN.
      7. Zerocea la clase 'nothing' en la combinación.
      8. Normaliza y elige la clase con mayor probabilidad.

    Returns:
        (predicted_label, confidence, metadata_dict)
    """
    np, _ = import_ml_dependencies()
    global_class_names = global_class_names or load_class_names()
    supported_class_names = supported_class_names or load_landmark_metadata()['supported_class_names']

    analysis = analyze_hand_image(image)
    if not analysis['hand_detected']:
        return NO_HAND_LABEL, 0.0, {
            'hand_detected': False,
            'preprocess_mode': 'no-hand-detected',
            'prediction_mode': 'no-hand-detected',
            'image_variant': 'none',
            'landmarks': [],
        }

    if landmark_model is None:
        landmark_model = get_or_train_landmark_model()
    if image_model is None:
        image_model = get_or_train_image_model()

    # --- Predicción con landmarks (MLP) ---
    landmark_prediction = landmark_model.predict(
        analysis['feature_vector'].reshape(1, -1),
        verbose=0,
    )[0]
    landmark_probs = expand_landmark_probabilities(
        landmark_prediction,
        supported_class_names,
        global_class_names,
    )

    # --- Predicción con imagen (CNN) ---
    image_probs, image_variant = build_image_probability_vector(
        image_model,
        analysis['crop_image'],
    )

    # --- Combinación híbrida ponderada ---
    nothing_index = (
        global_class_names.index('nothing')
        if 'nothing' in global_class_names
        else None
    )
    landmark_probs = normalize_probability_vector(landmark_probs)
    image_probs = normalize_probability_vector(image_probs)
    combined_probs = (
        LANDMARK_MODEL_WEIGHT * landmark_probs
        + IMAGE_MODEL_WEIGHT * image_probs
    )
    if nothing_index is not None:
        combined_probs[nothing_index] = 0.0  # 'nothing' no se predice en híbrido
    combined_probs = normalize_probability_vector(combined_probs)

    predicted_index = int(np.argmax(combined_probs))
    landmark_index = int(np.argmax(landmark_probs))
    image_index = int(np.argmax(image_probs))
    predicted_label = global_class_names[predicted_index]

    return predicted_label, float(combined_probs[predicted_index]), {
        'hand_detected': True,
        'preprocess_mode': 'mediapipe-landmarks',
        'prediction_mode': 'hybrid-landmarks-image',
        'image_variant': image_variant,
        'landmark_label': global_class_names[landmark_index],
        'landmark_confidence': float(landmark_probs[landmark_index]),
        'image_label': global_class_names[image_index],
        'image_confidence': float(image_probs[image_index]),
        'handedness': analysis['handedness'],
        'landmarks': analysis['landmarks'],
    }


def analyze_hand_image(image):
    """Analiza una imagen PIL con MediaPipe HandLandmarker.

    Extrae:
      - landmarks de la mano (21 puntos × 3 coordenadas)
      - crop cuadrado alrededor de la mano con margen
      - handedness (izquierda/derecha)

    Si hay múltiples manos, elige la de mayor área.

    Returns:
        Diccionario con hand_detected, crop_image, feature_vector, handedness.
    """
    np, _ = import_ml_dependencies()
    mp = import_mediapipe_dependencies()
    detector = get_hand_detector()

    image_array = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    results = detector.detect(mp_image)
    if not results.hand_landmarks:
        return {
            'hand_detected': False,
            'crop_image': None,
            'feature_vector': None,
            'handedness': 'Unknown',
            'landmarks': [],
        }

    best_index = select_best_hand_index(results.hand_landmarks)
    hand_landmarks = results.hand_landmarks[best_index]
    handedness = 'Unknown'
    if results.handedness and len(results.handedness) > best_index and results.handedness[best_index]:
        handedness = results.handedness[best_index][0].category_name

    crop_image = build_hand_crop_from_landmarks(image, hand_landmarks)
    feature_vector = compute_landmark_feature_vector(hand_landmarks)
    if crop_image is None or feature_vector is None:
        return {
            'hand_detected': False,
            'crop_image': None,
            'feature_vector': None,
            'handedness': handedness,
        }

    return {
        'hand_detected': True,
        'crop_image': crop_image,
        'feature_vector': feature_vector,
        'handedness': handedness,
        'landmarks': [
            [float(landmark.x), float(landmark.y), float(landmark.z)]
            for landmark in hand_landmarks
        ],
    }


def select_best_hand_index(hand_landmarks_list):
    """Selecciona el índice de la mano más grande (por área del bounding box)
    cuando MediaPipe detecta múltiples manos."""
    best_index = 0
    best_area = -1.0
    for index, hand_landmarks in enumerate(hand_landmarks_list):
        xs = [landmark.x for landmark in hand_landmarks]
        ys = [landmark.y for landmark in hand_landmarks]
        area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_index = index
    return best_index


def build_hand_crop_from_landmarks(image, hand_landmarks):
    """Recorta un cuadrado alrededor de la mano usando los landmarks.

    Calcula el bounding box de los landmarks, agrega un margen
    (HAND_MARGIN_RATIO) y recorta un cuadrado centrado en la mano.

    Returns:
        PIL.Image cropeada, o None si el recorte es inválido.
    """
    width, height = image.size
    xs = [landmark.x for landmark in hand_landmarks]
    ys = [landmark.y for landmark in hand_landmarks]
    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))

    box_width = max(1.0, (max_x - min_x) * width)
    box_height = max(1.0, (max_y - min_y) * height)
    center_x = ((min_x + max_x) / 2.0) * width
    center_y = ((min_y + max_y) / 2.0) * height
    side = int(max(box_width, box_height) * (1.0 + HAND_MARGIN_RATIO * 2.0))
    side = max(1, min(side, width, height))
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def compute_landmark_feature_vector(hand_landmarks):
    """Normaliza los 21 landmarks de MediaPipe a un vector de 63 features.

    Pasos de normalización:
      1. Resta la muñeca (landmark[0]) como origen.
      2. Rota para que el dedo medio MCP (landmark[9]) apunte hacia arriba.
      3. Refleja horizontalmente si es necesario (índice vs meñique).
      4. Escala a norma unitaria.

    Esto hace que el vector sea invariante a posición, rotación y escala
    de la mano, mejorando la generalización del MLP.

    Returns:
        ndarray de 63 floats, o None si la normalización falla.
    """
    np, _ = import_ml_dependencies()
    coords = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype='float32',
    )
    if coords.shape != (21, 3):
        return None

    # Centrar en la muñeca (landmark 0)
    coords = coords - coords[0]

    # Rotar para que el dedo medio MCP (landmark 9) apunte hacia arriba
    reference_xy = coords[9, :2]
    reference_norm = float(np.linalg.norm(reference_xy))
    if reference_norm < 1e-6:
        return None

    angle = math.atan2(float(reference_xy[1]), float(reference_xy[0]))
    rotation = (-math.pi / 2.0) - angle
    cos_value = math.cos(rotation)
    sin_value = math.sin(rotation)
    rotated_x = coords[:, 0] * cos_value - coords[:, 1] * sin_value
    rotated_y = coords[:, 0] * sin_value + coords[:, 1] * cos_value
    coords[:, 0] = rotated_x
    coords[:, 1] = rotated_y

    # Reflejar si el índice (landmark 5) está a la derecha del meñique (landmark 17)
    if coords[5, 0] > coords[17, 0]:
        coords[:, 0] *= -1.0

    # Escalar a norma unitaria
    scale = float(np.max(np.linalg.norm(coords, axis=1)))
    if scale < 1e-6:
        return None

    coords /= scale
    return coords.reshape(-1).astype('float32')


# ==============================================================================
# FUNCIONES AUXILIARES DE PROBABILIDAD
# ==============================================================================


def expand_landmark_probabilities(prediction, supported_class_names, global_class_names):
    """Expande el vector de probabilidades del MLP de landmarks
    (que solo predice sobre las clases soportadas) al espacio completo
    de 29 clases. Las clases no soportadas quedan en 0."""
    np, _ = import_ml_dependencies()
    expanded = np.zeros(len(global_class_names), dtype='float32')
    for index, class_name in enumerate(supported_class_names):
        expanded[global_class_names.index(class_name)] = float(prediction[index])
    return expanded


def build_image_probability_vector(image_model, crop_image):
    """Ejecuta la CNN sobre el crop de la mano y elige la mejor
    variante entre original y espejada (mirrored).

    Esto ayuda a manejar manos izquierdas/derechas que la CNN
    podría no haber visto durante el entrenamiento.

    Returns:
        (probability_vector, variant_name)
    """
    np, _ = import_ml_dependencies()
    Image, ImageOps = import_image_dependencies()

    normalized = ImageOps.fit(crop_image, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
    mirrored = ImageOps.mirror(normalized)
    candidates = [
        ('original', normalized),
        ('mirrored', mirrored),
    ]
    batch = np.stack(
        [np.asarray(candidate_image).astype('float32') for _, candidate_image in candidates],
        axis=0,
    )
    predictions = image_model.predict(batch, verbose=0)
    best_index = max(
        range(len(candidates)),
        key=lambda candidate_index: float(np.max(predictions[candidate_index])),
    )
    return predictions[best_index].astype('float32'), candidates[best_index][0]


def normalize_probability_vector(probabilities):
    """Normaliza un vector de probabilidades para que sume 1.
    Si la suma es 0, retorna un vector de ceros."""
    np, _ = import_ml_dependencies()
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return np.zeros_like(probabilities)
    return probabilities / total


# ==============================================================================
# GENERACIÓN DE GRÁFICAS DE RESULTADOS
# ==============================================================================

# Directorio donde se guardarán las gráficas generadas
GRAPHS_DIR = Path(__file__).resolve().parent / 'static' / 'images'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def generate_evaluation_graphs(landmark_model, landmark_dataset, image_model, global_class_names):
    """Genera 3 gráficas de resultados del modelo híbrido usando
    matplotlib y las guarda como PNG en static/images/.

    Gráficas generadas:
      1. Matriz de confusión del MLP de landmarks sobre validación.
      2. Precisión por clase (accuracy por letra).
      3. Distribución de confianza en predicciones correctas vs incorrectas.

    Args:
        landmark_model: modelo MLP de landmarks entrenado.
        landmark_dataset: diccionario con val_x, val_y, supported_class_names.
        image_model: modelo CNN MobileNetV2 (no se usa directamente en las
                     gráficas, pero se mantiene por compatibilidad).
        global_class_names: lista de nombres de todas las clases (29).

    Returns:
        Diccionario con rutas relativas a las 3 imágenes para el template.
    """
    # --- Generar predicciones sobre validación ---
    val_preds = landmark_model.predict(landmark_dataset['val_x'], verbose=0)
    val_pred_labels = np.argmax(val_preds, axis=1)

    supported_names = landmark_dataset['supported_class_names']
    val_true = landmark_dataset['val_y']

    # Crear subconjunto de global_class_names solo con clases soportadas
    supported_indices = [global_class_names.index(name) for name in supported_names]

    # 1. MATRIZ DE CONFUSIÓN
    _plot_confusion_matrix(
        val_true, val_pred_labels, supported_names,
        filename=str(GRAPHS_DIR / 'confusion_matrix.png'),
    )

    # 2. PRECISIÓN POR CLASE
    _plot_per_class_accuracy(
        val_true, val_pred_labels, supported_names,
        filename=str(GRAPHS_DIR / 'per_class_accuracy.png'),
    )

    # 3. DISTRIBUCIÓN DE CONFIANZA
    confidences = np.max(val_preds, axis=1)
    correct = val_pred_labels == val_true
    _plot_confidence_distribution(
        confidences, correct,
        filename=str(GRAPHS_DIR / 'confidence_distribution.png'),
    )

    return {
        'graph_confusion_matrix': 'images/confusion_matrix.png',
        'graph_per_class_accuracy': 'images/per_class_accuracy.png',
        'graph_confidence_distribution': 'images/confidence_distribution.png',
    }


def _plot_confusion_matrix(y_true, y_pred, class_names, filename):
    """Genera y guarda una matriz de confusión como heatmap."""
    from sklearn.metrics import confusion_matrix
    plt.style.use('dark_background')
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float32') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    # Etiquetas
    short_names = [name[:4] for name in class_names]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(short_names, fontsize=7)

    # Valores en celdas seleccionadas (solo diagonal y errores > 0.05)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            if val > 0.05 or i == j:
                text_color = 'white' if val > 0.5 else '#cccccc'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=text_color)

    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title('Matriz de Confusión - MLP Landmarks (normalizada)', fontsize=14, pad=16)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)


def _plot_per_class_accuracy(y_true, y_pred, class_names, filename):
    """Genera y guarda un gráfico de barras con precisión por clase."""
    from sklearn.metrics import confusion_matrix
    plt.style.use('dark_background')
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)

    # Ordenar por accuracy descendente
    indices = np.argsort(per_class_acc)[::-1]
    sorted_names = [class_names[i] for i in indices]
    sorted_acc = per_class_acc[indices]

    colors = ['#22c55e' if v >= 0.8 else '#eab308' if v >= 0.5 else '#ef4444' for v in sorted_acc]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(sorted_names)), sorted_acc, color=colors, edgecolor='none')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Precisión', fontsize=12)
    ax.set_title('Precisión por Clase', fontsize=14, pad=16)

    # Valores al final de cada barra
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        ax.text(min(acc + 0.02, 0.95), i, f'{acc:.2f}', va='center', fontsize=8, color='#cccccc')

    # Línea de umbral
    ax.axvline(x=0.8, color='#38bdf8', linestyle='--', linewidth=0.8, alpha=0.5, label='Umbral 0.8')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)


def _plot_confidence_distribution(confidences, correct, filename):
    """Genera y guarda un histograma de distribución de confianza."""
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 21)

    correct_conf = confidences[correct]
    incorrect_conf = confidences[~correct]

    ax.hist(correct_conf, bins=bins, alpha=0.7, label='Correctas',
            color='#22c55e', edgecolor='none')
    ax.hist(incorrect_conf, bins=bins, alpha=0.7, label='Incorrectas',
            color='#ef4444', edgecolor='none')

    ax.set_xlabel('Confianza de la predicción', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución de Confianza: Correctas vs Incorrectas', fontsize=14, pad=16)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)


# ==============================================================================
# VISUALIZACIÓN Y PREVIEW
# ==============================================================================


def build_preview_examples(class_names, sample_size=12):
    """Selecciona imágenes aleatorias de distintas clases del dataset
    para mostrarlas como ejemplos de predicción en la interfaz."""
    Image, _ = import_image_dependencies()
    rng = random.Random(123)
    selected_classes = class_names[:]
    rng.shuffle(selected_classes)
    selected_classes = selected_classes[:min(sample_size, len(selected_classes))]

    examples = []
    for class_name in selected_classes:
        class_dir = TRAIN_DIR / class_name
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        )
        if not image_paths:
            continue

        image_path = rng.choice(image_paths)
        image = Image.open(image_path).convert('RGB')
        examples.append({
            'path': str(image_path),
            'actual_label': class_name,
            'image': image.copy(),
        })
        image.close()

    return examples


def pil_image_to_base64(image):
    """Convierte una imagen PIL a string base64 (data URI)."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    encoded = b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


# ==============================================================================
# PERSISTENCIA DE NOMBRES DE CLASE
# ==============================================================================


def load_class_names():
    """Carga los nombres de clase desde el archivo JSON.
    Si no existe y el entrenamiento está habilitado, los genera
    a partir de las carpetas del dataset."""
    if IMAGE_CLASS_NAMES_PATH.exists():
        return json.loads(IMAGE_CLASS_NAMES_PATH.read_text(encoding='utf-8'))

    if not settings.TRAINING_ENABLED:
        raise RuntimeError(
            'No se encontro asl_alphabet_class_names.json y el entrenamiento esta desactivado.'
        )

    ensure_image_dataset()
    class_names = sorted([path.name for path in TRAIN_DIR.iterdir() if path.is_dir()])
    save_class_names(class_names)
    return class_names


def save_class_names(class_names):
    """Guarda los nombres de clase en un archivo JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CLASS_NAMES_PATH.write_text(
        json.dumps(list(class_names), ensure_ascii=True, indent=2),
        encoding='utf-8',
    )


# ==============================================================================
# ACCIONES Y TOKENS PARA LA CONSTRUCCIÓN DE PALABRAS
# ==============================================================================


def resolve_prediction_action(label, confidence, hand_detected=True):
    """Determina la acción a tomar según la letra detectada y su confianza.

    Acciones posibles:
      - 'ignore': se ignora la predicción (sin mano, muy baja confianza, nothing)
      - 'append': agrega la letra a la palabra actual
      - 'space': agrega un espacio
      - 'delete': borra el último carácter
    """
    if not hand_detected or label == NO_HAND_LABEL:
        return 'ignore'
    if confidence < PREDICTION_THRESHOLD:
        return 'ignore'
    return SPECIAL_ACTIONS.get(label, 'append')


def build_action_token(label, confidence, hand_detected=True):
    """Construye el token (caracter) que se agregará a la palabra
    según la letra detectada."""
    if not hand_detected or label == NO_HAND_LABEL:
        return ''
    if confidence < PREDICTION_THRESHOLD:
        return ''
    if label == 'space':
        return ' '
    if label in {'del', 'nothing'}:
        return ''
    return label


def pretty_label(label):
    """Convierte una etiqueta interna a su forma legible para mostrar
    en la interfaz (ej. 'no_hand' → 'SIN MANO')."""
    return DISPLAY_LABELS.get(label, label)


def format_preview_token(label):
    """Formatea una etiqueta para mostrarla en la secuencia de preview
    (ej. 'space' → '<espacio>')."""
    if label == 'space':
        return '<espacio>'
    if label == 'del':
        return '<borrar>'
    if label == NO_HAND_LABEL:
        return '<sin-mano>'
    if label == 'nothing':
        return '<sin-sena>'
    return label
