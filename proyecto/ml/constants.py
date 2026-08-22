"""Constantes e hiperparámetros del sistema híbrido de reconocimiento ASL."""

import os
from pathlib import Path

BASE_ML_DIR = Path(__file__).resolve().parent.parent

# --- Dataset ---
DATASET_HANDLE = 'grassknoted/asl-alphabet'
DATASET_DIR = BASE_ML_DIR / 'datasets' / 'asl_alphabet'
TRAIN_DIR = DATASET_DIR / 'asl_alphabet_train' / 'asl_alphabet_train'

# --- Directorio de modelos ---
MODEL_DIR = BASE_ML_DIR / 'models'

# --- Rutas de archivos de modelos ---
IMAGE_MODEL_PATH = MODEL_DIR / 'asl_alphabet_mobilenet_v2.keras'
IMAGE_CLASS_NAMES_PATH = MODEL_DIR / 'asl_alphabet_class_names.json'
LANDMARK_MODEL_PATH = MODEL_DIR / 'asl_landmarks_mlp.keras'
LANDMARK_METADATA_PATH = MODEL_DIR / 'asl_landmarks_metadata.json'
LANDMARK_DATA_CACHE_PATH = MODEL_DIR / 'asl_landmarks_dataset_v1.npz'
DYNAMIC_MODEL_PATH = MODEL_DIR / 'asl_dynamic_gru.keras'
DYNAMIC_CLASS_NAMES_PATH = MODEL_DIR / 'asl_dynamic_class_names.json'
HAND_LANDMARKER_MODEL_PATH = MODEL_DIR / 'hand_landmarker.task'

HAND_LANDMARKER_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
)

MEDIAPIPE_CACHE_DIR = BASE_ML_DIR / '.cache' / 'matplotlib'
MEDIAPIPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MEDIAPIPE_CACHE_DIR))

# --- Hiperparámetros CNN ---
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
BASE_TRAIN_EPOCHS = 3
FINE_TUNE_EPOCHS = 1
VALIDATION_SPLIT = 0.1

# --- Hiperparámetros sistema híbrido ---
PREDICTION_THRESHOLD = 0.72
HAND_MARGIN_RATIO = 0.30
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.45

# --- Hiperparámetros MLP ---
LANDMARK_FEATURE_SIZE = 63
LANDMARK_CACHE_VERSION = 1
LANDMARK_TRAIN_EPOCHS = 24
LANDMARK_BATCH_SIZE = 512

# --- Secuencias para senas dinamicas ---
SEQUENCE_LENGTH = 32
SEQUENCE_FEATURES = 69
DYNAMIC_MODEL_MIN_BYTES = 50000
DYNAMIC_CLASS_NAMES = (
    'hello',
    'thank_you',
    'sorry',
    'help',
    'please',
    'yes',
    'no',
    'water',
    'good',
    'bad',
)

LANDMARK_MODEL_WEIGHT = 0.78
IMAGE_MODEL_WEIGHT = 0.22

LANDMARK_EXCLUDED_CLASSES = {'nothing'}
NO_HAND_LABEL = 'no_hand'

IMAGE_MODEL_MIN_BYTES = 100000
LANDMARK_MODEL_MIN_BYTES = 50000

SPECIAL_ACTIONS = {
    'del': 'delete',
    'nothing': 'ignore',
    'space': 'space',
}

DISPLAY_LABELS = {
    'del': 'BORRAR',
    'no_hand': 'SIN MANO',
    'nothing': 'SIN SENA',
    'space': 'ESPACIO',
}

GRAPHS_DIR = BASE_ML_DIR / 'static' / 'images'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# --- Seguridad API ---
MAX_IMAGE_PAYLOAD_BYTES = 2 * 1024 * 1024
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 1
