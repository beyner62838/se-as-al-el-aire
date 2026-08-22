"""Descarga de dataset y modelos auxiliares."""

from urllib.request import urlopen

from django.conf import settings

from proyecto.ml.constants import (
    DATASET_DIR,
    DATASET_HANDLE,
    HAND_LANDMARKER_MODEL_PATH,
    HAND_LANDMARKER_MODEL_URL,
    MODEL_DIR,
    TRAIN_DIR,
)


def dataset_is_ready():
    return TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir())


def ensure_hand_landmarker_model():
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


def model_file_looks_ready(model_path, min_bytes):
    return model_path.exists() and model_path.stat().st_size >= min_bytes


def load_saved_keras_model(tf, model_path, min_bytes):
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
