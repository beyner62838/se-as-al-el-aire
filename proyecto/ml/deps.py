"""Importadores diferidos de dependencias ML pesadas."""


def import_ml_dependencies():
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            'Faltan dependencias de machine learning. Activa el entorno virtual e instala requirements.txt.'
        ) from exc
    return np, tf


def import_image_dependencies():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            'Falta Pillow. Instala las dependencias del proyecto antes de usar la camara.'
        ) from exc
    return Image, ImageOps


def import_mediapipe_dependencies():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            'Falta MediaPipe. Activa el entorno virtual e instala requirements.txt para usar landmarks.'
        ) from exc
    return mp
