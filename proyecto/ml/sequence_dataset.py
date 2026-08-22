"""Preparacion de secuencias de landmarks para senas dinamicas."""

from __future__ import annotations

import numpy as np


LANDMARK_COUNT = 21
LANDMARK_COORDINATES = 3
SEQUENCE_LENGTH = 32
SEQUENCE_FEATURES = 69


def validate_landmark_sequence(sequence: np.ndarray) -> np.ndarray:
    """Valida una secuencia con forma (frames, 21, 3)."""
    values = np.asarray(sequence, dtype="float32")
    if values.ndim != 3 or values.shape[1:] != (LANDMARK_COUNT, LANDMARK_COORDINATES):
        raise ValueError("La secuencia debe tener forma (frames, 21, 3).")
    if values.shape[0] == 0:
        raise ValueError("La secuencia debe contener al menos un frame.")
    return values


def normalize_landmark_frame(frame: np.ndarray) -> np.ndarray:
    """Normaliza una mano respecto a la muneca y a su escala."""
    values = np.asarray(frame, dtype="float32")
    if values.shape != (LANDMARK_COUNT, LANDMARK_COORDINATES):
        raise ValueError("El frame debe tener forma (21, 3).")

    normalized = values - values[0]
    scale = float(np.max(np.linalg.norm(normalized, axis=1)))
    if scale < 1e-6:
        return np.zeros(LANDMARK_COUNT * LANDMARK_COORDINATES, dtype="float32")
    return (normalized / scale).reshape(-1).astype("float32")


def resample_sequence(sequence: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Interpola una secuencia a una cantidad fija de frames."""
    values = validate_landmark_sequence(sequence)
    if length < 2:
        raise ValueError("La longitud de la secuencia debe ser mayor que uno.")
    if values.shape[0] == length:
        return values.copy()

    source_positions = np.linspace(0.0, 1.0, values.shape[0])
    target_positions = np.linspace(0.0, 1.0, length)
    result = np.empty((length, LANDMARK_COUNT, LANDMARK_COORDINATES), dtype="float32")
    for landmark in range(LANDMARK_COUNT):
        for coordinate in range(LANDMARK_COORDINATES):
            result[:, landmark, coordinate] = np.interp(
                target_positions,
                source_positions,
                values[:, landmark, coordinate],
            )
    return result


def build_sequence_features(
    sequence: np.ndarray,
    length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera features de postura, trayectoria y velocidad, mas una mascara.

    Cada frame contiene 63 coordenadas normalizadas, la posicion de la muneca
    y la velocidad de la muneca: 66 valores en total.
    """
    values = resample_sequence(sequence, length)
    normalized_frames = np.stack(
        [normalize_landmark_frame(frame) for frame in values],
        axis=0,
    )
    wrist_positions = values[:, 0, :].copy()
    wrist_origin = wrist_positions[0].copy()
    hand_spans = values[:, 1:, :] - values[:, :1, :]
    wrist_scale = float(np.max(np.linalg.norm(hand_spans, axis=2)))
    if wrist_scale < 1e-6:
        wrist_scale = 1.0
    wrist_positions = (wrist_positions - wrist_origin) / wrist_scale
    wrist_velocity = np.vstack(
        [np.zeros((1, LANDMARK_COORDINATES), dtype="float32"), np.diff(wrist_positions, axis=0)]
    )
    features = np.concatenate(
        [normalized_frames, wrist_positions, wrist_velocity],
        axis=1,
    ).astype("float32")
    mask = np.ones(length, dtype="float32")
    return features, mask
