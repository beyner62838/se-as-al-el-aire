"""Inferencia de secuencias para el clasificador de senas dinamicas."""

from __future__ import annotations

import numpy as np

from proyecto.ml.sequence_dataset import build_sequence_features


def predict_sequence(model, sequence):
    """Predice una secuencia de landmarks y devuelve etiqueta y confianza."""
    features, _ = build_sequence_features(sequence)
    probabilities = np.asarray(model.predict(features[np.newaxis, ...], verbose=0)[0])
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("El modelo temporal devolvio una salida invalida.")
    class_index = int(np.argmax(probabilities))
    return class_index, float(probabilities[class_index])