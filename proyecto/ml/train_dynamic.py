"""Entrenamiento local del modelo temporal de senas dinamicas.

El archivo de entrada debe ser un NPZ con:
- sequences: array (ejemplos, frames, 21, 3)
- labels: indices enteros entre 0 y 9
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from proyecto.ml.constants import (
    DYNAMIC_CLASS_NAMES,
    DYNAMIC_MODEL_PATH,
    MODEL_DIR,
)
from proyecto.ml.sequence_dataset import build_sequence_features
from proyecto.ml.temporal_model import build_temporal_model


def load_training_data(input_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Carga y transforma las secuencias guardadas en un archivo NPZ."""
    with np.load(input_path, allow_pickle=False) as data:
        sequences = data['sequences']
        labels = np.asarray(data['labels'], dtype='int32')

    if sequences.ndim != 4 or sequences.shape[2:] != (21, 3):
        raise ValueError('sequences debe tener forma (ejemplos, frames, 21, 3).')
    if labels.ndim != 1 or labels.shape[0] != sequences.shape[0]:
        raise ValueError('labels debe tener un indice por secuencia.')
    if np.any(labels < 0) or np.any(labels >= len(DYNAMIC_CLASS_NAMES)):
        raise ValueError('Hay etiquetas fuera de las 10 clases configuradas.')

    features = np.stack(
        [build_sequence_features(sequence)[0] for sequence in sequences],
        axis=0,
    )
    return features, labels


def load_combined_training_data(input_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Combina varios NPZ con el mismo contrato de secuencias y etiquetas."""
    datasets = [load_training_data(input_path) for input_path in input_paths]
    features = np.concatenate([dataset[0] for dataset in datasets], axis=0)
    labels = np.concatenate([dataset[1] for dataset in datasets], axis=0)
    return features, labels


def train(input_paths: list[str], epochs: int, validation_split: float) -> None:
    """Entrena y guarda el modelo temporal y su lista de clases."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('Instala las dependencias del proyecto antes de entrenar.') from exc

    features, labels = load_combined_training_data(input_paths)
    model = build_temporal_model(tf, len(DYNAMIC_CLASS_NAMES))
    model.fit(
        features,
        labels,
        validation_split=validation_split,
        shuffle=True,
        epochs=epochs,
        batch_size=min(32, len(features)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(DYNAMIC_MODEL_PATH)
    (MODEL_DIR / 'asl_dynamic_class_names.json').write_text(
        json.dumps(list(DYNAMIC_CLASS_NAMES), ensure_ascii=True, indent=2),
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena el GRU de senas dinamicas.')
    parser.add_argument('--input', required=True, nargs='+', help='Archivos NPZ de secuencias.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--validation-split', type=float, default=0.2)
    args = parser.parse_args()
    train(args.input, args.epochs, args.validation_split)


if __name__ == '__main__':
    main()
