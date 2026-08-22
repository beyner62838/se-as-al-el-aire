"""Utilidades de etiquetas, tokens y conversión de imágenes."""

from base64 import b64encode
from io import BytesIO

from proyecto.ml.constants import (
    DISPLAY_LABELS,
    NO_HAND_LABEL,
    PREDICTION_THRESHOLD,
    SPECIAL_ACTIONS,
)


def pretty_label(label):
    return DISPLAY_LABELS.get(label, label)


def format_preview_token(label):
    if label == 'space':
        return '<espacio>'
    if label == 'del':
        return '<borrar>'
    if label == NO_HAND_LABEL:
        return '<sin-mano>'
    if label == 'nothing':
        return '<sin-sena>'
    return label


def resolve_prediction_action(label, confidence, hand_detected=True):
    if not hand_detected or label == NO_HAND_LABEL:
        return 'ignore'
    if confidence < PREDICTION_THRESHOLD:
        return 'ignore'
    return SPECIAL_ACTIONS.get(label, 'append')


def build_action_token(label, confidence, hand_detected=True):
    if not hand_detected or label == NO_HAND_LABEL:
        return ''
    if confidence < PREDICTION_THRESHOLD:
        return ''
    if label == 'space':
        return ' '
    if label in {'del', 'nothing'}:
        return ''
    return label


def pil_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    encoded = b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


def normalize_probability_vector(probabilities):
    from proyecto.ml.deps import import_ml_dependencies

    np, _ = import_ml_dependencies()
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return np.zeros_like(probabilities)
    return probabilities / total


def expand_landmark_probabilities(prediction, supported_class_names, global_class_names):
    from proyecto.ml.deps import import_ml_dependencies

    np, _ = import_ml_dependencies()
    expanded = np.zeros(len(global_class_names), dtype='float32')
    for index, class_name in enumerate(supported_class_names):
        expanded[global_class_names.index(class_name)] = float(prediction[index])
    return expanded
