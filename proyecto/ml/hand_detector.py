"""Detección de manos y extracción de landmarks con MediaPipe."""

import math
from functools import lru_cache

from proyecto.ml.constants import (
    HAND_LANDMARKER_MODEL_PATH,
    HAND_MARGIN_RATIO,
    LANDMARK_FEATURE_SIZE,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
)
from proyecto.ml.dataset import ensure_hand_landmarker_model
from proyecto.ml.deps import import_image_dependencies, import_mediapipe_dependencies, import_ml_dependencies


@lru_cache(maxsize=1)
def get_hand_detector():
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


def analyze_hand_image(image):
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
    }


def select_best_hand_index(hand_landmarks_list):
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
    np, _ = import_ml_dependencies()
    coords = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype='float32',
    )
    if coords.shape != (21, 3):
        return None

    coords = coords - coords[0]

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

    if coords[5, 0] > coords[17, 0]:
        coords[:, 0] *= -1.0

    scale = float(np.max(np.linalg.norm(coords, axis=1)))
    if scale < 1e-6:
        return None

    coords /= scale
    return coords.reshape(-1).astype('float32')
