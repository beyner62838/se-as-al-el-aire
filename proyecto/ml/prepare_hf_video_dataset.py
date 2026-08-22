"""Prepara videos ASL alojados directamente en Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from proyecto.ml.constants import DYNAMIC_CLASS_NAMES
from proyecto.ml.sequence_dataset import LANDMARK_COUNT, LANDMARK_COORDINATES

REPO = 'ZahidYasinMittha/American-Sign-Language-Dataset'
BASE_API = f'https://huggingface.co/api/datasets/{REPO}/tree/main'
BASE_RESOLVE = f'https://huggingface.co/datasets/{REPO}/resolve/main'
VIDEO_DIR = Path('data/hf_asl_videos')
OUTPUT_PATH = Path('data/dynamic_sequences_hf.npz')


def class_matches(path: str, class_name: str) -> bool:
    stem = Path(path).stem.upper().replace('_', ' ')
    target = class_name.replace('_', ' ').upper()
    return stem.rsplit('-', 1)[-1] == target


def list_video_paths() -> list[str]:
    root = json.loads(urllib.request.urlopen(BASE_API + '?recursive=false').read())
    paths = []
    for entry in root:
        if entry.get('type') != 'directory':
            continue
        entries = json.loads(urllib.request.urlopen(BASE_API + '/' + entry['path'] + '?recursive=true').read())
        paths.extend(item['path'] for item in entries if item.get('type') == 'file' and item['path'].lower().endswith('.mp4'))
    return paths


def download(path: str) -> Path | None:
    destination = VIDEO_DIR / Path(path).name
    if destination.exists():
        return destination
    url = BASE_RESOLVE + '/' + urllib.parse.quote(path, safe='/') + '?download=true'
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.read())
    except Exception:
        return None
    return destination if destination.exists() else None


def extract_sequence(video_path: Path, frame_count: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    positions = np.linspace(0, total - 1, frame_count).astype(int)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proyecto.settings')
    import django
    django.setup()
    from proyecto.views import analyze_hand_image

    frames = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        analysis = analyze_hand_image(Image.fromarray(rgb))
        if analysis['hand_detected']:
            frames.append(np.asarray(analysis['landmarks'], dtype='float32'))
        else:
            frames.append(np.zeros((LANDMARK_COUNT, LANDMARK_COORDINATES), dtype='float32'))
    capture.release()
    return np.stack(frames).astype('float32')


def prepare(max_per_class: int, frame_count: int) -> None:
    paths = list_video_paths()
    sequences = []
    labels = []
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.unlink(missing_ok=True)
    for label_index, class_name in enumerate(DYNAMIC_CLASS_NAMES):
        candidates = [path for path in paths if class_matches(path, class_name)][:max_per_class]
        for path in candidates:
            video_path = download(path)
            if video_path is None:
                print(f'SKIP download {path}')
                continue
            sequence = extract_sequence(video_path, frame_count)
            if sequence is None:
                print(f'SKIP landmarks {path}')
                continue
            sequences.append(sequence)
            labels.append(label_index)
            np.savez_compressed(
                OUTPUT_PATH,
                sequences=np.stack(sequences),
                labels=np.asarray(labels, dtype='int32'),
            )
            print(f'OK {class_name} {path}')
    if not sequences:
        raise RuntimeError('No se pudo preparar ninguna secuencia de Hugging Face.')
    np.savez_compressed(OUTPUT_PATH, sequences=np.stack(sequences), labels=np.asarray(labels, dtype='int32'))
    print(f'Guardado {OUTPUT_PATH}: {len(sequences)} secuencias')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-class', type=int, default=5)
    parser.add_argument('--frames', type=int, default=32)
    args = parser.parse_args()
    prepare(args.max_per_class, args.frames)


if __name__ == '__main__':
    main()
