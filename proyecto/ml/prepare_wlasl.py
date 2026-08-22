"""Descarga WLASL y crea secuencias de landmarks para el modelo dinamico."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from proyecto.ml.constants import DYNAMIC_CLASS_NAMES
from proyecto.ml.sequence_dataset import LANDMARK_COUNT, LANDMARK_COORDINATES

METADATA_PATH = Path('data/WLASL_v0.3.json')
VIDEO_DIR = Path('data/wlasl_videos')
OUTPUT_PATH = Path('data/dynamic_sequences.npz')


def normalize_gloss(gloss: str) -> str:
    return gloss.lower().replace(' ', '_')


def selected_entries(max_per_class: int):
    items = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    selected = []
    for label_index, class_name in enumerate(DYNAMIC_CLASS_NAMES):
        matches = [item for item in items if normalize_gloss(item['gloss']) == class_name]
        for item in matches[:1]:
            for instance in item.get('instances', [])[:max_per_class]:
                selected.append((label_index, class_name, instance))
    return selected


def download_video(instance: dict) -> Path | None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    destination = VIDEO_DIR / f"{instance['video_id']}.mp4"
    if destination.exists():
        return destination
    command = [
        sys.executable, '-m', 'yt_dlp', '--no-playlist', '--quiet', '--no-warnings',
        '-f', 'mp4[height<=480]/mp4/best',
        '-o', str(destination), instance['url'],
    ]
    try:
        subprocess.run(command, check=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    return destination if destination.exists() else None


def extract_sequence(video_path: Path, instance: dict, frame_count: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    start = max(0, int(instance.get('frame_start', 1)) - 1)
    end = int(instance.get('frame_end', -1))
    if end < 0:
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = np.linspace(start, max(start, end - 1), frame_count).astype(int)
    frames = []
    from proyecto.views import analyze_hand_image

    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        analysis = analyze_hand_image(Image.fromarray(rgb))
        if analysis['hand_detected']:
            frames.append(np.asarray(analysis['landmarks'], dtype='float32'))
        else:
            frames.append(np.zeros((LANDMARK_COUNT, LANDMARK_COORDINATES), dtype='float32'))
    capture.release()
    if len(frames) != frame_count:
        return None
    return np.stack(frames).astype('float32')


def prepare(max_per_class: int, frame_count: int) -> None:
    import django
    import os
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proyecto.settings')
    django.setup()

    sequences = []
    labels = []
    for label_index, class_name, instance in selected_entries(max_per_class):
        video_path = download_video(instance)
        if video_path is None:
            print(f'SKIP download {class_name} {instance["video_id"]}')
            continue
        sequence = extract_sequence(video_path, instance, frame_count)
        if sequence is None:
            print(f'SKIP landmarks {class_name} {instance["video_id"]}')
            continue
        sequences.append(sequence)
        labels.append(label_index)
        print(f'OK {class_name} {instance["video_id"]}')

    if not sequences:
        raise RuntimeError('No se pudo preparar ninguna secuencia.')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, sequences=np.stack(sequences), labels=np.asarray(labels, dtype='int32'))
    print(f'Guardado {OUTPUT_PATH}: {len(sequences)} secuencias')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-class', type=int, default=10)
    parser.add_argument('--frames', type=int, default=32)
    args = parser.parse_args()
    prepare(args.max_per_class, args.frames)


if __name__ == '__main__':
    main()
