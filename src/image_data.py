"""Image dataset ingestion and feature extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def _resolve_image_root(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser().resolve()
    candidate = data_dir / "images"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return data_dir


def discover_image_dataset(data_dir: Path) -> bool:
    root = _resolve_image_root(data_dir)
    if not root.exists():
        return False

    class_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not class_dirs:
        return False

    for class_dir in class_dirs:
        if any(_iter_image_files(class_dir)):
            return True
    return False


def _extract_image_features(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.ravel()


def load_image_dataset(data_dir: Path, image_size: int = 128, max_images: int | None = None) -> pd.DataFrame:
    root = _resolve_image_root(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image dataset root not found: {data_dir}")

    rows: list[np.ndarray] = []
    labels: list[str] = []
    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()])

    for label_dir in class_dirs:
        label = label_dir.name
        for image_path in sorted(_iter_image_files(label_dir)):
            if max_images is not None and len(rows) >= max_images:
                break
            rows.append(_extract_image_features(image_path, image_size))
            labels.append(label)

    if len(rows) == 0:
        raise FileNotFoundError(f"No image files found in dataset root: {root}")

    columns = [f"pixel_{index}" for index in range(rows[0].shape[0])]
    dataset = pd.DataFrame(np.vstack(rows), columns=columns)
    dataset["label"] = labels
    return dataset


def build_image_prediction_payload(data_dir: Path, image_size: int = 128, rows: int = 3) -> pd.DataFrame:
    root = _resolve_image_root(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image dataset root not found: {data_dir}")

    features: list[np.ndarray] = []
    file_names: list[str] = []
    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()])

    for label_dir in class_dirs:
        for image_path in sorted(_iter_image_files(label_dir)):
            if len(features) >= rows:
                break
            features.append(_extract_image_features(image_path, image_size))
            file_names.append(str(image_path.relative_to(root)))
        if len(features) >= rows:
            break

    if len(features) == 0:
        raise FileNotFoundError(f"No image files found in dataset root: {root}")

    columns = [f"pixel_{index}" for index in range(features[0].shape[0])]
    payload = pd.DataFrame(np.vstack(features), columns=columns)
    payload["file_name"] = file_names
    return payload
