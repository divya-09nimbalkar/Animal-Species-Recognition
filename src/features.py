"""CNN-based feature extraction helpers using PyTorch (optional).

This module provides utilities to extract deep features from image datasets
using torchvision models. If `torch` or `torchvision` are not installed,
functions will raise ImportError with a helpful message.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except Exception:  # catch ImportError or others
    torch = None

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


def _get_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_feature_extractor(model_name: str = "resnet18", device: str | None = None) -> nn.Module:
    if torch is None:
        raise ImportError("PyTorch (torch) is required for CNN feature extraction. Install torch and torchvision to use this feature.")

    if device is None:
        device = _get_device()

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        # remove the final classification layer
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.eval()
    model.to(device)
    return model


def extract_cnn_features(
    data_dir: Path,
    image_size: int = 128,
    batch_size: int = 32,
    model_name: str = "resnet18",
    max_images: int | None = None,
) -> pd.DataFrame:
    """Extract deep features for each image and return a DataFrame with features and labels.

    The returned DataFrame contains columns `f_0..f_N-1`, plus `label` and `file_name`.
    """
    if torch is None:
        raise ImportError("PyTorch (torch) is required for CNN feature extraction. Install torch and torchvision to use this feature.")

    device = torch.device(_get_device())
    model = build_feature_extractor(model_name, device)

    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    root = _resolve_image_root(data_dir)

    rows: list[np.ndarray] = []
    labels: list[str] = []
    file_names: list[str] = []

    count = 0

    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for image_path in sorted(_iter_image_files(label_dir)):
            if max_images is not None and count >= max_images:
                break
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model(tensor)
                    # ResNet style output is (1, C, 1, 1) -> flatten
                    features = features.cpu().numpy().reshape(-1)
                    rows.append(features)
                    labels.append(label)
                    file_names.append(str(image_path.relative_to(root)))
                    count += 1
            except Exception as exc:
                logger.exception("Failed to process image %s: %s", image_path, exc)
        if max_images is not None and count >= max_images:
            break

    if len(rows) == 0:
        raise FileNotFoundError(f"No image files found in dataset root: {root}")

    # Build DataFrame
    columns = [f"f_{i}" for i in range(rows[0].shape[0])]
    df = pd.DataFrame(np.vstack(rows), columns=columns)
    df["label"] = labels
    df["file_name"] = file_names
    return df
