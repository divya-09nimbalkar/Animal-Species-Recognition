"""Data handling and synthetic dataset support."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from image_data import discover_image_dataset, load_image_dataset, build_image_prediction_payload
except ModuleNotFoundError:
    from src.image_data import discover_image_dataset, load_image_dataset, build_image_prediction_payload

logger = logging.getLogger(__name__)


def discover_dataset(data_dir: Path, image_size: int = 128) -> pd.DataFrame:
    data_dir = data_dir.expanduser().resolve()
    candidates = ["annotations.csv", "features.csv"]

    for candidate in candidates:
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            logger.info("Loading dataset from %s", candidate_path)
            return pd.read_csv(candidate_path)

    if discover_image_dataset(data_dir):
        logger.info("Loading image dataset from %s", data_dir)
        return load_image_dataset(data_dir, image_size=image_size)

    logger.warning(
        "No dataset file found in %s. Generating a synthetic sample dataset.",
        data_dir,
    )
    return _create_synthetic_dataset()


def _create_synthetic_dataset(rows: int = 300, classes: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(rows, 16))
    labels = rng.choice([f"species_{i+1}" for i in range(classes)], size=rows)
    columns = {f"feature_{i}": features[:, i] for i in range(features.shape[1])}
    columns["label"] = labels
    return pd.DataFrame(columns)


def load_train_data(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    image_size: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = discover_dataset(data_dir, image_size=image_size)
    if "label" not in dataset.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X = dataset.drop(columns=["label"])
    y = dataset["label"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_prediction_payload(data_dir: Path, rows: int = 3, image_size: int = 128) -> pd.DataFrame:
    if discover_image_dataset(data_dir):
        return build_image_prediction_payload(data_dir, image_size=image_size, rows=rows)

    dataset = discover_dataset(data_dir, image_size=image_size)
    if "label" in dataset.columns:
        return dataset.drop(columns=["label"]).head(rows)
    return dataset.head(rows)
