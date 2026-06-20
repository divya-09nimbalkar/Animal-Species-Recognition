"""Model definition, persistence, and evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_model(random_state: int = 42) -> Pipeline:
    logger.debug("Building model pipeline with random_state=%s", random_state)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    random_state=random_state,
                    max_iter=500,
                ),
            ),
        ]
    )


def save_model(model: Pipeline, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved trained model to %s", path)


def load_model(path: Path) -> Pipeline:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def evaluate_model(model: Pipeline, X, y) -> dict[str, float]:
    predictions = model.predict(X)
    report = classification_report(y, predictions, output_dict=True)
    accuracy = accuracy_score(y, predictions)
    logger.info("Evaluation accuracy: %.3f", accuracy)
    return {"accuracy": accuracy, "report": report}
