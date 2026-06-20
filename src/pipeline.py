"""Training, evaluation, and inference orchestration."""

from __future__ import annotations

from pathlib import Path
import logging

try:
    from config import PipelineConfig
    from data import build_prediction_payload, load_train_data
    from model import build_model, evaluate_model, load_model, save_model
except ModuleNotFoundError:
    from src.config import PipelineConfig
    from src.data import build_prediction_payload, load_train_data
    from src.model import build_model, evaluate_model, load_model, save_model
    try:
        from features import extract_cnn_features
    except ModuleNotFoundError:
        try:
            from src.features import extract_cnn_features
        except Exception:
            extract_cnn_features = None
try:
    from features import extract_cnn_features
except ModuleNotFoundError:
    try:
        from src.features import extract_cnn_features
    except Exception:
        extract_cnn_features = None
logger = logging.getLogger(__name__)


def train(config: PipelineConfig) -> Path:
    logger.info("Starting training pipeline")
    X_train, X_test, y_train, y_test = load_train_data(
        config.data_dir,
        random_state=config.random_state,
        image_size=config.image_size,
    )
    # Optionally extract CNN features first
    if getattr(config, "use_cnn", False):
        if extract_cnn_features is None:
            raise RuntimeError("CNN feature extractor is not available because PyTorch is not installed.")
        logger.info("Extracting CNN features using %s", config.cnn_model_name)
        features_df = extract_cnn_features(
            config.data_dir, image_size=config.image_size, batch_size=config.cnn_batch_size, model_name=config.cnn_model_name
        )
        X = features_df.drop(columns=["label", "file_name"])
        y = features_df["label"]
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = load_train_data(
            config.data_dir, random_state=config.random_state, image_size=config.image_size
        )
    pipeline = build_model(config.random_state)
    pipeline.fit(X_train, y_train)
    save_model(pipeline, config.model_path)
    evaluate_model(pipeline, X_test, y_test)
    return config.model_path


def evaluate(config: PipelineConfig) -> dict[str, float]:
    logger.info("Starting evaluation pipeline")
    model = load_model(config.model_path)
    _, X_test, _, y_test = load_train_data(
        config.data_dir,
        random_state=config.random_state,
        image_size=config.image_size,
    )
    return evaluate_model(model, X_test, y_test)


def predict(config: PipelineConfig) -> Path:
    logger.info("Starting prediction pipeline")
    model = load_model(config.model_path)
    payload = build_prediction_payload(config.data_dir, image_size=config.image_size)
    features = payload.drop(columns=["file_name"], errors="ignore")
    predictions = model.predict(features)
    result_path = config.output_dir / "predictions.csv"
    payload["prediction"] = predictions
    payload.to_csv(result_path, index=False)
    logger.info("Wrote predictions to %s", result_path)
    return result_path
