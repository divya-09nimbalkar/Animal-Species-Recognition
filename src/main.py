"""CLI entry point for the Animal Species Recognition pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from config import PipelineConfig
    from pipeline import evaluate, predict, train
    from utils import configure_logging
except ModuleNotFoundError:
    from src.config import PipelineConfig
    from src.pipeline import evaluate, predict, train
    from src.utils import configure_logging

__version__ = "0.1.0"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Animal_Species_Recognition",
        description="Train, evaluate, or run inference for an animal species recognition pipeline.",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Execution mode for the pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/raw"),
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../models"),
        help="Directory for models, outputs, and artifacts.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../models/animal_species_model.joblib"),
        help="Path to the saved model file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training or evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Resize images to this square dimension for ingestion.",
    )
    parser.add_argument(
        "--use-cnn",
        action="store_true",
        help="Enable CNN feature extraction before training.",
    )
    parser.add_argument(
        "--cnn-model-name",
        type=str,
        default="resnet18",
        help="CNN backbone to use for feature extraction.",
    )
    parser.add_argument(
        "--cnn-batch-size",
        type=int,
        default=32,
        help="Batch size used during CNN feature extraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        image_size=args.image_size,
        use_cnn=args.use_cnn,
        cnn_model_name=args.cnn_model_name,
        cnn_batch_size=args.cnn_batch_size,
        random_state=args.random_state,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    configure_logging(args.verbose)

    config = build_config(args)
    config.ensure_directories()

    if config.mode == "train":
        train(config)
    elif config.mode == "evaluate":
        evaluate(config)
    elif config.mode == "predict":
        predict(config)
    else:
        print(f"Unsupported mode: {config.mode}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
