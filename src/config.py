"""Pipeline configuration for Animal Species Recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    mode: str
    data_dir: Path
    output_dir: Path
    model_path: Path
    batch_size: int
    epochs: int
    image_size: int = 128
    use_cnn: bool = False
    cnn_model_name: str = "resnet18"
    cnn_batch_size: int = 32
    feature_cache: Path = Path("models/features")
    random_state: int = 42
    verbose: bool = False

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
