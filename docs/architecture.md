# Architecture Overview

This project is built as a modular animal species recognition pipeline with a clear separation of responsibilities.

## Core modules

- `src/main.py`
  - CLI entrypoint and argument parsing.
- `src/config.py`
  - Data class for runtime pipeline configuration.
- `src/utils.py`
  - Logging setup and shared helpers.
- `src/data.py`
  - Dataset discovery, loading, and synthetic fallback support.
  - Supports tabular CSV datasets and image-folder datasets for computer vision ingestion.
- `src/model.py`
  - Model pipeline construction, persistence, and evaluation helpers.
- `src/pipeline.py`
  - High-level orchestration for training, evaluation, and prediction.

## Data flow

1. `main.py` parses CLI arguments and builds a `PipelineConfig`.
2. `pipeline.train()` loads data, trains a model, saves the checkpoint, and performs a validation pass.
3. `pipeline.evaluate()` loads the saved model and evaluates it on test data.
4. `pipeline.predict()` loads the saved model and writes predictions to `models/predictions.csv`.

## Project structure

- `data/raw/` — source dataset files, such as `annotations.csv` or `features.csv`.
- `data/processed/` — prepared artifacts and cleaned datasets.
- `models/` — trained model files, checkpoints, and prediction outputs.
- `notebooks/` — interactive exploratory analysis and model reports.
- `tests/` — automated unit tests.
- `docs/` — architecture, usage guidance, and extension notes.
