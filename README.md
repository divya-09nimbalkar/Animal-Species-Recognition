# Animal Species Recognition

## Overview
A modular animal species recognition pipeline built for reproducibility, extensibility, and real-world deployment.

This repository includes a complete training, evaluation, and prediction workflow with a clean package layout, automated tests, and architecture documentation.

## Features
- CLI-driven execution: `train`, `evaluate`, `predict`
- Structured data loading with synthetic fallback for rapid prototyping
- Scikit-learn model pipeline with preprocessing and evaluation
- Model persistence using `joblib`
- Clear modular separation: config, data, model, pipeline, and utilities

## Folder structure
- `data/`
  - `raw/` — raw dataset inputs
  - `processed/` — prepared dataset artifacts
- `notebooks/` — exploratory analysis and modeling notebooks
- `src/` — production-grade source code
- `tests/` — automated unit tests
- `models/` — exported model artifacts and prediction results
- `docs/` — architecture and usage documentation

## Setup
```bash
cd "Animal_Species_Recognition"
pip install -r requirements.txt
```

## Usage
Train the model:
```bash
python src/main.py --mode train --data-dir data/raw --output-dir models
```

For an image dataset, use a class-folder structure under `data/raw`:

```text
data/raw/
  species_cat/
    cat_01.jpg
    cat_02.jpg
  species_dog/
    dog_01.jpg
    dog_02.jpg
```

Then run:
```bash
python src/main.py --mode train --data-dir data/raw --output-dir models --image-size 128
```

To enable CNN feature extraction (optional):

1. Install PyTorch and torchvision for your platform. Example CPU-only wheel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. Run the pipeline with `--use-cnn` to extract ResNet features before training:

```bash
python src/main.py --mode train --data-dir data/raw --output-dir models --use-cnn --image-size 128
```

Evaluate a saved model:
```bash
python src/main.py --mode evaluate --data-dir data/raw --model-path models/animal_species_model.joblib
```

Generate predictions:
```bash
python src/main.py --mode predict --data-dir data/raw --model-path models/animal_species_model.joblib --output-dir models
```

## Testing
Run the unit tests:
```bash
python -m unittest discover -s tests
```

Open the demo notebook:
```bash
jupyter notebook notebooks/animal_species_demo.ipynb
```

## Documentation
See `docs/architecture.md` for a high-level project design and module responsibilities.
