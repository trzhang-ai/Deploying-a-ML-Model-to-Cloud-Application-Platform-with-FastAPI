# Implementation package

This directory contains the runtime application, modeling code, data, and
serialized artifacts for the [Census Income Prediction API](../README.md).

## Key paths

- `main.py`: FastAPI schemas, startup artifact loading, and inference routes.
- `starter/ml/data.py`: categorical encoding and feature preparation.
- `starter/ml/model.py`: model training, inference, metrics, and slice reports.
- `starter/train_model.py`: cross-validation and final artifact generation.
- `data/census.csv`: cleaned Census Income dataset.
- `model/`: classifier, encoder, and label-binarizer artifacts.
- `model_card_template.md`: model behavior, intended use, and limitations.

The repository-level [README](../README.md) is the canonical setup and usage
guide. The locked `pyproject.toml` and `uv.lock` environment is the supported
dependency path.
