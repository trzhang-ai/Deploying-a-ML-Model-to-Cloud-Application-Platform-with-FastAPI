# Python 3.13 Migration Notes

This document summarizes the runtime migration completed for Python 3.13.

## Runtime baseline

- Python: `>=3.13,<3.14`
- FastAPI, pandas, NumPy, scikit-learn, and development tools are resolved in
  `uv.lock`.
- `uv sync --locked` is the canonical environment installation path.

## Code change

`OneHotEncoder` now uses `sparse_output=False`, matching the current
scikit-learn API.

## Verification

GitHub Actions recreates the locked Python 3.13 environment, runs the complete
pytest suite, and applies flake8 to the application, training, modeling, and
test code before deployment.
