# Installation

## Basic Installation

Install trend_classifier from PyPI:

```bash
pip install trend-classifier
```

Or with uv:

```bash
uv add trend-classifier
```

## Optional Dependencies

trend_classifier has optional extras for extended functionality:

### PELT Algorithm

For the PELT change-point detection algorithm (via [ruptures](https://centre-borelli.github.io/ruptures-docs/)):

```bash
pip install trend-classifier[pelt]
```

### Hyperparameter Optimization

For automatic parameter tuning with [Optuna](https://optuna.org/):

```bash
pip install trend-classifier[optimization]
```

### Machine Learning

For scikit-learn based features:

```bash
pip install trend-classifier[ml]
```

### All Extras

Install everything:

```bash
pip install trend-classifier[all]
```

## Development Installation

For contributing to trend_classifier:

```bash
git clone https://github.com/izikeros/trend_classifier.git
cd trend_classifier
make dev
```

This sets up:

- Virtual environment with uv
- All development dependencies
- Pre-commit hooks

## Verify Installation

```python
import trend_classifier
print(trend_classifier.__version__)

# Check available detectors
from trend_classifier import list_detectors
print(list_detectors())
```

Expected output:

```
0.3.0
['sliding_window', 'bottom_up', 'pelt']  # pelt only if ruptures installed
```

## Requirements

- Python 3.10+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- pydantic >= 2.0.0
