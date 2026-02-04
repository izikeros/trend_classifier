<coding_guidelines>
# AI Agent Instructions

Instructions for AI agents working with the trend_classifier package.

## Overview

trend_classifier is a Python package for automated signal segmentation, trend classification and analysis. It's commonly used for financial time series analysis and algorithmic trading applications.

## Project Structure

```
trend_classifier/
├── src/trend_classifier/     # Main package source
│   ├── __init__.py
│   ├── configuration.py      # Config classes and presets
│   ├── segment.py            # Segment and SegmentList classes
│   ├── segmentation.py       # Segmenter facade class
│   ├── metrics.py            # Error calculation functions
│   ├── visuals.py            # Plotting functions
│   └── detectors/            # Pluggable detection algorithms
│       ├── base.py           # BaseDetector ABC, DetectionResult
│       ├── sliding_window.py # Original sliding window algorithm
│       ├── pelt.py           # PELT via ruptures (optional)
│       └── bottom_up.py      # Bottom-up merge segmentation
├── tests/                    # Test files
├── docs/                     # MkDocs documentation
├── notebooks/                # Jupytext paired notebooks (.py + .ipynb)
└── pyproject.toml            # Project configuration
```

## Development Standards

### Tooling Stack

- **Package Manager**: uv
- **Build Backend**: Hatchling
- **Linter/Formatter**: Ruff
- **Type Checker**: ty (Astral)
- **Testing**: pytest + pytest-cov
- **Security**: bandit + pip-audit
- **Documentation**: mkdocs-material
- **Notebooks**: jupytext (percent format)

### Quick Commands

```bash
make dev          # Set up development environment
make test         # Run tests
make test-cov     # Run tests with coverage
make lint         # Check code style
make format       # Auto-format code
make type-check   # Run type checker
make security     # Run security checks
make docs         # Build documentation
make serve-docs   # Serve docs locally
```

### Code Style

- Line length: 88 characters (Ruff/Black default)
- Imports: sorted by Ruff (isort rules)
- Type hints: Required for all public functions
- Docstrings: Google style

### Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: bug fix
docs: documentation changes
refactor: code refactoring
test: add/update tests
chore: maintenance tasks
perf: performance improvements
```

**Important**: Never add `Co-authored-by` lines to commit messages.

### Testing Requirements

- All new features must have tests
- Maintain >80% code coverage
- Run `make test-cov` before submitting PR
- Tests use yfinance for sample data (in dev dependencies)

### Release Process

Releases are automated via GitHub Actions when a version tag is pushed:

```bash
make release-patch  # 0.1.0 → 0.1.1
make release-minor  # 0.1.0 → 0.2.0
make release-major  # 0.1.0 → 1.0.0
```

### Security Considerations

- Never commit secrets, API keys, or credentials
- Run `make security` before committing
- Review bandit findings for security issues

## Working with Notebooks

This project uses **jupytext** to maintain notebooks as paired files:
- `.py` (percent format) - version controlled, human-readable, editable
- `.ipynb` (notebook) - for interactive use in Jupyter

### Workflow for Editing Notebooks

**CRITICAL**: Always work with the `.py` file, never edit `.ipynb` directly.

1. **Before making changes** - Sync to ensure .py has latest content:
   ```bash
   uv run jupytext --sync notebooks/<notebook>.py
   ```
   This ensures your .py file reflects any changes made interactively in the notebook.

2. **Make your edits** in the `.py` file only

3. **After making changes** - Sync back to .ipynb:
   ```bash
   uv run jupytext --sync notebooks/<notebook>.py
   ```

4. **Sync all notebooks at once**:
   ```bash
   uv run jupytext --sync notebooks/*.py
   ```

### Best Practices

- **Never edit .ipynb files directly** - your changes will be overwritten on next sync
- **Always sync before editing** - the .ipynb may have been modified interactively
- **Commit both files** - .py and .ipynb should always be committed together
- **Check sync status** before editing:
  ```bash
  uv run jupytext --test notebooks/*.py  # Fails if out of sync
  ```
- **Resolve merge conflicts** in the .py file only, then sync to regenerate .ipynb

### Notebook Format

Notebooks use jupytext percent format with YAML header:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook Title
#
# Description here.

# %%
# Code cell
import trend_classifier

# %% [markdown]
# ## Section Header
```

### Available Notebooks

| Notebook | Description |
|----------|-------------|
| 01_quickstart | 5-minute introduction |
| 02_segment_analysis | Segment properties and DataFrame export |
| 03_visualization | All plotting methods |
| 04_configuration | Parameter tuning |
| 05_classification | Trend categorization |
| 06_advanced_optimization | Optuna hyperparameter tuning |
| 07_detector_comparison | Compare detection algorithms |

## Domain-Specific Notes

### Architecture

- **Segmenter** is a facade class that delegates to pluggable detectors
- **BaseDetector** defines the interface for all detection algorithms
- **Strategy pattern** allows swapping algorithms at runtime

### Detection Algorithms

```python
# Legacy API (still works)
seg = Segmenter(x=x, y=y, n=40)

# New API with detector selection
seg = Segmenter(x=x, y=y, detector="pelt", detector_params={"penalty": 5})

# Available detectors: sliding_window, bottom_up, pelt (requires ruptures)
from trend_classifier import list_detectors
print(list_detectors())
```

### Optional Dependencies

Install extras for additional functionality:

```bash
pip install trend_classifier[pelt]         # PELT algorithm (ruptures)
pip install trend_classifier[optimization] # Optuna for tuning
pip install trend_classifier[ml]           # scikit-learn
pip install trend_classifier[all]          # Everything
```

### Key Classes

- **Segmenter**: Main entry point for segmentation
- **Segment**: Represents a detected trend segment
- **SegmentList**: List of segments with DataFrame export
- **Config**: Pydantic model for configuration
- **DetectionResult**: Result from detector with metadata

### Data Handling

- Works with pandas DataFrames (especially yfinance data)
- Handles multi-index columns from yfinance automatically
- Internal arrays are numpy float64 for efficiency

## Contact

- Author: Krystian Safjan
- Email: ksafjan@gmail.com
- GitHub: [@izikeros](https://github.com/izikeros)
</coding_guidelines>
