# Contributing

Thank you for your interest in contributing to trend_classifier!

## Development Setup

1. **Clone the repository:**

```bash
git clone https://github.com/izikeros/trend_classifier.git
cd trend_classifier
```

2. **Set up development environment:**

```bash
make dev
```

This installs:
- All dependencies via uv
- Pre-commit hooks
- Development tools

3. **Verify setup:**

```bash
make test
make lint
```

## Making Changes

### Code Style

- **Formatter:** Ruff (Black-compatible)
- **Line length:** 88 characters
- **Type hints:** Required for public functions
- **Docstrings:** Google style

```bash
make format  # Auto-format code
make lint    # Check style
```

### Testing

- **Framework:** pytest
- **Coverage:** Maintain >80%

```bash
make test      # Run tests
make test-cov  # With coverage report
```

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

## Working with Notebooks

See [AGENTS.md](https://github.com/izikeros/trend_classifier/blob/main/AGENTS.md) for detailed notebook workflow.

**Key points:**

1. Edit `.py` files only (never `.ipynb`)
2. Sync before editing: `uv run jupytext --sync notebooks/*.py`
3. Sync after editing: `uv run jupytext --sync notebooks/*.py`
4. Commit both `.py` and `.ipynb`

## Adding a New Detector

1. Create `src/trend_classifier/detectors/my_detector.py`:

```python
from trend_classifier.detectors.base import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    name = "my_detector"
    
    def __init__(self, param1=10):
        self.param1 = param1
        self._x = self._y = None
    
    def fit(self, x, y):
        self._x, self._y = x, y
        return self
    
    def detect(self):
        self._validate_fitted()
        # Your logic here
        return DetectionResult(segments=..., breakpoints=...)
```

2. Register in `detectors/__init__.py`:

```python
from .my_detector import MyDetector
DETECTOR_REGISTRY["my_detector"] = MyDetector
```

3. Add tests in `tests/test_detectors.py`

4. Update documentation

## Pull Request Process

1. Create a feature branch from `main`
2. Make changes following the style guide
3. Add/update tests
4. Run verification:

```bash
make lint
make test
make security
```

5. Commit with conventional commit message
6. Push and create PR
7. Address review feedback

## Questions?

- Open an issue for bugs or feature requests
- Email: ksafjan@gmail.com
- GitHub: [@izikeros](https://github.com/izikeros)
