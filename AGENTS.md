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
│   ├── configuration.py      # Config classes
│   ├── segment.py            # Segment class
│   └── segmentation.py       # Segmenter class
├── tests/                    # Test files
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks
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
- Tests use yfinance for sample data (in test group)

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

## Domain-Specific Notes

- Package works with pandas DataFrames for time series data
- Segmentation uses pydantic for configuration validation
- matplotlib is used for visualization
- numpy for numerical computations

## Contact

- Author: Krystian Safjan
- Email: ksafjan@gmail.com
- GitHub: [@izikeros](https://github.com/izikeros)
</coding_guidelines>
