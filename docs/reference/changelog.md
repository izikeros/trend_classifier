# Changelog

All notable changes to trend_classifier are documented here.

## [0.3.0] - 2026-02-04

### Added
- **Pluggable detector system** with Strategy pattern
- **New detectors:** PELTDetector (via ruptures), BottomUpDetector
- **Optional dependencies:** `[pelt]`, `[optimization]`, `[ml]`, `[all]`
- New API: `Segmenter(detector="pelt", detector_params={...})`
- `DetectionResult` dataclass with breakpoints and metadata
- Comprehensive MkDocs documentation with tutorials
- Detector comparison notebook (07)

### Changed
- Refactored Segmenter as facade class
- Internal arrays now use numpy float64

### Fixed
- Optuna objective now uses composite score to prevent over-segmentation

## [0.2.5] - 2026-02-03

### Added
- Progress callback for long sequences
- Input validation for data length

### Changed
- Improved algorithm robustness for near-zero slopes

### Fixed
- yfinance multi-index column handling in notebooks
- numpy warnings for small segments

## [0.2.4] - 2026-02-03

### Changed
- Migrated to hatchling build backend and uv package manager
- Unified CI workflows (setup-python@v5, uv caching)
- Standardized Makefile with release automation
- Reorganized notebooks into numbered learning path (01-06)

### Added
- AGENTS.md with AI agent instructions
- VISIBILITY_CHECKLIST.md
- Educational headers in all notebooks

### Removed
- Legacy pdm.lock, setup.cfg, tox configurations

## [0.2.3] - 2024-07-19

### Changed
- Updated pre-commit flake8 plugins
- Moved GitHub Actions workflows
- Code reformatting

## [0.2.2] - 2024-07-19

### Changed
- Updated docstrings

## [0.2.1] - 2024-07-19

### Added
- git-cliff configuration for changelog generation
- Codecov GitHub Action

### Changed
- Updated badges in README

## [0.2.0] - 2024-07-18

### Added
- Signal drawing utility application
- Pre-loaded test data signals
- Bumpversion configuration
- Codecov integration

### Changed
- **BREAKING:** Moved source code to `src/` folder
- Migrated from PDM to simplified pyproject.toml
- Simplified tox configuration

### Removed
- pytest GitHub Action (replaced with tox)
- Unused configuration files

## [0.1.12] - 2024-07-17

### Fixed
- Type hints compatibility issues

### Changed
- Updated development dependencies
- Removed Python 3.7 support

## [0.1.11] - 2024-07-17

### Added
- MANIFEST.in for source distribution
- GitHub Action to test with tox
- Test for README code example

### Changed
- Replaced poetry with pdm
- Package version bump

## [0.1.10] - 2024-03-21

### Added
- Ruff configuration
- Alternative trend detection documentation

### Changed
- Migrated from poetry to pdm (gradual)
- Dropped Python 3.8 support
- Updated flake8 configuration

## [0.1.9] - 2023-05-26

### Changed
- Upgraded pre-commit configuration
- Upgraded tox configuration
- Updated Makefile
- Updated requirements

## [0.1.8] - 2023-02-01

### Changed
- Updated type hints to support Python older than 3.10
- Updated pre-commit hooks

## [0.1.7] - 2022-09-13

### Added
- New configuration preset (CONFIG_REL_SLOPE_ONLY)
- Plot single segment without context method
- Allow None for alpha/beta in config

### Changed
- Improved decision making for segment boundaries
- Remove outstanding windows from segment listing

## [0.1.6] - 2022-09-09

### Added
- DataFrame screenshot image

### Changed
- Package version update

## [0.1.5] - 2022-09-09

### Added
- pandas as requirement
- Column reordering in segment DataFrame
- pytest-cov for coverage
- CodeClimate badges

### Changed
- Refactored plotting functions to separate module
- Divided large code into smaller methods

## [0.1.4] - 2022-09-08

### Added
- GitHub Actions for CI
- Badges in README
- Tests and test improvements
- Pre-commit hooks (name-tests-test)
- .nojekyll for docs

### Changed
- Updated type hints
- Upgraded Python in CI from 3.9 to 3.10.6
- Updated QA tool configurations

## [0.1.3] - 2022-09-08

### Added
- Segments class (SegmentList) to replace raw list
- `__all__` exports in package init
- Docstrings

### Changed
- Made internal functions private (_determine_trend_end_point, _error)
- Updated default fig_size for plot_segments

## [0.1.2] - 2022-09-07

### Changed
- Updated author link in README

## [0.1.1] - 2022-09-07

### Fixed
- Image link in README

## [0.1.0] - 2022-09-07

### Added
- Initial release
- Sliding window segmentation algorithm
- Configuration class with Pydantic
- Segment class for trend representation
- Basic visualization (plot_segments, plot_segment)
- DataFrame input support
- Example with yfinance data
