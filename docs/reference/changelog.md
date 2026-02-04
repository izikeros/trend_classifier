# Changelog

All notable changes to trend_classifier are documented here.

## [0.3.0] - 2026-02-04

### Added

- **Pluggable detector system** - Strategy pattern for swappable algorithms
- **New detectors:**
  - `PELTDetector` - Optimal change point detection via ruptures
  - `BottomUpDetector` - Merge-based segmentation
- **Optional dependencies:**
  - `trend-classifier[pelt]` - PELT algorithm
  - `trend-classifier[optimization]` - Optuna tuning
  - `trend-classifier[ml]` - scikit-learn features
  - `trend-classifier[all]` - Everything
- **New API:** `Segmenter(detector="pelt", detector_params={...})`
- **DetectionResult** dataclass with breakpoints and metadata
- **New notebooks:**
  - Detector comparison tutorial
  - Improved Optuna optimization

### Changed

- Refactored `Segmenter` as facade class
- Internal arrays now use numpy float64 for efficiency
- Improved error handling for near-zero slopes

### Fixed

- yfinance multi-index column handling
- Optuna objective now uses composite score to prevent over-segmentation

## [0.2.5] - 2024-xx-xx

### Added

- Progress callback for long sequences
- Input validation for data length

### Fixed

- numpy warnings for small segments

## [0.2.0] - 2024-xx-xx

### Added

- `to_dataframe()` method for SegmentList
- Detrended signal plotting
- Configuration presets (CONFIG_ABS, CONFIG_REL)

### Changed

- Migrated to Pydantic v2

## [0.1.0] - 2022-xx-xx

### Added

- Initial release
- Sliding window segmentation
- Basic visualization
- DataFrame input support
