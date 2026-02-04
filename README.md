# trend_classifier

[![PyPI version](https://img.shields.io/pypi/v/trend-classifier.svg)](https://pypi.org/project/trend-classifier/)
[![Python versions](https://img.shields.io/pypi/pyversions/trend-classifier.svg)](https://pypi.org/project/trend-classifier/)
[![License](https://img.shields.io/pypi/l/trend-classifier.svg)](https://github.com/izikeros/trend_classifier/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/trend-classifier.svg)](https://pypi.org/project/trend-classifier/)
[![codecov](https://codecov.io/gh/izikeros/trend_classifier/graph/badge.svg?token=ZC9GNJEQQ0)](https://codecov.io/gh/izikeros/trend_classifier)

Automated signal segmentation, trend classification and analysis.

> **[Documentation](https://izikeros.github.io/trend_classifier/)** | **[Tutorials](https://izikeros.github.io/trend_classifier/tutorials/)** | **[API Reference](https://izikeros.github.io/trend_classifier/reference/api/segmenter/)**

## Quick Start

```python
from trend_classifier import Segmenter

seg = Segmenter(x=x_data, y=y_data, n=20)
seg.calculate_segments()
seg.plot_segments()
```

![Segmentation example](https://github.com/izikeros/trend_classifier/blob/main/img/screenshoot_1.jpg?raw=true)

## Installation

```bash
pip install trend-classifier
```

**With optional dependencies:**

```bash
pip install trend-classifier[pelt]         # PELT algorithm (ruptures)
pip install trend-classifier[optimization] # Hyperparameter tuning (optuna)
pip install trend-classifier[all]          # All extras
```

## Features

- **Multiple detection algorithms**:
  - `sliding_window` - Original algorithm, interpretable, good for most cases
  - `bottom_up` - Merge-based, control exact segment count
  - `pelt` - Optimal segmentation via [ruptures](https://github.com/deepcharles/ruptures) library

- **Rich segment information**: slope, offset, volatility, trend consistency
- **DataFrame export**: `seg.segments.to_dataframe()`
- **Visualization**: `plot_segments()`, `plot_segment()`
- **Configurable**: Fine-tune sensitivity with `alpha`, `beta`, window size

## Example with Stock Data

```python
import yfinance as yf
from trend_classifier import Segmenter

# Download data
df = yf.download("AAPL", start="2020-01-01", end="2023-01-01", progress=False)

# Segment and visualize
seg = Segmenter(df=df, column="Close", n=20)
seg.calculate_segments()
seg.plot_segments()

# Export to DataFrame
seg.segments.to_dataframe()
```

## Using Different Detectors

```python
from trend_classifier import Segmenter

# PELT algorithm (requires: pip install trend-classifier[pelt])
seg = Segmenter(x=x, y=y, detector="pelt", detector_params={"penalty": 10})
seg.calculate_segments()

# Bottom-up with target segment count
seg = Segmenter(x=x, y=y, detector="bottom_up", detector_params={"max_segments": 10})
seg.calculate_segments()
```

## Segment Properties

Each segment contains:

| Property | Description |
|----------|-------------|
| `start`, `stop` | Index range |
| `slope` | Trend direction and steepness |
| `std` | Volatility (after detrending) |
| `reason_for_new_segment` | Why segment boundary was placed |

```python
segment = seg.segments[0]
print(f"Slope: {segment.slope:.4f}, Volatility: {segment.std:.4f}")
```

## Documentation

Full documentation with tutorials and API reference:

**https://izikeros.github.io/trend_classifier/**

## License

[MIT](LICENSE) © [Krystian Safjan](https://safjan.com/)
