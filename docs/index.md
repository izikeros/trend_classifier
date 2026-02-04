# trend_classifier

**Automated signal segmentation, trend classification and analysis.**

[![PyPI version](https://img.shields.io/pypi/v/trend-classifier.svg)](https://pypi.org/project/trend-classifier/)
[![Python versions](https://img.shields.io/pypi/pyversions/trend-classifier.svg)](https://pypi.org/project/trend-classifier/)
[![License](https://img.shields.io/pypi/l/trend-classifier.svg)](https://github.com/izikeros/trend_classifier/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/trend-classifier.svg)](https://pypi.org/project/trend-classifier/)

---

## What is trend_classifier?

trend_classifier automatically segments time series data into regions with similar trends. It's designed for:

- **Financial analysis** - Identify bull/bear markets, support/resistance levels
- **Signal processing** - Segment sensor data, detect regime changes
- **Algorithmic trading** - Extract trend features for trading strategies

## Key Features

- **Multiple detection algorithms** - Choose the best algorithm for your data
- **Easy to use** - Get started in 5 lines of code
- **Flexible output** - Export to DataFrames, visualize with matplotlib
- **Well tested** - 80%+ code coverage, production ready

## Quick Example

```python
import yfinance as yf
from trend_classifier import Segmenter

# Download stock data
df = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Segment the time series
seg = Segmenter(df=df, column="Close", n=40)
seg.calculate_segments()

# Visualize results
seg.plot_segments()

# Export to DataFrame
df_segments = seg.segments.to_dataframe()
```

![Segmentation Example](https://github.com/izikeros/trend_classifier/blob/main/img/screenshoot_1.jpg?raw=true)

## Available Detectors

| Detector | Description | Best For |
|----------|-------------|----------|
| `sliding_window` | Window-based with linear regression | General use, interpretable |
| `bottom_up` | Merge-based segmentation | Noisy data, target segment count |
| `pelt` | PELT algorithm (requires ruptures) | Optimal segmentation, large data |

```python
# Use different detectors
seg = Segmenter(df=df, detector="pelt", detector_params={"penalty": 5})
seg = Segmenter(df=df, detector="bottom_up", detector_params={"max_segments": 10})
```

## Installation

```bash
pip install trend-classifier
```

With optional dependencies:

```bash
pip install trend-classifier[pelt]         # PELT algorithm
pip install trend-classifier[optimization] # Optuna tuning
pip install trend-classifier[all]          # Everything
```

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Quick Start**

    ---

    Get up and running in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step guides with real examples

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-compass:{ .lg .middle } **How-To Guides**

    ---

    Solve specific problems

    [:octicons-arrow-right-24: How-To](how-to/choose-detector.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation

    [:octicons-arrow-right-24: Reference](reference/api/segmenter.md)

</div>
