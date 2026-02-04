# Tutorials

Step-by-step guides to master trend_classifier.

## Learning Path

These tutorials build on each other. We recommend following them in order:

| # | Tutorial | Duration | What You'll Learn |
|---|----------|----------|-------------------|
| 1 | [Quick Start](../getting-started/quickstart.md) | 5 min | Basic usage, first segmentation |
| 2 | [Segment Analysis](02_segment_analysis.ipynb) | 10 min | Segment properties, DataFrame export |
| 3 | [Visualization](03_visualization.ipynb) | 10 min | All plotting methods |
| 4 | [Configuration](04_configuration.ipynb) | 15 min | Parameters, presets, tuning |
| 5 | [Classification](05_classification.ipynb) | 10 min | Categorize trends |
| 6 | [Optimization](06_advanced_optimization.ipynb) | 20 min | Optuna hyperparameter tuning |
| 7 | [Detector Comparison](07_detector_comparison.ipynb) | 15 min | Compare algorithms |

## Prerequisites

Before starting, ensure you have:

```bash
pip install trend-classifier[all]
pip install yfinance  # For sample data
```

## Interactive Notebooks

All tutorials are Jupyter notebooks. You can:

1. **Read online** - View rendered versions in this documentation
2. **Run locally** - Clone the repo and run in Jupyter
3. **Open in Colab** - Click the Colab badge in each notebook

```bash
git clone https://github.com/izikeros/trend_classifier.git
cd trend_classifier
jupyter lab notebooks/
```

## Quick Reference

### Common Patterns

**Basic segmentation:**
```python
seg = Segmenter(df=df, column="Close", n=40)
seg.calculate_segments()
seg.plot_segments()
```

**Export results:**
```python
df = seg.segments.to_dataframe()
```

**Use different detector:**
```python
seg = Segmenter(df=df, detector="pelt", detector_params={"penalty": 5})
```

### Key Classes

- `Segmenter` - Main entry point
- `Segment` - Single trend segment
- `SegmentList` - Collection with DataFrame export
- `Config` - Configuration presets
