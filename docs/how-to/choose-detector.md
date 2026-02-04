# How to Choose a Detector

This guide helps you select the right detection algorithm for your use case.

## Quick Decision Tree

```
Is your data noisy?
├── Yes → Do you know the target segment count?
│   ├── Yes → Use `bottom_up`
│   └── No → Use `pelt` with higher penalty
└── No → Do you need interpretable parameters?
    ├── Yes → Use `sliding_window`
    └── No → Use `pelt` for optimal results
```

## Algorithm Comparison

| Criteria | sliding_window | bottom_up | pelt |
|----------|----------------|-----------|------|
| **Speed** | Fast | Medium | Fast |
| **Optimality** | Good | Good | Optimal |
| **Interpretability** | High | Medium | Low |
| **Noisy data** | Sensitive | Robust | Robust |
| **Control segments** | Indirect | Direct | Indirect |
| **Dependencies** | None | None | ruptures |

## When to Use Each

### Sliding Window (Default)

**Best for:**

- General-purpose trend detection
- When you need interpretable parameters
- Interactive exploration of data

```python
seg = Segmenter(df=df, detector="sliding_window", detector_params={
    "n": 40,           # Window size (larger = smoother)
    "alpha": 2.0,      # Slope sensitivity (larger = fewer splits)
    "overlap_ratio": 0.33,
})
```

**Tips:**

- Start with `n` = 5-10% of your data length
- Increase `alpha` if getting too many segments
- Use `beta=None` to ignore offset changes

### Bottom-Up

**Best for:**

- When you know how many segments you want
- Noisy data where sliding window oversegments
- Exploratory analysis with target granularity

```python
seg = Segmenter(df=df, detector="bottom_up", detector_params={
    "max_segments": 10,       # Target number of segments
    "initial_segment_size": 5,  # Starting granularity
})
```

**Tips:**

- Set `max_segments` based on expected trend changes
- Smaller `initial_segment_size` = more accurate boundaries
- Good for comparing different granularities

### PELT

**Best for:**

- Optimal segmentation (mathematically proven)
- Large datasets (O(n) complexity)
- When you want automatic segment count

```python
# Requires: pip install trend-classifier[pelt]
seg = Segmenter(df=df, detector="pelt", detector_params={
    "penalty": 5,      # Higher = fewer segments
    "model": "l2",     # Cost model (l2, l1, rbf)
    "min_size": 2,     # Minimum segment length
})
```

**Tips:**

- Start with `penalty = log(n) * 2` where n is data length
- Use `model="linear"` for trend detection
- Use `model="l2"` for level shift detection

## Comparing Results

Run all detectors on the same data to compare:

```python
from trend_classifier import Segmenter, list_detectors

results = {}
for detector in list_detectors():
    params = {"n": 40} if detector == "sliding_window" else \
             {"max_segments": 10} if detector == "bottom_up" else \
             {"penalty": 10}
    
    seg = Segmenter(df=df, detector=detector, detector_params=params)
    seg.calculate_segments()
    
    results[detector] = {
        "n_segments": len(seg.segments),
        "error": seg.calc_area_outside_trend(),
    }

print(results)
```

## See Also

- [Detector Comparison Tutorial](../tutorials/07_detector_comparison.ipynb) - Visual comparison
- [Tune Parameters](tune-parameters.md) - Fine-tune any detector
- [API Reference: Detectors](../reference/api/detectors.md) - Full parameter docs
