# How to Tune Parameters

This guide shows how to optimize segmentation parameters for your data.

## Manual Tuning

### Sliding Window Parameters

```python
from trend_classifier import Segmenter
from trend_classifier.configuration import Config

# Create custom config
config = Config(
    N=40,              # Window size
    overlap_ratio=0.33, # Window overlap (0-1)
    alpha=2.0,         # Slope threshold
    beta=2.0,          # Offset threshold
)

seg = Segmenter(df=df, config=config)
seg.calculate_segments()
```

**Parameter effects:**

| Parameter | ↑ Increase | ↓ Decrease |
|-----------|------------|------------|
| `N` | Smoother, fewer segments | More sensitive, more segments |
| `alpha` | Fewer segments | More segments |
| `beta` | Fewer segments | More segments |
| `overlap_ratio` | More overlap, slower | Less overlap, faster |

### Visual Comparison

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for ax, n in zip(axes, [20, 40, 80]):
    seg = Segmenter(df=df, n=n)
    seg.calculate_segments()
    
    ax.plot(df.index, df["Close"])
    for s in seg.segments:
        ax.axvline(x=df.index[s.start], color='red', alpha=0.5)
    ax.set_title(f"N={n}: {len(seg.segments)} segments")

plt.tight_layout()
plt.show()
```

## Automated Tuning with Optuna

For optimal parameters, use hyperparameter optimization:

```python
# Requires: pip install trend-classifier[optimization]
import optuna

def objective(trial):
    n = trial.suggest_int("n", 20, 100, step=10)
    alpha = trial.suggest_float("alpha", 0.5, 5.0, step=0.5)
    beta = trial.suggest_float("beta", 0.5, 5.0, step=0.5)
    
    seg = Segmenter(df=df, detector="sliding_window", detector_params={
        "n": n, "alpha": alpha, "beta": beta
    })
    seg.calculate_segments()
    
    # Objective: minimize error + segment count penalty
    error = seg.calc_area_outside_trend()
    n_segments = len(seg.segments)
    target_segments = 20
    
    penalty = 0.1 * ((n_segments - target_segments) / target_segments) ** 2
    return error + penalty

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
```

## Preset Configurations

trend_classifier includes preset configurations:

```python
from trend_classifier.configuration import CONFIG_ABS, CONFIG_REL

# Absolute error for slope comparison
seg = Segmenter(df=df, config=CONFIG_ABS)

# Relative error (default, better for varying scales)
seg = Segmenter(df=df, config=CONFIG_REL)
```

## Tips by Use Case

### Financial Data (Daily)

```python
config = Config(N=40, alpha=2.0, beta=2.0)  # ~2 months window
```

### Financial Data (Intraday)

```python
config = Config(N=60, alpha=1.5, beta=1.5)  # 1 hour with 1-min bars
```

### Noisy Sensor Data

```python
# Use bottom-up for better noise handling
seg = Segmenter(x=x, y=y, detector="bottom_up", detector_params={
    "max_segments": 10
})
```

### Finding Exact Change Points

```python
# PELT gives optimal change points
seg = Segmenter(x=x, y=y, detector="pelt", detector_params={
    "penalty": 10,
    "model": "l2"
})
```

## Evaluating Results

### Quantitative Metrics

```python
seg.calculate_segments()

# Fitting error (lower = better fit)
print(f"Error: {seg.calc_area_outside_trend():.4f}")

# Segment count
print(f"Segments: {len(seg.segments)}")

# Average segment length
lengths = [s.stop - s.start for s in seg.segments]
print(f"Avg length: {sum(lengths)/len(lengths):.1f}")
```

### Visual Inspection

Always visualize results:

```python
seg.plot_segments()  # Overview
seg.plot_detrended_signal()  # Residuals
```

## See Also

- [Configuration Tutorial](../tutorials/04_configuration.ipynb) - Deep dive
- [Optimization Tutorial](../tutorials/06_advanced_optimization.ipynb) - Full Optuna example
