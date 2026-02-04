# Quick Start

Get up and running with trend_classifier in 5 minutes.

## Basic Usage

### 1. Import and Load Data

```python
import yfinance as yf
from trend_classifier import Segmenter

# Download Apple stock data
df = yf.download("AAPL", start="2020-01-01", end="2023-01-01", progress=False)
print(f"Loaded {len(df)} data points")
```

### 2. Create Segmenter and Detect Trends

```python
# Create segmenter with DataFrame input
seg = Segmenter(df=df, column="Close", n=40)

# Calculate segments
seg.calculate_segments()

print(f"Found {len(seg.segments)} segments")
```

### 3. Visualize Results

```python
# Plot all segments with trend lines
seg.plot_segments()
```

### 4. Analyze Segments

```python
# Export to DataFrame
df_segments = seg.segments.to_dataframe()
print(df_segments[["start", "stop", "slope", "std"]])
```

Output:
```
   start  stop      slope       std
0      0    45   0.234521  1.234567
1     46   120  -0.156789  2.345678
2    121   200   0.089012  1.567890
...
```

## Using Arrays Instead of DataFrames

```python
import numpy as np

# Your data as arrays
x = np.arange(500)
y = np.cumsum(np.random.randn(500))  # Random walk

# Create segmenter
seg = Segmenter(x=x, y=y, n=30)
seg.calculate_segments()
seg.plot_segments()
```

## Choosing a Detector

trend_classifier supports multiple detection algorithms:

=== "Sliding Window (Default)"

    ```python
    seg = Segmenter(df=df, detector="sliding_window", detector_params={
        "n": 40,           # Window size
        "alpha": 2.0,      # Slope threshold
        "beta": 2.0,       # Offset threshold
    })
    ```

=== "Bottom-Up"

    ```python
    seg = Segmenter(df=df, detector="bottom_up", detector_params={
        "max_segments": 10,  # Target number of segments
    })
    ```

=== "PELT"

    ```python
    # Requires: pip install trend-classifier[pelt]
    seg = Segmenter(df=df, detector="pelt", detector_params={
        "penalty": 5,   # Higher = fewer segments
        "model": "l2",  # Cost model
    })
    ```

## What's Next?

- **[Tutorials](../tutorials/index.md)** - Deep dive into each feature
- **[How-To: Choose a Detector](../how-to/choose-detector.md)** - Pick the right algorithm
- **[API Reference](../reference/api/segmenter.md)** - Complete documentation
