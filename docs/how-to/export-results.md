# How to Export Results

This guide shows how to export segmentation results for further analysis.

## Export to DataFrame

The most common export method:

```python
from trend_classifier import Segmenter

seg = Segmenter(df=df, n=40)
seg.calculate_segments()

# Export to pandas DataFrame
df_segments = seg.segments.to_dataframe()
print(df_segments)
```

Output:
```
   start  stop     slope    offset  slopes_std  offsets_std       std      span reason_for_new_segment
0      0    45  0.234521  120.5678    0.045123     2.345678  1.234567  45.67890                  slope
1     46   120 -0.156789  145.6789    0.032456     3.456789  2.345678  67.89012                 offset
2    121   200  0.089012  130.1234    0.028901     1.234567  1.567890  34.56789        slope and offset
```

## Column Descriptions

| Column | Description |
|--------|-------------|
| `start` | Start index of segment |
| `stop` | End index of segment |
| `slope` | Trend slope (positive = rising) |
| `offset` | Y-intercept of trend line |
| `slopes_std` | Std dev of window slopes |
| `offsets_std` | Std dev of window offsets |
| `std` | Std dev of detrended values |
| `span` | Normalized range of values |
| `reason_for_new_segment` | Why segment ended |

## Export to CSV

```python
df_segments = seg.segments.to_dataframe()
df_segments.to_csv("segments.csv", index=False)
```

## Export to JSON

```python
import json

# Convert segments to dict
segments_data = [
    {
        "start": s.start,
        "stop": s.stop,
        "slope": s.slope,
        "offset": s.offset,
        "std": s.std,
    }
    for s in seg.segments
]

with open("segments.json", "w") as f:
    json.dump(segments_data, f, indent=2)
```

## Access Individual Segments

```python
# Iterate over segments
for i, segment in enumerate(seg.segments):
    print(f"Segment {i}: {segment.start}-{segment.stop}, slope={segment.slope:.4f}")

# Access specific segment
first_segment = seg.segments[0]
print(f"First segment slope: {first_segment.slope}")
print(f"First segment volatility (std): {first_segment.std}")
```

## Get Detrended Signal

```python
# Access detrended values (residuals)
detrended = seg.y_de_trended

# Plot
import matplotlib.pyplot as plt
plt.plot(detrended)
plt.title("Detrended Signal")
plt.axhline(y=0, color='red', linestyle='--')
plt.show()
```

## Calculate Segment Statistics

```python
import numpy as np

df_segments = seg.segments.to_dataframe()

# Basic statistics
print(f"Number of segments: {len(df_segments)}")
print(f"Average segment length: {(df_segments['stop'] - df_segments['start']).mean():.1f}")
print(f"Average slope: {df_segments['slope'].mean():.4f}")

# Classify trends
df_segments['trend'] = np.where(
    df_segments['slope'] > 0.01, 'up',
    np.where(df_segments['slope'] < -0.01, 'down', 'flat')
)

print(df_segments['trend'].value_counts())
```

## Merge with Original Data

```python
# Add segment labels to original data
df['segment_id'] = -1
for i, segment in enumerate(seg.segments):
    df.iloc[segment.start:segment.stop+1, df.columns.get_loc('segment_id')] = i

# Now you can group by segment
print(df.groupby('segment_id')['Close'].agg(['mean', 'std', 'min', 'max']))
```

## Export Visualization

```python
import matplotlib.pyplot as plt

# Save plot to file
seg.plot_segments()
plt.savefig("segments.png", dpi=150, bbox_inches='tight')
plt.close()
```

## See Also

- [Segment Analysis Tutorial](../tutorials/02_segment_analysis.ipynb) - Detailed examples
- [API Reference: Segment](../reference/api/segment.md) - All segment attributes
