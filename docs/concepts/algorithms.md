# Algorithms

How each detection algorithm works.

## Sliding Window Detector

**Type:** Online, heuristic

### How It Works

1. Slide a window of size `N` across the data with overlap
2. Fit a linear trend (y = mx + b) in each window using least squares
3. Compare slope/offset to previous window
4. If difference exceeds threshold, start new segment

```
Data:    ────────────────────────────────────
Window 1: [======]
Window 2:    [======]
Window 3:       [======]  ← slope changes → NEW SEGMENT
Window 4:          [======]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | 60 | Window size |
| `overlap_ratio` | 0.33 | Window overlap (0-1) |
| `alpha` | 2.0 | Slope change threshold |
| `beta` | 2.0 | Offset change threshold |

### Complexity

- **Time:** O(n × N / offset) ≈ O(n) for typical settings
- **Space:** O(n) for storing segments

### When to Use

✅ General-purpose trend detection  
✅ Need interpretable parameters  
✅ Interactive parameter tuning  
❌ Very noisy data (may oversegment)  
❌ Need guaranteed optimal results

---

## Bottom-Up Detector

**Type:** Offline, merge-based

### How It Works

1. Start with many small segments (size = `initial_segment_size`)
2. Calculate merge cost for each adjacent pair
3. Merge the pair with lowest cost
4. Repeat until reaching `max_segments`

```
Initial:  [=][=][=][=][=][=][=][=][=][=]
Step 1:   [=][=][=][==][=][=][=][=][=]   ← lowest cost merge
Step 2:   [=][=][===][=][=][=][=][=]
...
Final:    [========][============]
```

### Merge Cost

The cost of merging segments A and B is:

```
cost = error(merged) - error(A) - error(B)
```

Where `error` is the sum of squared residuals from the linear fit.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_segments` | 10 | Target number of segments |
| `initial_segment_size` | 5 | Starting segment size |
| `merge_cost_threshold` | None | Stop if cost exceeds this |

### Complexity

- **Time:** O(n² / initial_segment_size) worst case
- **Space:** O(n)

### When to Use

✅ Know desired segment count  
✅ Noisy data  
✅ Want consistent granularity  
❌ Very long sequences (slow)  
❌ Need optimal breakpoints

---

## PELT Detector

**Type:** Offline, exact optimization

### How It Works

PELT (Pruned Exact Linear Time) finds the optimal segmentation that minimizes:

```
∑ cost(segment_i) + penalty × (number of segments)
```

Key insight: It prunes candidate change points that can never be optimal, achieving O(n) average complexity.

### Cost Models

| Model | Detects | Formula |
|-------|---------|---------|
| `l2` | Mean shifts | Sum of squared deviations from mean |
| `l1` | Median shifts | Sum of absolute deviations |
| `rbf` | Distribution changes | Kernel-based |
| `linear` | Slope changes | Linear regression error |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `penalty` | Auto | Higher = fewer segments |
| `model` | "l2" | Cost model |
| `min_size` | 2 | Minimum segment length |

### Complexity

- **Time:** O(n) average, O(n²) worst case
- **Space:** O(n)

### When to Use

✅ Need optimal segmentation  
✅ Large datasets  
✅ Academic/research applications  
❌ Need fine-grained control  
❌ Avoid external dependencies

---

## Comparison

### Accuracy vs Speed

```
Accuracy      PELT (optimal)
    ▲           │
    │           │
    │       Bottom-Up
    │           │
    │   Sliding Window
    │           │
    └───────────┴──────────► Speed
              Fast
```

### By Data Characteristics

| Data Type | Recommended |
|-----------|-------------|
| Clean, trends | `sliding_window` |
| Noisy | `bottom_up` or `pelt` |
| Level shifts | `pelt` with `model="l2"` |
| Trend changes | `sliding_window` or `pelt` with `model="linear"` |
| Very long | `pelt` (O(n)) |

### Quality Metrics

Compare algorithms using:

```python
seg.calc_area_outside_trend()  # Fitting error (lower = better)
len(seg.segments)              # Segment count
```

Balance both - more segments always gives lower error but may overfit.
