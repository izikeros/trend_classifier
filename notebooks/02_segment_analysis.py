# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 - Segment Analysis
#
# Deep dive into segment properties and data export.
#
# ## What You'll Learn
# - Understand all segment attributes
# - Export segments to pandas DataFrame
# - Analyze segment statistics
# - Interpret why segments were created

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import yfinance as yf

from trend_classifier import Segmenter

# Download data
df = yf.download("AAPL", start="2018-09-15", end="2022-09-05", interval="1d", progress=False)

# Create and calculate segments
seg = Segmenter(df=df, column="Adj Close", n=20)
seg.calculate_segments()

print(f"Analyzing {len(seg.segments)} segments")

# %% [markdown]
# ## Segment Attributes
#
# Each `Segment` object contains rich information about the detected trend:
#
# | Attribute | Description |
# |-----------|-------------|
# | `start` | Start index in the time series |
# | `stop` | End index in the time series |
# | `slope` | Overall slope of the linear trend |
# | `offset` | Y-intercept of the linear trend |
# | `std` | Standard deviation after detrending (volatility) |
# | `span` | Normalized range of values |
# | `slopes` | List of slopes from individual windows |
# | `offsets` | List of offsets from individual windows |
# | `slopes_std` | Std of slopes (trend consistency) |
# | `offsets_std` | Std of offsets |
# | `reason_for_new_segment` | Why this segment ended |

# %%
# Examine a single segment in detail
segment = seg.segments[2]

print("=== Segment Details ===")
print(f"Range: index {segment.start} to {segment.stop} ({segment.stop - segment.start} points)")
print(f"Slope: {segment.slope:.4f}")
print(f"Offset: {segment.offset:.2f}")
print(f"Volatility (std): {segment.std:.4f}")
print(f"Span: {segment.span:.1f}")
print(f"Trend consistency (slopes_std): {segment.slopes_std:.4f}")
print(f"Reason segment ended: '{segment.reason_for_new_segment}'")

# %%
# The full representation
print("\nFull repr:")
print(repr(segment))

# %% [markdown]
# ## Export to DataFrame
#
# The `to_dataframe()` method converts all segments to a pandas DataFrame for easy analysis.

# %%
# Convert segments to DataFrame
df_segments = seg.segments.to_dataframe()
df_segments

# %%
# Select most useful columns
df_summary = df_segments[["start", "stop", "slope", "std", "span", "reason_for_new_segment"]].copy()
df_summary["length"] = df_summary["stop"] - df_summary["start"]
df_summary

# %% [markdown]
# ## Statistical Analysis

# %%
# Descriptive statistics
print("=== Segment Statistics ===")
print(f"Number of segments: {len(df_segments)}")
print(f"Average segment length: {df_summary['length'].mean():.1f} points")
print(f"Shortest segment: {df_summary['length'].min()} points")
print(f"Longest segment: {df_summary['length'].max()} points")

# %%
# Slope distribution
print("\n=== Slope Distribution ===")
print(df_segments["slope"].describe())

# %%
# Count trends by direction
uptrends = (df_segments["slope"] > 0).sum()
downtrends = (df_segments["slope"] < 0).sum()

print(f"\nUptrends: {uptrends} ({100*uptrends/len(df_segments):.1f}%)")
print(f"Downtrends: {downtrends} ({100*downtrends/len(df_segments):.1f}%)")

# %% [markdown]
# ## Understanding `reason_for_new_segment`
#
# This attribute explains why the algorithm decided to end a segment:
# - **"slope"**: The slope changed significantly
# - **"offset"**: The offset changed significantly  
# - **"slope and offset"**: Both changed

# %%
# Count reasons
reason_counts = df_segments["reason_for_new_segment"].value_counts()
print("Why segments ended:")
print(reason_counts)

# %% [markdown]
# ## Volatility Analysis
#
# The `std` attribute measures volatility after removing the linear trend.
# Low std = clean trend, High std = noisy/choppy trend.

# %%
# Find cleanest and noisiest trends
cleanest_idx = df_segments["std"].idxmin()
noisiest_idx = df_segments["std"].idxmax()

print(f"Cleanest trend: Segment {cleanest_idx} (std={df_segments.loc[cleanest_idx, 'std']:.4f})")
print(f"Noisiest trend: Segment {noisiest_idx} (std={df_segments.loc[noisiest_idx, 'std']:.4f})")

# %%
# Visualize cleanest vs noisiest
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, idx, title in [(axes[0], cleanest_idx, "Cleanest Trend"), 
                        (axes[1], noisiest_idx, "Noisiest Trend")]:
    s = seg.segments[idx]
    x_seg = seg.x[s.start:s.stop]
    y_seg = seg.y[s.start:s.stop]
    ax.plot(x_seg, y_seg)
    ax.set_title(f"{title} (std={s.std:.4f})")
    ax.set_xlabel("Index")
    ax.set_ylabel("Price")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Next Steps
#
# - **03_visualization.py** - All plotting methods in detail
# - **04_configuration.py** - Tune parameters to get better segments
