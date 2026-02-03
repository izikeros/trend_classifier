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
# # Quick Start Examples
#
# This notebook contains examples featured in the README documentation:
# - Basic stock price segmentation (AAPL)
# - Accessing segment properties
# - Custom segment classification
# - Synthetic signal examples (sine, triangular waves)
#
# **Prerequisites:** `pip install trend-classifier yfinance`

# %% [markdown]
# ## Example 1: Basic Stock Price Segmentation
#
# The simplest way to use trend_classifier with financial data.

# %%
import yfinance as yf

from trend_classifier import Segmenter

# Download Apple stock data
df = yf.download("AAPL", start="2018-09-15", end="2022-09-06", interval="1d", progress=False)

# Create segmenter and calculate segments
seg = Segmenter(df=df, column="Adj Close", n=20)
seg.calculate_segments()
seg.plot_segments()

# %%
# Number of data points
print(f"Total data points: {len(df)}")
print(f"Segments detected: {len(seg.segments)}")

# %% [markdown]
# ## Example 2: Accessing Segment Properties
#
# Each segment contains detailed information about the detected trend.

# %%
# View the first segment
segment = seg.segments[0]
print(f"Segment representation:\n{repr(segment)}")

# %%
# Access specific properties
print(f"Start indices: {segment.starts}")
print(f"Stop index: {segment.stop}")
print(f"Slope: {segment.slope:.4f}")

# %% [markdown]
# ## Example 3: Plotting Individual Segments with Trendlines

# %%
# Plot a specific segment with its trendlines (no surrounding context)
seg.plot_segment_with_trendlines_no_context(idx=1)

# %% [markdown]
# ## Example 4: Custom Segment Classification
#
# Classify segments based on their slope into up/down/horizontal trends.

# %%
def segment_classifier(s, threshold=0.1):
    """Classify a segment based on its slope."""
    if s.slope > threshold:
        return "up"
    elif s.slope < -threshold:
        return "down"
    else:
        return "horizontal"


# Classify all segments
for i, s in enumerate(seg.segments):
    classification = segment_classifier(s)
    print(f"Segment {i}: slope={s.slope:.3f} -> {classification}")

# %%
# Get indices of each category
up_idx = [i for i, s in enumerate(seg.segments) if segment_classifier(s) == "up"]
down_idx = [i for i, s in enumerate(seg.segments) if segment_classifier(s) == "down"]
horiz_idx = [i for i, s in enumerate(seg.segments) if segment_classifier(s) == "horizontal"]

print(f"Uptrend segments: {up_idx}")
print(f"Downtrend segments: {down_idx}")
print(f"Horizontal segments: {horiz_idx}")

# %% [markdown]
# ### Visualize Each Category

# %%
# Plot uptrend segments
if up_idx:
    print("Uptrend segments:")
    seg.plot_segment(up_idx)

# %%
# Plot downtrend segments
if down_idx:
    print("Downtrend segments:")
    seg.plot_segment(down_idx)

# %%
# Plot horizontal segments
if horiz_idx:
    print("Horizontal segments:")
    seg.plot_segment(horiz_idx)

# %% [markdown]
# ## Example 5: Synthetic Signals
#
# trend_classifier works with any time series, not just financial data.

# %% [markdown]
# ### Sine Wave

# %%
import numpy as np

from trend_classifier import Segmenter

# Generate a sine wave
x = np.linspace(0, 2 * np.pi, 200).tolist()
y = np.sin(x).tolist()

seg = Segmenter(x=x, y=y)
segments = seg.calculate_segments()
seg.plot_segments()

print(f"Detected {len(segments)} segments in sine wave")

# %% [markdown]
# ### Absolute Sine Wave (Rectified)

# %%
x = np.linspace(0, 2 * np.pi, 200).tolist()
y = np.abs(np.sin(x)).tolist()

seg = Segmenter(x=x, y=y)
segments = seg.calculate_segments()
seg.plot_segments()

print(f"Detected {len(segments)} segments in rectified sine wave")

# %% [markdown]
# ### Triangular Wave (Small Period)

# %%
x = list(range(200))
y = (list(range(25)) + list(range(25, 0, -1))) * 4  # 4 cycles

seg = Segmenter(x=x, y=y)
segments = seg.calculate_segments()
seg.plot_segments()

print(f"Detected {len(segments)} segments in triangular wave (small period)")

# %% [markdown]
# ### Triangular Wave (Large Period)

# %%
x = list(range(200))
y = (list(range(50)) + list(range(50, 0, -1))) * 2  # 2 cycles

seg = Segmenter(x=x, y=y)
segments = seg.calculate_segments()
seg.plot_segments()

print(f"Detected {len(segments)} segments in triangular wave (large period)")
