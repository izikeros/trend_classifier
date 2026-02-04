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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 - Quick Start
#
# Get up and running with trend_classifier in under 5 minutes.
#
# ## What You'll Learn
# - Install the package
# - Segment a time series into trends
# - Visualize the results
#
# ## Prerequisites
# ```bash
# pip install trend-classifier yfinance
# ```

# %% [markdown]
# ## Step 1: Import and Download Data

# %%
import yfinance as yf

from trend_classifier import Segmenter

# Download Apple stock data (4 years of daily prices)
df = yf.download("AAPL", start="2018-09-15", end="2022-09-05", interval="1d", progress=False)
print(f"Downloaded {len(df)} data points")

# %% [markdown]
# ## Step 2: Create Segmenter and Calculate Segments
#
# The `Segmenter` class is the main entry point. Pass your data and a window size `n`.

# %%
# Create segmenter - can pass DataFrame directly
seg = Segmenter(df=df, column="Close", n=20)

# Calculate segments (this finds regions with similar trends)
segments = seg.calculate_segments()

print(f"Found {len(segments)} segments")

# %% [markdown]
# ## Step 3: Visualize Results

# %%
# Plot all segments with trend lines
seg.plot_segments()

# %% [markdown]
# **Reading the plot:**
# - Blue line: Original price data
# - Green dashed lines: Upward trends
# - Red dashed lines: Downward trends
# - Gray vertical lines: Segment boundaries

# %% [markdown]
# ## Step 4: Access Segment Information

# %%
# Get the first segment
first_segment = seg.segments[0]
print(f"First segment: index {first_segment.start} to {first_segment.stop}")
print(f"Slope: {first_segment.slope:.4f}")

# %%
# Quick summary of all segments
for i, s in enumerate(seg.segments):
    direction = "↑" if s.slope > 0 else "↓"
    print(f"Segment {i}: {direction} slope={s.slope:+.3f}, length={s.stop - s.start}")

# %% [markdown]
# ## Next Steps
#
# - **02_segment_analysis.py** - Deep dive into segment properties
# - **03_visualization.py** - All plotting methods
# - **04_configuration.py** - Tuning parameters for better results
