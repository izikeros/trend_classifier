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
# # Basic Trend Detection
#
# A minimal example showing the core workflow:
# 1. Download financial data
# 2. Calculate trend segments
# 3. Visualize results
# 4. Analyze detrended segments
#
# **Note:** This notebook demonstrates the `Segmenter` class which is
# the recommended high-level API for trend detection.
#
# **Prerequisites:** `pip install trend-classifier yfinance`

# %% [markdown]
# ## Setup and Data Download

# %%
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from trend_classifier import Segmenter

# %%
# Download stock data
symbol = "AAPL"
df = yf.download(symbol, start="2018-09-15", end="2022-08-15", progress=False)

print(f"Downloaded {len(df)} data points for {symbol}")

# %% [markdown]
# ## Calculate Trend Segments
#
# The Segmenter automatically:
# - Detects regions with consistent trends
# - Merges adjacent windows with similar slopes
# - Provides segment metadata (slope, offset, span)

# %%
column = "Adj Close"
x = list(range(len(df)))
y = df[column].tolist()

# Create segmenter and calculate segments
seg = Segmenter(x=x, y=y, n=20)
segments = seg.calculate_segments()

print(f"Detected {len(segments)} segments")

# %% [markdown]
# ## Visualize Segments

# %%
seg.plot_segments(fig_size=(12, 5))

# %% [markdown]
# ## Analyze Segment Properties

# %%
# Show segment details
seg.describe_segments()

# %% [markdown]
# ## Detrending Analysis
#
# Remove the linear trend from each segment to analyze the residual
# volatility. This helps identify:
# - **Low std**: Clean trends with consistent direction
# - **High std**: Noisy trends with high volatility

# %%
plt.figure(figsize=(12, 5))

detrended_all = []
segment_stats = []

for idx, segment in enumerate(seg.segments):
    # Get segment data
    start, stop = segment.start, segment.stop
    xx = x[start:stop]
    yy = y[start:stop]

    # Fit linear trend
    fit = np.polyfit(xx, yy, 1)
    fit_fn = np.poly1d(fit)
    trend = fit_fn(xx)

    # Calculate detrended signal
    detrended = np.array(yy) - np.array(trend)
    detrended_all.extend(detrended)

    # Calculate statistics
    std = np.std(detrended)
    span = 1000 * (np.max(detrended) - np.min(detrended)) / np.mean(yy)

    segment_stats.append({"segment": idx, "std": std, "span": span, "slope": segment.slope})
    print(f"Segment {idx}: std={std:.3f}, span={span:.1f}, slope={segment.slope:.4f}")

# Plot detrended signal
plt.plot(detrended_all, alpha=0.7)
plt.xlabel("Data Point")
plt.ylabel("Detrended Value")
plt.title("Detrended Signal (Residuals After Removing Linear Trends)")

# Add vertical lines at segment boundaries
cumsum = 0
for segment in seg.segments:
    cumsum += segment.span
    plt.axvline(x=cumsum, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Volatility vs Span Scatter Plot
#
# Visualize the relationship between segment duration and volatility.

# %%
import pandas as pd

df_stats = pd.DataFrame(segment_stats)

plt.figure(figsize=(8, 6))
plt.scatter(df_stats["span"], df_stats["std"], alpha=0.7, s=50)
plt.xlabel("Span (normalized range)")
plt.ylabel("Standard Deviation")
plt.title("Segment Volatility vs. Span")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Summary statistics
print("\nSegment Statistics Summary:")
print(df_stats.describe())
