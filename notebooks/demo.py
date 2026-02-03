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
# # Segment Classification Demo
#
# This notebook demonstrates how to:
# - Configure the Segmenter with custom parameters
# - Classify detected segments by trend direction (up/down/horizontal)
# - Filter segments by slope magnitude and volatility (standard deviation)
# - Visualize specific segment categories
#
# **Prerequisites:** `pip install trend-classifier yfinance`

# %% [markdown]
# ## Setup and Configuration

# %%
import warnings

import pandas as pd
import yfinance as yf

from trend_classifier.configuration import CONFIG_REL
from trend_classifier.segmentation import Segmenter

warnings.filterwarnings("ignore")

# %%
# Configure the segmenter parameters
cfg = CONFIG_REL
cfg.N = 80  # Window size for trend calculation
cfg.overlap_ratio = 0.33  # Overlap between adjacent windows
cfg.alpha = 0.5  # Weight for slope difference in segment merging
cfg.beta = 0.5  # Weight for offset difference in segment merging

# %% [markdown]
# ## Download and Process Data

# %%
symbol = "AAPL"
df = yf.download(symbol, start="2018-09-15", end="2022-09-05", interval="1d", progress=False)

column = "Adj Close"
x_in = list(range(0, len(df.index.tolist()), 1))
y_in = df[column].tolist()

# %% [markdown]
# ## Calculate and Visualize Segments

# %%
seg = Segmenter(x_in, y_in, cfg)
seg.calculate_segments()
seg.describe_segments()
seg.plot_segments(fig_size=(22, 5))

# %%
# View all detected segments
seg.segments

# %% [markdown]
# ## Analyze Segment Properties
#
# Extract key properties from each segment for classification:
# - **Slope**: Direction and magnitude of the trend
# - **Std**: Standard deviation (volatility/noise within the segment)
# - **Span**: Length of the segment

# %%
seg_list = []
for s in seg.segments:
    seg_list.append([s.slope, s.slopes_std, s.span])

df_segments = pd.DataFrame(seg_list, columns=["Slope", "Std", "Span"]).sort_values("Slope")
df_segments

# %%
df_segments.describe()

# %% [markdown]
# ## Segment Classification
#
# Define thresholds for classifying segments:
# - **SL_TH (Slope Threshold)**: Segments with |slope| < 0.2 are considered horizontal
# - **STD_TH (Std Threshold)**: Segments with std < 0.1 are considered "clean" (low volatility)

# %%
SL_TH = 0.2  # Slope threshold for horizontal classification
STD_TH = 0.1  # Standard deviation threshold for volatility classification

# %% [markdown]
# ### Horizontal Segments - Low Volatility
# Flat trends with consistent behavior (sideways consolidation)

# %%
df_horiz_low = df_segments.loc[(abs(df_segments.Slope) < SL_TH) & (df_segments.Std < STD_TH)]
idx = df_horiz_low.index.tolist()
if idx:
    seg.plot_segment(idx, fig_size=(15, 4))
else:
    print("No segments match this criteria")

# %% [markdown]
# ### Horizontal Segments - High Volatility
# Flat trends with high noise (choppy sideways movement)

# %%
df_horiz_hi = df_segments.loc[(abs(df_segments.Slope) < SL_TH) & (df_segments.Std > STD_TH)]
idx = df_horiz_hi.index.tolist()
if idx:
    seg.plot_segment(idx, fig_size=(15, 4))
else:
    print("No segments match this criteria")

# %% [markdown]
# ### Uptrend Segments - Low Volatility
# Clean upward trends (strong bullish momentum)

# %%
df_up_low = df_segments.loc[(df_segments.Slope > SL_TH) & (df_segments.Std < STD_TH)]
idx = df_up_low.index.tolist()
if idx:
    seg.plot_segment(idx, fig_size=(15, 4))
else:
    print("No segments match this criteria")

# %% [markdown]
# ### Uptrend Segments - High Volatility
# Volatile upward trends (bullish with high uncertainty)

# %%
df_up_high = df_segments.loc[(df_segments.Slope > SL_TH) & (df_segments.Std > STD_TH)]
idx = df_up_high.index.tolist()
if idx:
    seg.plot_segment(idx, fig_size=(15, 4))
else:
    print("No segments match this criteria")

# %% [markdown]
# ### Downtrend Segments - Low Volatility
# Clean downward trends (strong bearish momentum)

# %%
df_down_low = df_segments.loc[(df_segments.Slope < -SL_TH) & (df_segments.Std < STD_TH)]
idx = df_down_low.index.tolist()
if idx:
    seg.plot_segment(idx, fig_size=(15, 4))
else:
    print("No segments match this criteria")

# %% [markdown]
# ### Downtrend Segments - High Volatility
# Volatile downward trends (bearish with high uncertainty)

# %%
df_down_high = df_segments.loc[(df_segments.Slope < -SL_TH) & (df_segments.Std > STD_TH)]
idx = df_down_high.index.tolist()
if idx:
    seg.plot_segment(idx, fig_size=(15, 4))
else:
    print("No segments match this criteria")
