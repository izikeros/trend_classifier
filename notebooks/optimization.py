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
# # Hyperparameter Optimization with Optuna
#
# This notebook demonstrates how to optimize Segmenter parameters using Optuna.
#
# ## Parameters Being Optimized
#
# - **N**: Window size for trend calculation (larger = smoother, fewer segments)
# - **overlap_ratio**: Overlap between adjacent windows (higher = more granular detection)
# - **alpha**: Weight for slope difference when merging segments
# - **beta**: Weight for offset difference when merging segments
#
# ## Objective
#
# Minimize the "area outside trend" - the cumulative deviation between
# the actual signal and the fitted trendlines. Lower values indicate
# better fit of the detected trends to the data.
#
# **Prerequisites:** `pip install trend-classifier yfinance optuna`

# %% [markdown]
# ## Setup

# %%
import warnings

import optuna
import yfinance as yf

from trend_classifier.configuration import CONFIG_REL
from trend_classifier.segmentation import Segmenter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna output

# %% [markdown]
# ## Download Data
#
# Using Bitcoin (BTC-USD) as it has more varied trend patterns than typical stocks.

# %%
cfg = CONFIG_REL

symbol = "BTC-USD"
df = yf.download(symbol, start="2016-09-15", end="2022-09-05", interval="1d", progress=False)

column = "Adj Close"
x_in = list(range(0, len(df.index.tolist()), 1))
y_in = df[column].tolist()

print(f"Downloaded {len(df)} data points for {symbol}")

# %% [markdown]
# ## Define Optimization Objective
#
# The objective function:
# 1. Receives trial parameters from Optuna
# 2. Configures the Segmenter with those parameters
# 3. Calculates segments and measures fit quality
# 4. Returns the error metric to minimize

# %%
def objective(trial):
    """
    Optuna objective function for Segmenter hyperparameter optimization.

    Returns the "area outside trend" metric - lower is better.
    """
    # Sample parameters from search space
    N = trial.suggest_int(name="N", low=10, high=60, step=5)
    overlap = trial.suggest_float(name="overlap", low=0.2, high=0.8, step=0.2)
    alpha = trial.suggest_float(name="alpha", low=0.5, high=4, step=0.25)
    beta = trial.suggest_float(name="beta", low=0.5, high=4, step=0.25)

    # Configure segmenter
    cfg.N = N
    cfg.overlap_ratio = overlap
    cfg.alpha = alpha
    cfg.beta = beta

    # Calculate segments and measure error
    seg = Segmenter(x_in, y_in, cfg)
    seg.calculate_segments()

    # Return the area outside detected trends (minimize this)
    err = seg.calc_area_outside_trend()
    return err


# %% [markdown]
# ## Run Optimization
#
# Run 100 trials to find the best parameter combination.

# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# %% [markdown]
# ## Results

# %%
print("Best parameters found:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

print(f"\nBest error (area outside trend): {study.best_value:.4f}")

# %% [markdown]
# ## Visualize with Optimized Parameters

# %%
# Apply best parameters
cfg.N = study.best_params["N"]
cfg.overlap_ratio = study.best_params["overlap"]
cfg.alpha = study.best_params["alpha"]
cfg.beta = study.best_params["beta"]

# Create segmenter with optimized config
seg_optimized = Segmenter(x_in, y_in, cfg)
seg_optimized.calculate_segments()
seg_optimized.describe_segments()
seg_optimized.plot_segments(fig_size=(18, 5))

# %% [markdown]
# ## Compare with Default Parameters

# %%
# Reset to defaults
cfg_default = CONFIG_REL

seg_default = Segmenter(x_in, y_in, cfg_default)
seg_default.calculate_segments()

print(f"Default config: {len(seg_default.segments)} segments, error={seg_default.calc_area_outside_trend():.4f}")
print(f"Optimized config: {len(seg_optimized.segments)} segments, error={seg_optimized.calc_area_outside_trend():.4f}")
