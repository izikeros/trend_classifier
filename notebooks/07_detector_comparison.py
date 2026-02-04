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
# # 07 - Detector Comparison
#
# Compare different segmentation algorithms on the same data.
#
# ## What You'll Learn
# - Available detection algorithms
# - How to switch between detectors
# - Performance and quality comparison
# - When to use which detector
#
# ## Available Detectors
#
# | Detector | Description | Best For |
# |----------|-------------|----------|
# | `sliding_window` | Original algorithm, window-based | General use, interpretable |
# | `bottom_up` | Merge-based segmentation | Noisy data, target segment count |
# | `pelt` | PELT via ruptures (optional) | Optimal segmentation, large data |

# %% [markdown]
# ## Setup

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from trend_classifier import Segmenter, list_detectors

warnings.filterwarnings("ignore")

# Check available detectors
print("Available detectors:", list_detectors())

# %% [markdown]
# ## Generate Test Data
#
# We'll create synthetic data with clear trend changes to compare algorithms.

# %%
np.random.seed(42)

# Create data with 4 distinct trends
n_points = 500
noise_level = 2.0

# Trend 1: Rising (0-100)
t1 = np.linspace(0, 30, 100) + np.random.normal(0, noise_level, 100)
# Trend 2: Flat (100-200)
t2 = 30 + np.random.normal(0, noise_level, 100)
# Trend 3: Falling (200-350)
t3 = np.linspace(30, 10, 150) + np.random.normal(0, noise_level, 150)
# Trend 4: Rising steep (350-500)
t4 = np.linspace(10, 50, 150) + np.random.normal(0, noise_level, 150)

y = np.concatenate([t1, t2, t3, t4])
x = np.arange(len(y), dtype=np.float64)

# True change points
true_breakpoints = [100, 200, 350]

print(f"Data: {len(y)} points with {len(true_breakpoints)} true change points")

# %%
# Visualize the data with true breakpoints
plt.figure(figsize=(14, 4))
plt.plot(x, y, 'b-', alpha=0.7, label='Signal')
for bp in true_breakpoints:
    plt.axvline(x=bp, color='red', linestyle='--', alpha=0.7, label='True breakpoint' if bp == true_breakpoints[0] else '')
plt.title("Test Signal with True Change Points")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare Detectors
#
# Let's run each available detector on the same data.

# %%
results = {}

# 1. Sliding Window Detector (default)
seg_sw = Segmenter(x=x, y=y, detector="sliding_window", detector_params={"n": 40, "alpha": 2.0})
seg_sw.calculate_segments()
results["sliding_window"] = {
    "segments": seg_sw.segments,
    "n_segments": len(seg_sw.segments),
    "error": seg_sw.calc_area_outside_trend(),
}

# 2. Bottom-Up Detector
seg_bu = Segmenter(x=x, y=y, detector="bottom_up", detector_params={"max_segments": 10})
seg_bu.calculate_segments()
results["bottom_up"] = {
    "segments": seg_bu.segments,
    "n_segments": len(seg_bu.segments),
    "error": seg_bu.calc_area_outside_trend(),
}

# 3. PELT Detector (if available)
if "pelt" in list_detectors():
    seg_pelt = Segmenter(x=x, y=y, detector="pelt", detector_params={"penalty": 10})
    seg_pelt.calculate_segments()
    results["pelt"] = {
        "segments": seg_pelt.segments,
        "n_segments": len(seg_pelt.segments),
        "error": seg_pelt.calc_area_outside_trend(),
    }

print("Results Summary:")
print("-" * 50)
for name, data in results.items():
    print(f"{name:20s}: {data['n_segments']:3d} segments, error={data['error']:.6f}")

# %% [markdown]
# ## Visual Comparison

# %%
n_detectors = len(results)
fig, axes = plt.subplots(n_detectors, 1, figsize=(14, 3 * n_detectors), sharex=True)

if n_detectors == 1:
    axes = [axes]

colors = {"sliding_window": "green", "bottom_up": "orange", "pelt": "purple"}

for ax, (name, data) in zip(axes, results.items()):
    ax.plot(x, y, 'b-', alpha=0.5, linewidth=1)
    
    # Plot segment boundaries
    for seg in data["segments"]:
        ax.axvline(x=seg.start, color=colors.get(name, 'gray'), linestyle='--', alpha=0.7)
    
    # Plot true breakpoints
    for bp in true_breakpoints:
        ax.axvline(x=bp, color='red', linestyle=':', alpha=0.5)
    
    ax.set_title(f"{name}: {data['n_segments']} segments, error={data['error']:.6f}")
    ax.set_ylabel("Value")

axes[-1].set_xlabel("Index")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Breakpoint Accuracy
#
# How close are detected breakpoints to the true ones?

# %%
def evaluate_breakpoints(segments, true_bps, tolerance=20):
    """Evaluate breakpoint detection accuracy."""
    detected_bps = [s.start for s in segments[1:]]  # Skip first segment start
    
    # Find matches within tolerance
    matches = 0
    for true_bp in true_bps:
        for det_bp in detected_bps:
            if abs(true_bp - det_bp) <= tolerance:
                matches += 1
                break
    
    precision = matches / len(detected_bps) if detected_bps else 0
    recall = matches / len(true_bps) if true_bps else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "detected": len(detected_bps),
        "true": len(true_bps),
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

print("Breakpoint Detection Accuracy (tolerance=20):")
print("-" * 60)
print(f"{'Detector':<20} {'Detected':>10} {'Matches':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 60)

for name, data in results.items():
    metrics = evaluate_breakpoints(data["segments"], true_breakpoints)
    print(f"{name:<20} {metrics['detected']:>10} {metrics['matches']:>10} "
          f"{metrics['precision']:>10.2f} {metrics['recall']:>10.2f} {metrics['f1']:>10.2f}")

# %% [markdown]
# ## When to Use Which Detector
#
# ### Sliding Window (`sliding_window`)
# - **Pros**: Interpretable, configurable sensitivity, good for most cases
# - **Cons**: Sensitive to window size, may miss abrupt changes
# - **Best for**: General use, when you need explainability
#
# ### Bottom-Up (`bottom_up`)
# - **Pros**: Control exact segment count, good for noisy data
# - **Cons**: Computationally heavier, may not find optimal breakpoints
# - **Best for**: When you know desired segment count, noisy signals
#
# ### PELT (`pelt`)
# - **Pros**: Optimal segmentation, fast (O(n)), well-studied algorithm
# - **Cons**: Requires ruptures library, penalty tuning needed
# - **Best for**: Large datasets, when optimal segmentation matters

# %% [markdown]
# ## Using Custom Detector Instances
#
# For more control, create detector instances directly:

# %%
from trend_classifier.detectors import SlidingWindowDetector, BottomUpDetector

# Custom sliding window
detector = SlidingWindowDetector(
    n=50,
    overlap_ratio=0.4,
    alpha=1.5,
    beta=None,  # Disable offset checking
)

seg = Segmenter(x=x, y=y, detector=detector)
result = seg.fit_detect()

print(f"Custom detector found {len(result.segments)} segments")
print(f"Algorithm metadata: {result.metadata}")

# %% [markdown]
# ## Performance Comparison
#
# Compare execution time for each detector:

# %%
import time

# Generate larger dataset for timing
large_x = np.arange(5000, dtype=np.float64)
large_y = np.cumsum(np.random.randn(5000)) + np.sin(large_x / 100) * 10

print("Performance on 5000 data points:")
print("-" * 40)

for detector_name in list_detectors():
    params = {"n": 50} if detector_name == "sliding_window" else {"max_segments": 20} if detector_name == "bottom_up" else {"penalty": 20}
    
    start = time.perf_counter()
    seg = Segmenter(x=large_x, y=large_y, detector=detector_name, detector_params=params)
    seg.calculate_segments()
    elapsed = time.perf_counter() - start
    
    print(f"{detector_name:<20}: {elapsed*1000:>8.2f} ms, {len(seg.segments)} segments")

# %% [markdown]
# ## Conclusion
#
# You've learned:
# - How to use different detection algorithms
# - How to compare their results visually and quantitatively
# - When to choose each algorithm
#
# **Recommendation**: Start with `sliding_window` for most cases. Use `pelt` for
# optimal results on large datasets, or `bottom_up` when you need a specific
# number of segments.
