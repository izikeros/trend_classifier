# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: freqtrade
#     language: python
#     name: freqtrade
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pprint import pprint

# %% [markdown]
# # Download data from yahoo finance

# %%
symbol = "BTC-USD"
symbol = "AAPL"
symbol = "XRP-USD"
df = yf.download(symbol, start="2018-09-15", end="2022-08-15", progress=False)

# %% [markdown]
# # Cluster points into segments with similar trend

# %%
column = "Adj Close"

x = list(range(0, len(df.index.tolist()), 1))
y = df[column].tolist()
ymin, ymax = (
    np.min(y),
    np.max(y),
)

plt.subplots(figsize=(10, 8))
plt.plot(x, df[column], "#CCC")


class config(BaseModel):
    N = 40
    off = int(N / 3)
    alpha = 2
    beta = 2

prev_fit = None

segments = []
s_start = 0
slopes = []
for start in range(0, len(df) - N, off):
    fit = np.polyfit(x[start : start + N], y[start : start + N], 1)
    slopes.append(fit[0])
    if prev_fit is not None:
        r0 = abs((prev_fit[0] - fit[0]) / (fit[0]))
        r1 = abs((prev_fit[1] - fit[1]) / (fit[1]))

        if r0 >= alpha or r1 >= beta:
            s_stop = start + off / 2
            plt.vlines(s_stop, ymin, ymax, "#CCC")
            segments.append(
                (
                    int(s_start),
                    int(s_stop),
                    slopes,
                )
            )
            s_start = s_stop + 1
            slopes = []

    fit_fn = np.poly1d(fit)
    #     plt.plot(x[start : start + N], fit_fn(x[start : start + N]), "--y")
    prev_fit = fit
segments.append(
    (
        int(s_start),
        int(len(df)),
        slopes,
    )
)

# Calculate trend for the segment
for start, stop, slopes in segments:
    xx = x[start:stop]
    yy = y[start:stop]
    fit = np.polyfit(xx, yy, 1)
    fit_fn = np.poly1d(fit)

    all_positive_slopes = all([v >= 0 for v in slopes])
    all_negatives_slopes = all([v < 0 for v in slopes])

    if fit[0] >= 0 and all_positive_slopes:
        col = "g"
    elif fit[0] < 0 and all_negatives_slopes:
        col = "r"
    else:
        col = "y"
    plt.plot(x[start:stop], fit_fn(x[start:stop]), f"--{col}", linewidth=3)
plt.show()

# %% [markdown]
# # Detrend segments

# %%
plt.subplots(figsize=(10, 5))
xn = []
all_std = []
all_span = []
for idx, (start, stop, slopes) in enumerate(segments):
    xx = x[start:stop]
    yy = y[start:stop]
    fit = np.polyfit(xx, yy, 1)
    fit_fn = np.poly1d(fit)
    yt = np.array(fit_fn(xx))
    ydt = np.array(yy) - yt
    s = np.std(ydt)
    span = 1000 * (np.max(ydt) - np.min(ydt)) // np.mean(yy)
    xn.extend(ydt)
    print(f"std: {s:.3f}, rng: {span:.3f}")
    all_std.append(s)
    all_span.append(span)

ydt_min, ydt_max = (
    np.min(ydt),
    np.max(ydt),
)
for start, stop, _ in segments:
    plt.vlines(stop, ydt_min, ydt_max, "#CCC")
    print()

_ = plt.plot(xn)

# %%
plt.scatter(all_span, all_std)

# %%
