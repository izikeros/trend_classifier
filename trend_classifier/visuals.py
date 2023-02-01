from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from trend_classifier.types import FigSize


def _plot_detrended_signal(
    x: list[float], y_de_trended: list[float], fig_size: FigSize = (10, 5)
) -> None:
    """Plot de-trended signal.

    Args:
        x: x values.
        y_de_trended (np.ndarray): de-trended y values.
        fig_size: size of the figure
    """
    plt.subplots(figsize=fig_size)
    plt.plot(x, y_de_trended, "b-")  # noqa: FKA01
    # add x- and y-axis labels
    plt.xlabel("time", fontsize=14)
    plt.ylabel("de-trended value", fontsize=14)
    plt.show()


def _plot_segments(obj, fig_size: FigSize = (8, 4)) -> None:
    """Plot all segments and linear trend lines.

    Args:
        fig_size: size of the figure e.g. (8, 4)
    """
    plt.subplots(figsize=fig_size)
    plt.plot(obj.x, obj.y, color="#AAD", linestyle="solid")
    for segment in obj.segments:
        start = segment.start
        stop = segment.stop
        slopes = segment.slopes

        xx = obj.x[start:stop]
        yy = obj.y[start:stop]
        fit = np.polyfit(x=xx, y=yy, deg=1)
        fit_fn = np.poly1d(fit)

        all_positive_slopes = all([v >= 0 for v in slopes])
        all_negatives_slopes = all([v < 0 for v in slopes])

        if fit[0] >= 0 and all_positive_slopes:
            col = "g"
        elif fit[0] < 0 and all_negatives_slopes:
            col = "r"
        else:
            col = "#A66"

        plt.vlines(start, min(obj.y), max(obj.y), "#CCC")  # noqa: FKA01
        plt.vlines(stop, min(obj.y), max(obj.y), "#CCC")  # noqa: FKA01

        plt.plot(
            obj.x[start:stop],
            fit_fn(obj.x[start:stop]),
            color=f"{col}",
            linestyle="--",
            linewidth=3,
        )
    # add x- and y-axis labels
    plt.xlabel("time", fontsize=14)
    plt.ylabel("value", fontsize=14)
    plt.show()


def _plot_segment(
    obj,
    idx: Union[list[int], int],
    col: str = "red",
    fig_size: FigSize = (10, 5),
) -> None:
    """Plot segment with given index or multiple segments with given indices.

    Args:
        idx: index of the segment or list of indices of segments
        col: color of the segment
        fig_size: size of the figure
    """
    plt.subplots(figsize=fig_size)
    plt.plot(obj.x, obj.y, color="#AAD", linestyle="solid")
    if isinstance(idx, int):
        idx = [idx]

    for i in idx:
        segment = obj.segments[i]
        start = segment.start
        stop = segment.stop

        xx = obj.x[start:stop]
        yy = obj.y[start:stop]
        plt.plot(xx, yy, color=col, linestyle="-", linewidth=2)
        plt.scatter(xx[0], yy[0], color="k", s=10)
        plt.scatter(xx[-1], yy[-1], color="k", s=10)
        # add x- and y-axis labels
        plt.xlabel("time", fontsize=14)
        plt.ylabel("value", fontsize=14)
    plt.show()


def _plot_segment_with_trendlines_no_context(
    obj,
    idx: Union[list[int], int],
    signal_color: str = "#ccc",
    fig_size: FigSize = (10, 5),
) -> None:
    """Plot segment with given index or multiple segments with given indices.

    Args:
        idx: index of the segment or list of indices of segments
        fig_size: size of the figure
    """
    segment = obj.segments[idx]
    segment_start = segment.start
    segment_stop = segment.stop

    xx = obj.x[segment_start:segment_stop]
    yy = obj.y[segment_start:segment_stop]

    plt.subplots(figsize=fig_size)
    plt.plot(xx, yy, color=signal_color, linestyle="-", linewidth=2)

    for i, (start, slope, offset) in enumerate(
        zip(segment.starts, segment.slopes, segment.offsets)  # noqa: FKA01
    ):
        x = obj.x[start : obj.config.N + start]
        y = slope * np.array(x) + offset
        plt.plot(
            x,
            y,
            linestyle="--",
            linewidth=2,
            label=f"trendline {i}",
        )
    # add x- and y-axis labels
    plt.xlabel("time", fontsize=14)
    plt.ylabel("value", fontsize=14)
    plt.legend()
    plt.show()
