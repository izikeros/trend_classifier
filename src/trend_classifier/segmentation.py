from __future__ import annotations

import logging

import numpy as np

from trend_classifier.configuration import Config
from trend_classifier.models import Metrics
from trend_classifier.segment import Segment, SegmentList
from trend_classifier.visuals import (
    _plot_detrended_signal,
    _plot_segment,
    _plot_segment_with_trendlines_no_context,
    _plot_segments,
)

logger = logging.getLogger(__name__)
FigSize = tuple[float | int, float | int]


def calculate_error(
    a: float,
    b: float,
    metrics: Metrics = Metrics.ABSOLUTE_ERROR,
    min_denom: float = 1e-6,
) -> float:
    """Calculate how much two parameters differ.

    Used e.g. to calculate how much the slopes of linear trends in two windows differ.

    Args:
        a: First parameter.
        b: Second parameter.
        metrics: Metrics to use for the calculation.
        min_denom: Minimum denominator for relative error to avoid division by near-zero.

    Returns:
        Measure of difference between the two parameters.

    See Also:
        class `Metrics`
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Both 'a' and 'b' must be numeric values.")

    if metrics == Metrics.ABSOLUTE_ERROR:
        return abs(a - b)
    elif metrics == Metrics.RELATIVE_ABSOLUTE_ERROR:
        # Use min_denom to avoid explosion when a is near zero
        return abs(a - b) / max(abs(a), min_denom)
    else:
        raise ValueError(f"Unsupported metrics: {metrics}")


class Segmenter:
    """Class for segmenting a time series into segments with similar trend."""

    def __init__(
        self,
        x: list[int] | None = None,
        y: list[int] | None = None,
        df=None,
        column: str | None = "Adj Close",
        config: Config | None = None,
        n: int | None = None,
    ):
        """Initialize the segmenter.

        Args:
            x: List of x values.
            y: List of y values.
            df: Pandas DataFrame with time series.
            column: Name of the column with the time series.
            config: Configuration of the segmenter.
            n: Number of samples in a window.
        """
        self._handle_configuration(config, n)
        self._handle_input_data(column=column, df=df, x=x, y=y)
        self.y_de_trended: list | None = None
        self.segments: SegmentList[Segment] | None = None
        self.slope: float | None = None
        self.offset: float | None = None
        self.slopes_std: float | None = None
        self.offsets_std: float | None = None

    def _handle_configuration(self, config, n):
        # Handle configuration
        if config is None:
            # use default configuration if no configuration is provided
            self.config = Config()
            if n is not None:
                # override default N in configuration if N is provided
                self.config.N = n
        if config is not None:
            if n is not None:
                # raise error
                raise ValueError("Provide either config or N, not both.")
            self.config = config

    def _handle_input_data(self, column, df, x, y):
        # --- Handle input data
        # Accept numpy arrays, lists, or array-like objects
        if (
            x is not None
            and not isinstance(x, (list, np.ndarray))
            and not (hasattr(x, "__len__") and hasattr(x, "__getitem__"))
        ):
            raise TypeError(
                f"x must be a list or array-like, got {type(x)}. "
                "For pandas dataframe use 'df' keyword argument"
            )
        # error - no input data provided
        if x is None and y is None and df is None:
            raise ValueError("Provide timeseries data: either x and y or df.")
        # error - ambiguous input data provided - both x,y and df provided
        if x is not None and y is not None and df is not None:
            raise ValueError(
                "Provide timeseries data: either (x and y) or (df), not all."
            )
        # input data provided as x and y - convert to numpy arrays for efficiency
        if x is not None and y is not None:
            self.x = np.asarray(x, dtype=np.float64)
            self.y = np.asarray(y, dtype=np.float64)
        # input data provided as dataframe
        if df is not None:
            self.x = np.arange(len(df), dtype=np.float64)
            # Handle both single-level and multi-level column indices (yfinance compatibility)
            col_data = df[column]
            if hasattr(col_data, "squeeze"):
                col_data = col_data.squeeze()
            self.y = np.asarray(col_data.values, dtype=np.float64)

    def calculate_segments(self, progress_callback=None) -> list[Segment]:
        """Calculate segments with similar trend for the given timeserie.

        Calculates:
         - boundaries of segments
         - slopes and offsets of windows

        Args:
            progress_callback: Optional callback function(current, total) for progress reporting.
                Useful for long sequences. Called periodically during processing.

        Returns:
            List of detected Segment objects.

        Raises:
            ValueError: If x and y are not initialized or data is too short.
        """
        # check if initialized x and y
        if self.x is None or self.y is None:
            raise ValueError("Segmenter x and y must be initialized!")

        # Read data from config to short variables
        n = self.config.N

        # Validate data length
        data_len = len(self.x)
        if data_len < n:
            raise ValueError(
                f"Data length ({data_len}) must be at least window size N ({n}). "
                f"Reduce N or provide more data."
            )
        if data_len < 2 * n:
            logger.warning(
                f"Data length ({data_len}) is less than 2*N ({2 * n}). "
                "Results may be limited to a single segment."
            )
        overlap_ratio = self.config.overlap_ratio
        alpha = self.config.alpha
        beta = self.config.beta
        metrics_for_alpha = self.config.metrics_for_alpha
        metrics_for_beta = self.config.metrics_for_beta

        prev_fit = None

        segments = SegmentList()

        new_segment = {"s_start": 0, "slopes": [], "offsets": [], "starts": []}

        off = self._set_offset(n, overlap_ratio)

        # Calculate total iterations for progress reporting
        total_iterations = max(1, (len(self.x) - n) // off)

        for iteration, start in enumerate(range(0, len(self.x) - n, off)):
            # Report progress if callback provided (every 100 iterations)
            if progress_callback is not None and iteration % 100 == 0:
                progress_callback(iteration, total_iterations)

            end = start + n
            fit = np.polyfit(x=self.x[start:end], y=self.y[start:end], deg=1)
            new_segment["slopes"].append(fit[0])
            new_segment["offsets"].append(fit[1])
            new_segment["starts"].append(start)

            if prev_fit is not None:
                # asses if the slope is similar to the previous one
                prev_slope = float(prev_fit[0])
                this_slope = float(fit[0])
                r0 = calculate_error(prev_slope, this_slope, metrics=metrics_for_alpha)

                # asses if the offset is similar to the previous one
                prev_offset = float(prev_fit[1])
                this_offset = float(fit[1])
                r1 = calculate_error(prev_offset, this_offset, metrics=metrics_for_beta)

                is_slope_different = r0 >= alpha if alpha is not None else False
                is_offset_different = r1 >= beta if beta is not None else False
                new_segment["is_slope_different"] = is_slope_different
                new_segment["is_offset_different"] = is_offset_different

                new_segment = self._finish_segment_if_needed(
                    offset=off, new_segment=new_segment, segments=segments, start=start
                )
            prev_fit = fit

        # add last segment
        last_segment = Segment(
            start=int(new_segment["s_start"]),
            stop=len(self.x),
            slopes=new_segment["slopes"],
            offsets=new_segment["offsets"],
            starts=new_segment["starts"],
        )

        segments.append(last_segment)
        self.segments = segments

        # remove outstanding windows
        last_segment.remove_outstanding_windows(self.config.N)

        # add extra information to the segments
        self._describe_segments()

        return segments

    def _finish_segment_if_needed(self, offset, new_segment, segments, start):
        need_to_finish_segment = (
            new_segment["is_slope_different"] or new_segment["is_offset_different"]
        )
        if need_to_finish_segment:
            s_stop = _determine_trend_end_point(offset, start)
            reason = self.describe_reason_for_new_segment(
                new_segment["is_offset_different"], new_segment["is_slope_different"]
            )

            segment = Segment(
                start=int(new_segment["s_start"]),
                stop=int(s_stop),
                slopes=new_segment["slopes"],
                offsets=new_segment["offsets"],
                starts=new_segment["starts"],
                reason_for_new_segment=reason,
            )

            # remove outstanding windows
            segment.remove_outstanding_windows(self.config.N)

            segments.append(segment)
            new_segment["s_start"] = s_stop + 1
            new_segment["slopes"] = []
            new_segment["offsets"] = []
            new_segment["starts"] = []
        return new_segment

    @staticmethod
    def _set_offset(n, overlap_ratio):
        offset = int(n * overlap_ratio)
        if offset == 0:
            print("Overlap ratio is too small, setting it to 1")
            print("N = ", n)
            print("overlap_ratio = ", overlap_ratio)
            offset = 1
        return offset

    @staticmethod
    def describe_reason_for_new_segment(
        is_offset_different: bool, is_slope_different: bool
    ) -> str:
        """Describe reason for creating a new segment.

        Used for better explainability of the operation and decision-making.
        """
        reason = "slope" if is_slope_different else "offset"
        if is_slope_different and is_offset_different:
            reason = "slope and offset"
        return reason

    def _describe_segments(self) -> None:
        """Add extra information about the segments."""
        y_norm = []
        for idx, segment in enumerate(self.segments):
            start = segment.start
            stop = segment.stop

            # x and y for the segment
            xx = self.x[start : stop + 1]
            yy = self.y[start : stop + 1]

            # Skip segments with insufficient data points
            if len(xx) < 2:
                y_norm.extend([0.0])
                self.segments[idx].std = 0.0
                self.segments[idx].span = 0.0
                self.segments[idx].slope = 0.0
                self.segments[idx].offset = 0.0
                self.segments[idx].slopes_std = 0.0
                self.segments[idx].offsets_std = 0.0
                continue

            # trend calculation
            fit = np.polyfit(x=xx, y=yy, deg=1)
            fit_fn = np.poly1d(fit)

            # calculate point for the trend line
            yt = np.array(fit_fn(xx))

            ydt = np.array(yy) - yt

            # calculate standard deviation of the values with removed trend
            # Use ddof=0 to avoid warning when array has only 1 element
            s = np.std(ydt, ddof=0) if len(ydt) > 0 else 0.0

            # calculate span of the values in the segment normalized by
            # the mean value of the segment
            mean_yy = np.mean(yy)
            if mean_yy != 0:
                span = 1000 * (np.max(ydt) - np.min(ydt)) / abs(mean_yy)
            else:
                span = 0.0

            # store de-trended values
            y_norm.extend(ydt)

            # store volatility measures for the segment
            self.segments[idx].std = float(s)
            self.segments[idx].span = float(span)
            self.y_de_trended = y_norm
            self.segments[idx].slope = float(fit[0])
            self.segments[idx].offset = float(fit[1])

            # Calculate std of slopes/offsets, handle single-element arrays
            slopes = self.segments[idx].slopes
            offsets = self.segments[idx].offsets
            self.segments[idx].slopes_std = (
                float(np.std(slopes, ddof=0)) if len(slopes) > 0 else 0.0
            )
            self.segments[idx].offsets_std = (
                float(np.std(offsets, ddof=0)) if len(offsets) > 0 else 0.0
            )

    def plot_segment(
        self,
        idx: list[int] | int,
        col: str = "red",
        fig_size: FigSize = (10, 5),
    ) -> None:
        """Plot segment with given index or multiple segments with given indices.

        Args:
            idx: index of the segment or list of indices of segments
            col: color of the segment
            fig_size: size of the figure
        """
        _plot_segment(obj=self, idx=idx, col=col, fig_size=fig_size)

    def plot_segment_with_trendlines_no_context(
        self,
        idx: int,
        fig_size: FigSize = (10, 5),
    ) -> None:
        """Plot segment with given index.

        Args:
            idx: index of the segment or list of indices of segments
            fig_size: size of the figure
        """
        _plot_segment_with_trendlines_no_context(obj=self, idx=idx, fig_size=fig_size)

    def plot_segments(self, fig_size: FigSize = (8, 4)) -> None:
        """Plot all segments and linear trend lines.

        Args:
            fig_size: size of the figure e.g. (8, 4)
        """
        _plot_segments(self, fig_size)

    def plot_detrended_signal(self, fig_size: FigSize = (10, 5)) -> None:
        """Plot de-trended signal.

        Args:
            fig_size: size of the figure
        """
        _plot_detrended_signal(
            x=self.x, y_de_trended=self.y_de_trended, fig_size=fig_size
        )

    def calc_area_outside_trend(self) -> float:
        """Calculate area outside trend.

        Sum of absolute values of the points below/above the trend line.
        Normalized by the mean value of the signal.
        Normalized by the length of the signal.

        Returns:
            area outside trend

        """
        return np.sum(np.abs(self.y_de_trended)) / np.mean(self.y) / len(self.y)


def _determine_trend_end_point(off: int, start: int) -> int:
    """Determine end point of the trend.

    Args:
        off: offset of the window
        start: start point of the trend
    """
    # TODO: KS: 2022-09-06: proper calculation of the end of the segment
    s_stop = start + off / 2
    return int(s_stop)


# TODO: KS: 2022-09-06: Automatically determine parameters based on history:
#  see the difference e.g. between AAPL and BTC

# TODO: KS: 2022-09-06: Check how it works with different timeframes
#  (e.g. 1h, 4h, 1d)

# TODO: KS: 2022-09-06: improve quality on basic data V-shape or lambda-shape

# TODO: KS: 2022-09-07: add docstrings
