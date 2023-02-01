import warnings
from typing import Optional
from typing import Union

import numpy as np
from trend_classifier.configuration import Config
from trend_classifier.models import Metrics
from trend_classifier.segment import Segment
from trend_classifier.segment import SegmentList
from trend_classifier.types import FigSize
from trend_classifier.visuals import _plot_detrended_signal
from trend_classifier.visuals import _plot_segment
from trend_classifier.visuals import _plot_segment_with_trendlines_no_context
from trend_classifier.visuals import _plot_segments


def _error(a: float, b: float, metrics: Metrics = Metrics.ABSOLUTE_ERROR) -> float:
    """Calculate how much two parameters differ.

    Used e.g. to calculate how much the slopes of linear trends in two windows differ.

    Args:
        a: First parameter.
        b: Second parameter.
        metrics: Metrics to use for the calculation.

    Returns:
        Measure of difference between the two parameters.

    See Also:
        class `Metrics`

    """
    if metrics == Metrics.RELATIVE_ABSOLUTE_ERROR:
        return abs(a - b) / abs(a)
    if metrics == Metrics.ABSOLUTE_ERROR:
        return abs(a - b)


class Segmenter:
    """Class for segmenting a time series into segments with similar trend."""

    def __init__(
        self,
        x: Optional[list[int]] = None,
        y: Optional[list[int]] = None,
        df=None,
        column: Optional[str] = "Adj Close",
        config: Optional[Config] = None,
        n: Optional[int] = None,
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
        self.y_de_trended: Optional[list] = None
        self.segments: Optional[SegmentList[Segment]] = None
        self.slope: Optional[float] = None
        self.offset: Optional[float] = None
        self.slopes_std: Optional[float] = None
        self.offsets_std: Optional[float] = None

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
        # error - most likely pandas dataframe as argument instead of kwarg
        if x is not None and not isinstance(x, list):
            # TODO: KS: 2022-09-07: accept also numpy array, ndarray,np.matrix, pd.Series
            raise TypeError(
                "x must be a list, got {}. For pandas dataframe use 'df' keyword argument".format(
                    type(x)
                )
            )
        # error - no input data provided
        if x is None and y is None and df is None:
            raise ValueError("Provide timeseries data: either x and y or df.")
        # error - ambiguous input data provided - both x,y and df provided
        if x is not None and y is not None and df is not None:
            raise ValueError(
                "Provide timeseries data: either (x and y) or (df), not all."
            )
        # input data provided as x and y
        if x is not None and y is not None:
            self.x = x
            self.y = y
        # make warning if column provided but not dataframe
        if df is None and column is not None:
            warnings.warn("No dataframe provided, column argument will be ignored.")
        # input data provided as dataframe
        if df is not None:
            self.x = list(range(0, len(df.index.tolist()), 1))  # noqa: FKA01
            self.y = df[column].tolist()

    def calculate_segments(self) -> list[Segment]:
        """Calculate segments with similar trend for the given timeserie.

        Calculates:
         - boundaries of segments
        - slopes and offsets of windows

        """
        # check if initialized x and y
        if self.x is None or self.y is None:
            raise ValueError("Segmenter x and y must be initialized!")
        # Read data from config to short variables
        N = self.config.N
        overlap_ratio = self.config.overlap_ratio
        alpha = self.config.alpha
        beta = self.config.beta
        metrics_alpha = self.config.metrics_alpha
        metrics_beta = self.config.metrics_beta

        prev_fit = None

        segments = SegmentList()

        new_segment = {"s_start": 0, "slopes": [], "offsets": [], "starts": []}

        off = self._set_offset(N, overlap_ratio)

        for start in range(0, len(self.x) - N, off):  # noqa: FKA01
            end = start + N
            fit = np.polyfit(x=self.x[start:end], y=self.y[start:end], deg=1)
            new_segment["slopes"].append(fit[0])
            new_segment["offsets"].append(fit[1])
            new_segment["starts"].append(start)

            if prev_fit is not None:
                # asses if the slope is similar to the previous one
                prev_slope = float(prev_fit[0])
                this_slope = float(fit[0])
                r0 = _error(prev_slope, this_slope, metrics=metrics_alpha)

                # asses if the offset is similar to the previous one
                prev_offset = float(prev_fit[1])
                this_offset = float(fit[1])
                r1 = _error(prev_offset, this_offset, metrics=metrics_beta)

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
            stop=int(len(self.x)),
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

            # trend calculation
            fit = np.polyfit(x=xx, y=yy, deg=1)
            fit_fn = np.poly1d(fit)

            # calculate point for the trend line
            yt = np.array(fit_fn(xx))

            # FIXME: KS: 2022-09-02: Normalize (as below?)
            # normalize each point in yy by value of corresponding point in yt,
            # store results in ydtn
            # ydtn = yy / yt

            ydt = np.array(yy) - yt

            # calculate standard deviation of the values with removed trend
            s = np.std(ydt)

            # calculate span of the values in the segment normalized by
            # the mean value of the segment
            span = 1000 * (np.max(ydt) - np.min(ydt)) // np.mean(yy)

            # store de-trended values
            y_norm.extend(ydt)

            # store volatility measures for the segment
            self.segments[idx].std = s
            self.segments[idx].span = span
            self.y_de_trended = y_norm
            self.segments[idx].slope = fit[0]
            self.segments[idx].offset = fit[1]
            self.segments[idx].slopes_std = np.std(self.segments[idx].slopes)
            self.segments[idx].offsets_std = np.std(self.segments[idx].offsets)

    def plot_segment(
        self,
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
