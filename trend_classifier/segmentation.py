import warnings
from typing import Optional
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from trend_classifier.configuration import Config
from trend_classifier.models import Metrics
from trend_classifier.segment import Segment
from trend_classifier.types import FigSize


def error(a: float, b: float, metrics: Metrics = Metrics.ABSOLUTE_ERROR):
    """Calculate how much two parameters differ.

    Used e.g. to calculate how much the slopes of two micro-segments differ.

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
        return np.abs(a - b) / a
    if metrics == Metrics.ABSOLUTE_ERROR:
        return np.abs(a - b)


class Segmenter:
    def __init__(
        self,
        x: Optional[list[int]] = None,
        y: Optional[list[int]] = None,
        df=None,
        column: Optional[str] = "Adj Close",
        config: Optional[Config] = None,
        n: Optional[int] = None,
    ):

        # Handle configuration
        if config is None:
            # use default configuration if no configuration is provided
            self.config = Config()
            if n is not None:
                # override default N in configuration if N is provided
                self.config.N = n
        if config is not None and n is not None:
            # raise error
            raise ValueError("Provide either config or N, not both.")

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

        self.y_de_trended: Optional[list] = None

        self.segments: Optional[list[Segment]] = None
        self.slope: Optional[float] = None
        self.offset: Optional[float] = None
        self.slopes_std: Optional[float] = None
        self.offsets_std: Optional[float] = None

    def calculate_segments(self) -> list[Segment]:
        """Calculate segments with similar trend for the given timeserie.

        Calculates:
         - boundaries of segments
        - slopes and offsets of micro-segments

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

        segments = []
        s_start = 0
        slopes = []
        offsets = []
        off = int(N * overlap_ratio)
        if off == 0:
            print("Overlap ratio is too small, setting it to 1")
            print("N = ", N)
            print("overlap_ratio = ", overlap_ratio)
            off = 1
        for start in range(0, len(self.x) - N, off):  # noqa: FKA01
            fit = np.polyfit(
                x=self.x[start : start + N], y=self.y[start : start + N], deg=1
            )
            slopes.append(fit[0])
            offsets.append(fit[1])

            if prev_fit is not None:
                # asses if the slope is similar to the previous one
                prev_slope = float(prev_fit[0])
                this_slope = float(fit[0])
                r0 = error(prev_slope, this_slope, metrics=metrics_alpha)

                # asses if the offset is similar to the previous one
                prev_offset = float(prev_fit[1])
                this_offset = float(fit[1])
                r1 = error(prev_offset, this_offset, metrics=metrics_beta)

                is_slope_different = r0 >= alpha
                is_offset_different = r1 >= beta

                if is_slope_different or is_offset_different:
                    s_stop = determine_trend_end_point(off, start)
                    reason = self.describe_reason_for_new_segment(
                        is_offset_different, is_slope_different
                    )
                    segments.append(
                        Segment(
                            start=int(s_start),
                            stop=int(s_stop),
                            slopes=slopes,
                            offsets=offsets,
                            reason_for_new_segment=reason,
                        ),
                    )
                    s_start = s_stop + 1
                    slopes = []
                    offsets = []
            prev_fit = fit

        # add last segment
        segments.append(
            Segment(
                start=int(s_start),
                stop=int(len(self.x)),
                slopes=slopes,
                offsets=offsets,
            )
        )
        self.segments = segments

        # add extra information to the segments
        self._describe_segments()

        return segments

    @staticmethod
    def describe_reason_for_new_segment(
        is_offset_different: bool, is_slope_different: bool
    ) -> str:
        reason = "slope" if is_slope_different else "offset"
        if is_slope_different and is_offset_different:
            reason = "slope and offset"
        return reason

    def _describe_segments(self) -> None:
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
        plt.subplots(figsize=fig_size)
        plt.plot(self.x, self.y, color="#AAD", linestyle="solid")
        if isinstance(idx, int):
            idx = [idx]

        for i in idx:
            segment = self.segments[i]
            start = segment.start
            stop = segment.stop

            xx = self.x[start:stop]
            yy = self.y[start:stop]
            plt.plot(xx, yy, color=col, linestyle="-", linewidth=2)
            plt.scatter(xx[0], yy[0], color="k", s=10)
            plt.scatter(xx[-1], yy[-1], color="k", s=10)
            # add x- and y-axis labels
            plt.xlabel("time", fontsize=14)
            plt.ylabel("value", fontsize=14)
        plt.show()

    def plot_segments(self, fig_size: FigSize = (10, 5)) -> None:
        plt.subplots(figsize=fig_size)
        plt.plot(self.x, self.y, color="#AAD", linestyle="solid")
        for segment in self.segments:
            start = segment.start
            stop = segment.stop
            slopes = segment.slopes

            xx = self.x[start:stop]
            yy = self.y[start:stop]
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

            plt.vlines(start, min(self.y), max(self.y), "#CCC")  # noqa: FKA01
            plt.vlines(stop, min(self.y), max(self.y), "#CCC")  # noqa: FKA01

            plt.plot(
                self.x[start:stop],
                fit_fn(self.x[start:stop]),
                color=f"{col}",
                linestyle="--",
                linewidth=3,
            )
        # add x- and y-axis labels
        plt.xlabel("time", fontsize=14)
        plt.ylabel("value", fontsize=14)
        plt.show()

    def plot_detrended_signal(self, fig_size: FigSize = (10, 5)) -> None:
        plt.subplots(figsize=fig_size)
        plt.plot(self.x, self.y_de_trended, "b-")  # noqa: FKA01
        # add x- and y-axis labels
        plt.xlabel("time", fontsize=14)
        plt.ylabel("de-trended value", fontsize=14)
        plt.show()

    def calc_area_outside_trend(self) -> float:
        """Calculate area outside trend.

        Sum of absolute values of the points below/above the trend line.
        Normalized by the mean value of the signal.
        Normalized by the length of the signal.
        """
        return np.sum(np.abs(self.y_de_trended)) / np.mean(self.y) / len(self.y)


def determine_trend_end_point(off: int, start: int) -> int:
    # TODO: KS: 2022-09-06: proper calculation of the end of the segment
    s_stop = start + off / 2
    return int(s_stop)


# TODO: KS: 2022-09-06: Automatically determine parameters based on history:
#  see the difference e.g. between AAPL and BTC

# TODO: KS: 2022-09-06: Check how it works with different timeframes
#  (e.g. 1h, 4h, 1d)

# TODO: KS: 2022-09-06: improve quality on basic data V-shape or lambda-shape

# TODO: KS: 2022-09-07: add docstrings
