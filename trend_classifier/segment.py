"""Module with pydantic model of Segment and helper datastructure - SegmentList."""
import logging
from copy import deepcopy

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Segment(BaseModel):
    """Segment of a time series."""

    # ---- mandatory attributes
    start: int
    """Start index of the segment."""

    stop: int
    """Stop index of the segment."""

    slopes: list[float] = []
    """List of slopes of linear trends in windows in the segment."""

    offsets: list[float] = []
    """List of offsets of linear trends in windows in the segment."""

    starts: list[int] = []
    """List of start indices of windows in the segment."""

    # --- optional attributes with default values
    slope: float = 0
    """Slope of the segment."""

    offset: float = 0
    """Offset of the segments."""

    slopes_std: float = 0
    """Standard deviation of the slopes of linear trends in windows in the segment."""

    offsets_std: float = 0
    """Standard deviation of the offsets of linear trends in windows in the segment."""

    std: float = 0
    """Standard deviation of the samples in the segment with removed trend."""

    span: float = 0
    """Span of the values in the segment normalized by the mean value of the segment.
    Indicator if the volatility of the segment is high or low."""

    reason_for_new_segment: str = ""
    """Reason for creating a new segment (which criterion was violated)."""

    def __str__(self):
        return f"Segment(start={self.start}, stop={self.stop}, slope={self.slope:.4g})"

    def __repr__(self):
        s1 = f"Segment(start={self.start}, stop={self.stop}, slope={self.slope}, "
        s2 = f"offset={self.offset}, std={self.std}, span={self.span}, "
        s3 = f"reason_for_new_segment={self.reason_for_new_segment}, "
        s4 = f"slopes={self.slopes}, offsets={self.offsets}, slopes_std={self.slopes_std}, "
        s5 = f"offsets_std={self.offsets_std})"

        return s1 + s2 + s3 + s4 + s5

    def remove_outstanding_windows(self, n):
        new_slopes = deepcopy(self.slopes)
        new_offsets = deepcopy(self.offsets)
        new_starts = deepcopy(self.starts)
        for window_start in self.starts:
            n_windows = len(new_starts)
            window_end = window_start + n
            is_outstanding = window_end > self.stop

            if n_windows > 1 and is_outstanding:
                new_slopes.remove(self.slopes[self.starts.index(window_start)])
                new_offsets.remove(self.offsets[self.starts.index(window_start)])
                new_starts.remove(window_start)
                logger.debug(  # noqa: FKA01
                    "Removed window %f - %f.", window_start, window_start + n
                )
            else:
                logger.debug(  # noqa: FKA01
                    "Keeping window %f - %f.", window_start, window_start + n
                )
        self.slopes = new_slopes
        self.offsets = new_offsets
        self.starts = new_starts


class SegmentList(list):
    """List of segments. Each segment group samples with similar trend.

    New methods dedicated e.g. to processing od displaying list of segments
    can be added here.
    """

    def to_dataframe(self):
        """Convert segments to a pandas DataFrame.

        Returns:
            A pandas DataFrame.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is not installed. Install it with `pip install pandas`."
            )
        df = pd.DataFrame([s.__dict__ for s in self])
        # reorder columns
        df = df[
            [
                "start",
                "stop",
                "slope",
                "offset",
                "slopes_std",
                "offsets_std",
                "std",
                "span",
                "reason_for_new_segment",
                "slopes",
                "offsets",
            ]
        ]
        return df
