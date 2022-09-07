from pydantic import BaseModel


class Segment(BaseModel):
    """Segment of a time series.

    Attributes:
        self.start: Start index of the segment.
        self.stop: End index of the segment.
        self.slope: Slope of the segment.
        self.offset: Offset of the segment.
        self.reason_for_new_segment: Reason for creating a new segment (which criterion was violated).
        self.slopes: List of slopes of the micro-segments.
        self.offsets: Offsets of the micro-segments.
        self.slopes_std: Standard deviation of the slopes.
        self.offsets_std: Standard deviation of the offsets.
        self.span: span of the values in the segment normalized by the mean value of the
                            segment. Indicator if the volatility of the segment is high or low.
    """

    start: int
    stop: int
    slope: float = 0
    offset: float = 0
    slopes: list[float]
    offsets: list[float]
    slopes_std: float = 0
    offsets_std: float = 0
    std: float = 0
    span: float = 0
    reason_for_new_segment: str = ""

    def __str__(self):
        return f"Segment(start={self.start}, stop={self.stop}, slope={self.slope:.4g})"

    def __repr__(self):
        s1 = f"Segment(start={self.start}, stop={self.stop}, slope={self.slope}, "
        s2 = f"offset={self.offset}, std={self.std}, span={self.span}, "
        s3 = f"reason_for_new_segment={self.reason_for_new_segment}, "
        s4 = f"slopes={self.slopes}, offsets={self.offsets}, slopes_std={self.slopes_std}, "
        s5 = f"offsets_std={self.offsets_std})"

        return s1 + s2 + s3 + s4 + s5
