"""Bottom-up segmentation detector."""

from __future__ import annotations

import numpy as np

from trend_classifier.detectors.base import BaseDetector, DetectionResult
from trend_classifier.segment import Segment, SegmentList


class BottomUpDetector(BaseDetector):
    """Bottom-up merge segmentation detector.

    This algorithm starts with many small segments and iteratively merges
    adjacent segments with the smallest merge cost until reaching the
    desired number of segments or a cost threshold.

    This approach is good for noisy data as it considers the full signal
    before making decisions, unlike sliding window methods.

    Args:
        max_segments: Maximum number of segments to produce.
        merge_cost_threshold: Stop merging when cost exceeds this value.
            If None, uses max_segments to determine stopping point.
        initial_segment_size: Size of initial segments before merging.

    Example:
        >>> detector = BottomUpDetector(max_segments=10)
        >>> result = detector.fit_detect(x, y)
        >>> print(f"Found {len(result.segments)} segments")
    """

    name = "bottom_up"

    def __init__(
        self,
        max_segments: int = 10,
        merge_cost_threshold: float | None = None,
        initial_segment_size: int = 5,
    ):
        self.max_segments = max_segments
        self.merge_cost_threshold = merge_cost_threshold
        self.initial_segment_size = initial_segment_size

        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> BottomUpDetector:
        """Fit the detector to data.

        Args:
            x: Array of x values (indices).
            y: Array of y values (signal).

        Returns:
            Self for method chaining.
        """
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        return self

    def detect(self) -> DetectionResult:
        """Detect segments using bottom-up merging.

        Returns:
            DetectionResult with segments and breakpoints.

        Raises:
            RuntimeError: If called before fit().
        """
        self._validate_fitted()

        n = len(self._y)

        # Create initial fine-grained segments
        breakpoints = list(
            range(self.initial_segment_size, n, self.initial_segment_size)
        )

        # Iteratively merge segments with lowest cost
        while len(breakpoints) >= self.max_segments:
            if len(breakpoints) == 0:
                break

            # Find the merge with minimum cost
            min_cost = float("inf")
            min_idx = 0

            for i in range(len(breakpoints)):
                cost = self._merge_cost(breakpoints, i)
                if cost < min_cost:
                    min_cost = cost
                    min_idx = i

            # Check threshold
            if (
                self.merge_cost_threshold is not None
                and min_cost > self.merge_cost_threshold
            ):
                break

            # Perform merge (remove the breakpoint)
            breakpoints.pop(min_idx)

        # Create segments from final breakpoints
        segments = self._create_segments_from_breakpoints(breakpoints)

        return DetectionResult(
            segments=segments,
            breakpoints=breakpoints,
            metadata={
                "algorithm": self.name,
                "max_segments": self.max_segments,
                "initial_segment_size": self.initial_segment_size,
            },
        )

    def _merge_cost(self, breakpoints: list[int], idx: int) -> float:
        """Calculate cost of merging segments around breakpoint at idx.

        The cost is the increase in total squared error when merging
        two adjacent segments into one.
        """
        # Get segment boundaries
        start = 0 if idx == 0 else breakpoints[idx - 1]
        middle = breakpoints[idx]
        end = len(self._y) if idx == len(breakpoints) - 1 else breakpoints[idx + 1]

        # Calculate error for separate segments
        error_left = self._segment_error(start, middle)
        error_right = self._segment_error(middle, end)

        # Calculate error for merged segment
        error_merged = self._segment_error(start, end)

        # Cost is the increase in error
        return error_merged - (error_left + error_right)

    def _segment_error(self, start: int, stop: int) -> float:
        """Calculate squared error of linear fit for segment."""
        if stop <= start:
            return 0.0

        xx = self._x[start:stop]
        yy = self._y[start:stop]

        if len(xx) < 2:
            return 0.0

        # Fit linear trend
        fit = np.polyfit(xx, yy, deg=1)
        fit_fn = np.poly1d(fit)
        y_pred = fit_fn(xx)

        # Sum of squared errors
        return float(np.sum((yy - y_pred) ** 2))

    def _create_segments_from_breakpoints(self, breakpoints: list[int]) -> SegmentList:
        """Convert breakpoints to Segment objects."""
        segments = SegmentList()

        all_points = [0, *sorted(breakpoints), len(self._y)]

        for i in range(len(all_points) - 1):
            start = all_points[i]
            stop = all_points[i + 1]

            segment = self._create_segment_with_stats(start, stop)
            segments.append(segment)

        return segments

    def _create_segment_with_stats(self, start: int, stop: int) -> Segment:
        """Create a segment with computed statistics."""
        xx = self._x[start:stop]
        yy = self._y[start:stop]

        if len(xx) < 2:
            return Segment(
                start=start,
                stop=stop,
                slope=0.0,
                offset=float(yy[0]) if len(yy) > 0 else 0.0,
            )

        # Fit linear trend
        fit = np.polyfit(xx, yy, deg=1)
        slope, offset = float(fit[0]), float(fit[1])

        # Calculate detrended statistics
        fit_fn = np.poly1d(fit)
        y_trend = fit_fn(xx)
        y_detrended = yy - y_trend

        std = float(np.std(y_detrended, ddof=0)) if len(y_detrended) > 0 else 0.0

        mean_yy = np.mean(yy)
        if mean_yy != 0:
            span = float(
                1000 * (np.max(y_detrended) - np.min(y_detrended)) / abs(mean_yy)
            )
        else:
            span = 0.0

        return Segment(
            start=start,
            stop=stop,
            slope=slope,
            offset=offset,
            std=std,
            span=span,
            reason_for_new_segment="bottom_up",
        )
