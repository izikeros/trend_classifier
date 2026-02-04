"""PELT (Pruned Exact Linear Time) detector using ruptures library."""

from __future__ import annotations

import numpy as np

from trend_classifier.detectors.base import BaseDetector, DetectionResult
from trend_classifier.segment import Segment, SegmentList


class PELTDetector(BaseDetector):
    """PELT change point detector using the ruptures library.

    PELT (Pruned Exact Linear Time) is an efficient algorithm for detecting
    multiple change points in a signal. It finds the optimal segmentation
    by minimizing a cost function with a penalty for each change point.

    This detector requires the `ruptures` library to be installed:
        pip install ruptures

    Args:
        model: Cost model for segment evaluation. Options:
            - "l2": Least squares (default, good for mean shifts)
            - "l1": Least absolute deviation (robust to outliers)
            - "rbf": Kernel-based (detects distribution changes)
            - "linear": Linear regression model
        penalty: Penalty value for adding change points. Higher values
            result in fewer segments. If None, uses BIC-derived penalty.
        min_size: Minimum segment length.
        jump: Subsample factor for change point candidates.

    Example:
        >>> detector = PELTDetector(model="l2", penalty=3)
        >>> result = detector.fit_detect(x, y)
        >>> print(f"Found {len(result.segments)} segments")

    Note:
        For financial time series, try:
        - model="linear" with penalty=1-5 for trend detection
        - model="l2" with penalty=5-20 for level shifts
    """

    name = "pelt"

    def __init__(
        self,
        model: str = "l2",
        penalty: float | None = None,
        min_size: int = 2,
        jump: int = 1,
    ):
        self._check_ruptures_available()

        self.model = model
        self.penalty = penalty
        self.min_size = min_size
        self.jump = jump

        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._algo = None

    @staticmethod
    def _check_ruptures_available() -> None:
        """Check if ruptures library is installed."""
        try:
            import ruptures  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PELTDetector requires the 'ruptures' library. "
                "Install it with: pip install ruptures"
            ) from e

    def fit(self, x: np.ndarray, y: np.ndarray) -> PELTDetector:
        """Fit the detector to data.

        Args:
            x: Array of x values (indices).
            y: Array of y values (signal).

        Returns:
            Self for method chaining.
        """
        import ruptures as rpt

        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)

        # Reshape for ruptures (expects 2D array)
        signal = self._y.reshape(-1, 1)

        # Create and fit the PELT algorithm
        self._algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump)
        self._algo.fit(signal)

        return self

    def detect(self) -> DetectionResult:
        """Detect change points and create segments.

        Returns:
            DetectionResult with segments and breakpoints.

        Raises:
            RuntimeError: If called before fit().
        """
        self._validate_fitted()

        # Calculate penalty if not provided
        penalty = self.penalty
        if penalty is None:
            # BIC-like penalty based on data length
            penalty = np.log(len(self._y)) * 2

        # Get change points
        breakpoints = self._algo.predict(pen=penalty)

        # ruptures returns breakpoints including the end, remove the last one
        if breakpoints and breakpoints[-1] == len(self._y):
            breakpoints = breakpoints[:-1]

        # Create segments from breakpoints
        segments = self._create_segments_from_breakpoints(breakpoints)

        return DetectionResult(
            segments=segments,
            breakpoints=breakpoints,
            metadata={
                "algorithm": self.name,
                "model": self.model,
                "penalty": penalty,
                "min_size": self.min_size,
            },
        )

    def _create_segments_from_breakpoints(self, breakpoints: list[int]) -> SegmentList:
        """Convert breakpoints to Segment objects with statistics."""
        segments = SegmentList()

        # Add start point
        all_points = [0, *list(breakpoints), len(self._y)]

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
            reason_for_new_segment="pelt",
        )
