"""Sliding window detector with linear regression."""

from __future__ import annotations

import logging

import numpy as np

from trend_classifier.configuration import Config
from trend_classifier.detectors.base import BaseDetector, DetectionResult
from trend_classifier.metrics import calculate_error
from trend_classifier.models import Metrics
from trend_classifier.segment import Segment, SegmentList

logger = logging.getLogger(__name__)


class SlidingWindowDetector(BaseDetector):
    """Sliding window trend detector using linear regression.

    This is the original algorithm from trend_classifier. It slides a window
    across the time series, fits a linear trend in each window, and detects
    segment boundaries when slope or offset changes exceed thresholds.

    Args:
        n: Window size (number of samples per window).
        overlap_ratio: Overlap between adjacent windows (0-1).
        alpha: Threshold for slope change detection. None to disable.
        beta: Threshold for offset change detection. None to disable.
        metrics_for_alpha: Error metric for slope comparison.
        metrics_for_beta: Error metric for offset comparison.

    Example:
        >>> detector = SlidingWindowDetector(n=40, alpha=2.0)
        >>> result = detector.fit_detect(x, y)
        >>> print(f"Found {len(result.segments)} segments")
    """

    name = "sliding_window"

    def __init__(
        self,
        n: int = 60,
        overlap_ratio: float = 0.33,
        alpha: float | None = 2.0,
        beta: float | None = 2.0,
        metrics_for_alpha: Metrics = Metrics.RELATIVE_ABSOLUTE_ERROR,
        metrics_for_beta: Metrics = Metrics.RELATIVE_ABSOLUTE_ERROR,
    ):
        self.n = n
        self.overlap_ratio = overlap_ratio
        self.alpha = alpha
        self.beta = beta
        self.metrics_for_alpha = metrics_for_alpha
        self.metrics_for_beta = metrics_for_beta

        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: Config) -> SlidingWindowDetector:
        """Create detector from a Config object.

        Args:
            config: Configuration object with detector parameters.

        Returns:
            Configured SlidingWindowDetector instance.
        """
        return cls(
            n=config.N,
            overlap_ratio=config.overlap_ratio,
            alpha=config.alpha,
            beta=config.beta,
            metrics_for_alpha=config.metrics_for_alpha,
            metrics_for_beta=config.metrics_for_beta,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> SlidingWindowDetector:
        """Fit the detector to data.

        Args:
            x: Array of x values (indices).
            y: Array of y values (signal).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If data is too short for window size.
        """
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)

        data_len = len(self._x)
        if data_len < self.n:
            raise ValueError(
                f"Data length ({data_len}) must be at least window size N ({self.n}). "
                f"Reduce N or provide more data."
            )
        if data_len < 2 * self.n:
            logger.warning(
                f"Data length ({data_len}) is less than 2*N ({2 * self.n}). "
                "Results may be limited to a single segment."
            )

        return self

    def detect(self, progress_callback=None) -> DetectionResult:
        """Detect trend segments using sliding window analysis.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            DetectionResult with segments and breakpoints.

        Raises:
            RuntimeError: If called before fit().
        """
        self._validate_fitted()

        segments = SegmentList()
        breakpoints = []
        new_segment = {"s_start": 0, "slopes": [], "offsets": [], "starts": []}

        offset = self._calculate_offset()
        prev_fit = None

        total_iterations = max(1, (len(self._x) - self.n) // offset)

        for iteration, start in enumerate(range(0, len(self._x) - self.n, offset)):
            if progress_callback is not None and iteration % 100 == 0:
                progress_callback(iteration, total_iterations)

            end = start + self.n
            fit = np.polyfit(self._x[start:end], self._y[start:end], deg=1)
            new_segment["slopes"].append(fit[0])
            new_segment["offsets"].append(fit[1])
            new_segment["starts"].append(start)

            if prev_fit is not None:
                is_slope_different = self._check_slope_change(prev_fit, fit)
                is_offset_different = self._check_offset_change(prev_fit, fit)

                if is_slope_different or is_offset_different:
                    s_stop = self._determine_boundary(start, offset)
                    breakpoints.append(s_stop)
                    reason = self._describe_reason(
                        is_slope_different, is_offset_different
                    )

                    segment = self._finalize_segment(new_segment, s_stop, reason)
                    segments.append(segment)

                    new_segment = {
                        "s_start": s_stop + 1,
                        "slopes": [],
                        "offsets": [],
                        "starts": [],
                    }

            prev_fit = fit

        # Add final segment
        last_segment = self._finalize_segment(new_segment, len(self._x))
        segments.append(last_segment)

        # Compute detailed statistics for all segments
        self._compute_segment_statistics(segments)

        return DetectionResult(
            segments=segments,
            breakpoints=breakpoints,
            metadata={
                "algorithm": self.name,
                "n": self.n,
                "overlap_ratio": self.overlap_ratio,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        )

    def _calculate_offset(self) -> int:
        """Calculate step size between windows."""
        offset = int(self.n * self.overlap_ratio)
        if offset == 0:
            logger.warning(
                f"Overlap ratio {self.overlap_ratio} too small for N={self.n}, using offset=1"
            )
            offset = 1
        return offset

    def _check_slope_change(self, prev_fit, curr_fit) -> bool:
        """Check if slope changed significantly."""
        if self.alpha is None:
            return False
        prev_slope = float(prev_fit[0])
        curr_slope = float(curr_fit[0])
        error = calculate_error(prev_slope, curr_slope, self.metrics_for_alpha)
        return error >= self.alpha

    def _check_offset_change(self, prev_fit, curr_fit) -> bool:
        """Check if offset changed significantly."""
        if self.beta is None:
            return False
        prev_offset = float(prev_fit[1])
        curr_offset = float(curr_fit[1])
        error = calculate_error(prev_offset, curr_offset, self.metrics_for_beta)
        return error >= self.beta

    def _determine_boundary(self, start: int, offset: int) -> int:
        """Determine segment boundary point."""
        return int(start + offset / 2)

    def _describe_reason(
        self, is_slope_different: bool, is_offset_different: bool
    ) -> str:
        """Describe reason for creating a new segment."""
        if is_slope_different and is_offset_different:
            return "slope and offset"
        return "slope" if is_slope_different else "offset"

    def _finalize_segment(
        self, segment_data: dict, stop: int, reason: str = ""
    ) -> Segment:
        """Create a Segment from accumulated window data."""
        segment = Segment(
            start=int(segment_data["s_start"]),
            stop=int(stop),
            slopes=segment_data["slopes"],
            offsets=segment_data["offsets"],
            starts=segment_data["starts"],
            reason_for_new_segment=reason,
        )
        segment.remove_outstanding_windows(self.n)
        return segment

    def _compute_segment_statistics(self, segments: SegmentList) -> None:
        """Compute detailed statistics for each segment."""
        for segment in segments:
            start, stop = segment.start, segment.stop
            xx = self._x[start : stop + 1]
            yy = self._y[start : stop + 1]

            if len(xx) < 2:
                segment.std = 0.0
                segment.span = 0.0
                segment.slope = 0.0
                segment.offset = 0.0
                segment.slopes_std = 0.0
                segment.offsets_std = 0.0
                continue

            fit = np.polyfit(xx, yy, deg=1)
            fit_fn = np.poly1d(fit)
            y_trend = fit_fn(xx)
            y_detrended = yy - y_trend

            segment.slope = float(fit[0])
            segment.offset = float(fit[1])
            segment.std = (
                float(np.std(y_detrended, ddof=0)) if len(y_detrended) > 0 else 0.0
            )

            mean_yy = np.mean(yy)
            if mean_yy != 0:
                segment.span = float(
                    1000 * (np.max(y_detrended) - np.min(y_detrended)) / abs(mean_yy)
                )
            else:
                segment.span = 0.0

            if segment.slopes:
                segment.slopes_std = float(np.std(segment.slopes, ddof=0))
            if segment.offsets:
                segment.offsets_std = float(np.std(segment.offsets, ddof=0))
