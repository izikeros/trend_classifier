"""Base classes for trend detection algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from trend_classifier.segment import Segment, SegmentList


@dataclass
class DetectionResult:
    """Result from a trend detection algorithm.

    Attributes:
        segments: List of detected segments with trend information.
        breakpoints: List of indices where trend changes occur.
        metadata: Algorithm-specific metadata (e.g., cost values, statistics).
    """

    segments: SegmentList
    breakpoints: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of segments."""
        return len(self.segments)

    def to_dataframe(self):
        """Convert segments to DataFrame."""
        return self.segments.to_dataframe()


class BaseDetector(ABC):
    """Abstract base class for trend detection algorithms.

    All detection algorithms should inherit from this class and implement
    the required methods. This enables the Strategy pattern for swapping
    algorithms at runtime.

    Example:
        >>> class MyDetector(BaseDetector):
        ...     name = "my_detector"
        ...     def fit(self, x, y):
        ...         self._x, self._y = x, y
        ...         return self
        ...     def detect(self):
        ...         # Detection logic here
        ...         return DetectionResult(segments=SegmentList())
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name for identification."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> BaseDetector:
        """Fit the detector to the data.

        Args:
            x: Array of x values (indices or timestamps).
            y: Array of y values (signal values).

        Returns:
            Self for method chaining.
        """

    @abstractmethod
    def detect(self) -> DetectionResult:
        """Detect trend changes and return segments.

        Must be called after fit().

        Returns:
            DetectionResult containing segments and metadata.

        Raises:
            RuntimeError: If called before fit().
        """

    def fit_detect(self, x: np.ndarray, y: np.ndarray) -> DetectionResult:
        """Convenience method to fit and detect in one call.

        Args:
            x: Array of x values.
            y: Array of y values.

        Returns:
            DetectionResult containing segments and metadata.
        """
        return self.fit(x, y).detect()

    def _validate_fitted(self) -> None:
        """Check if the detector has been fitted.

        Raises:
            RuntimeError: If detector hasn't been fitted.
        """
        if not hasattr(self, "_x") or self._x is None:
            raise RuntimeError(
                f"{self.name} detector must be fitted before calling detect(). "
                "Call fit(x, y) first."
            )

    def _create_segment(
        self,
        start: int,
        stop: int,
        x: np.ndarray,
        y: np.ndarray,
        reason: str = "",
    ) -> Segment:
        """Create a Segment with computed trend statistics.

        Args:
            start: Start index of segment.
            stop: Stop index of segment (exclusive).
            x: Full x array.
            y: Full y array.
            reason: Reason for segment boundary.

        Returns:
            Segment with computed slope, offset, and statistics.
        """
        xx = x[start:stop]
        yy = y[start:stop]

        if len(xx) < 2:
            return Segment(
                start=start,
                stop=stop,
                slope=0.0,
                offset=float(yy[0]) if len(yy) > 0 else 0.0,
                reason_for_new_segment=reason,
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
            reason_for_new_segment=reason,
        )
