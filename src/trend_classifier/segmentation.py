"""Segmenter facade with support for multiple detection algorithms.

This module provides the main Segmenter class which serves as a facade
for various trend detection algorithms. It maintains backward compatibility
with the legacy API while enabling new features.

For direct access to detection algorithms, see `trend_classifier.detectors`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trend_classifier.configuration import Config
from trend_classifier.detectors import (
    DETECTOR_REGISTRY,
    BaseDetector,
    DetectionResult,
    SlidingWindowDetector,
    get_detector,
)
from trend_classifier.metrics import calculate_error
from trend_classifier.segment import Segment, SegmentList
from trend_classifier.visuals import (
    _plot_detrended_signal,
    _plot_segment,
    _plot_segment_with_trendlines_no_context,
    _plot_segments,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["Segmenter", "calculate_error"]

FigSize = tuple[float | int, float | int]


class Segmenter:
    """Facade for trend segmentation with multiple algorithm support.

    This class provides a unified interface for segmenting time series data
    into regions with similar trends. It supports multiple detection algorithms
    and maintains backward compatibility with the legacy API.

    Args:
        x: Array of x values (indices or timestamps).
        y: Array of y values (signal values).
        df: Pandas DataFrame with time series data.
        column: Column name to use from DataFrame.
        config: Configuration for the sliding window detector (legacy).
        n: Window size shortcut (legacy, use config instead).
        detector: Detection algorithm to use. Can be:
            - A string name: "sliding_window", "pelt", "bottom_up"
            - A BaseDetector instance for custom configuration
        detector_params: Parameters to pass to detector constructor
            (only used when detector is a string).

    Attributes:
        x: Input x values as numpy array.
        y: Input y values as numpy array.
        segments: Detected segments (after calling calculate_segments).
        y_de_trended: Detrended signal values.

    Example (Legacy API - still works):
        >>> seg = Segmenter(x=x, y=y, n=40)
        >>> seg.calculate_segments()
        >>> seg.plot_segments()

    Example (New API - recommended):
        >>> from trend_classifier.detectors import PELTDetector
        >>> seg = Segmenter(x=x, y=y, detector="pelt", detector_params={"penalty": 5})
        >>> result = seg.fit_detect()
        >>> print(f"Found {len(result.segments)} segments")

    Example (Custom detector):
        >>> detector = PELTDetector(model="linear", penalty=3)
        >>> seg = Segmenter(x=x, y=y, detector=detector)
        >>> seg.calculate_segments()
    """

    def __init__(
        self,
        x: list | np.ndarray | None = None,
        y: list | np.ndarray | None = None,
        df: pd.DataFrame | None = None,
        column: str = "Close",
        config: Config | None = None,
        n: int | None = None,
        # New API parameters
        detector: str | BaseDetector = "sliding_window",
        detector_params: dict | None = None,
    ):
        # Handle input data
        self._handle_input_data(column=column, df=df, x=x, y=y)

        # Handle configuration and detector setup
        self._setup_detector(config, n, detector, detector_params)

        # State variables
        self.y_de_trended: list | np.ndarray | None = None
        self.segments: SegmentList | None = None
        self._result: DetectionResult | None = None

    def _handle_input_data(self, column, df, x, y) -> None:
        """Process and validate input data."""
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

        if x is None and y is None and df is None:
            raise ValueError("Provide timeseries data: either x and y or df.")

        if x is not None and y is not None and df is not None:
            raise ValueError(
                "Provide timeseries data: either (x and y) or (df), not all."
            )

        if x is not None and y is not None:
            self.x = np.asarray(x, dtype=np.float64)
            self.y = np.asarray(y, dtype=np.float64)

        if df is not None:
            self.x = np.arange(len(df), dtype=np.float64)
            col_data = df[column]
            if hasattr(col_data, "squeeze"):
                col_data = col_data.squeeze()
            self.y = np.asarray(col_data.values, dtype=np.float64)

    def _setup_detector(
        self,
        config: Config | None,
        n: int | None,
        detector: str | BaseDetector,
        detector_params: dict | None,
    ) -> None:
        """Configure the detection algorithm."""
        # Handle legacy config/n parameters
        if config is not None or n is not None:
            if isinstance(detector, str) and detector != "sliding_window":
                raise ValueError(
                    "Cannot use 'config' or 'n' with non-sliding_window detector. "
                    "Use detector_params instead."
                )

            # Build config
            if config is None:
                config = Config()
            if n is not None:
                if config is not None and config != Config():
                    raise ValueError("Provide either config or N, not both.")
                config = Config(N=n)

            self.config = config
            self._detector = SlidingWindowDetector.from_config(config)

        elif isinstance(detector, str):
            # String detector name
            params = detector_params or {}
            self._detector = get_detector(detector, **params)

            # Create a config for backward compatibility
            if detector == "sliding_window":
                self.config = Config(
                    N=params.get("n", 60),
                    overlap_ratio=params.get("overlap_ratio", 0.33),
                    alpha=params.get("alpha", 2.0),
                    beta=params.get("beta", 2.0),
                )
            else:
                self.config = Config()  # Default config for non-sliding detectors

        elif isinstance(detector, BaseDetector):
            # Pre-configured detector instance
            self._detector = detector
            self.config = Config()  # Default config

        else:
            raise TypeError(
                f"detector must be a string or BaseDetector, got {type(detector)}"
            )

    def calculate_segments(self, progress_callback=None) -> list[Segment]:
        """Calculate segments with similar trend for the time series.

        This is the main method for detecting trend segments. It uses the
        configured detector algorithm.

        Args:
            progress_callback: Optional callback function(current, total) for
                progress reporting during long computations.

        Returns:
            List of detected Segment objects.

        Raises:
            ValueError: If data is not initialized or too short.
        """
        if self.x is None or self.y is None:
            raise ValueError("Segmenter x and y must be initialized!")

        # Use the detector
        if isinstance(self._detector, SlidingWindowDetector):
            self._result = self._detector.fit(self.x, self.y).detect(
                progress_callback=progress_callback
            )
        else:
            self._result = self._detector.fit_detect(self.x, self.y)

        self.segments = self._result.segments

        # Compute detrended values for backward compatibility
        self._compute_detrended_signal()

        return list(self.segments)

    def fit_detect(self) -> DetectionResult:
        """Fit and detect segments in one call.

        This is the new recommended API that returns a DetectionResult
        with additional metadata.

        Returns:
            DetectionResult containing segments, breakpoints, and metadata.

        Example:
            >>> seg = Segmenter(x=x, y=y, detector="pelt")
            >>> result = seg.fit_detect()
            >>> print(f"Algorithm: {result.metadata['algorithm']}")
            >>> print(f"Breakpoints: {result.breakpoints}")
        """
        self.calculate_segments()
        return self._result

    def _compute_detrended_signal(self) -> None:
        """Compute detrended signal for backward compatibility."""
        if self.segments is None:
            return

        y_detrended = []
        for segment in self.segments:
            start, stop = segment.start, segment.stop
            xx = self.x[start : stop + 1]
            yy = self.y[start : stop + 1]

            if len(xx) < 2:
                y_detrended.extend([0.0] * len(yy))
                continue

            fit = np.polyfit(xx, yy, deg=1)
            fit_fn = np.poly1d(fit)
            y_trend = fit_fn(xx)
            y_detrended.extend(yy - y_trend)

        self.y_de_trended = y_detrended

    # ========== Visualization Methods (unchanged) ==========

    def plot_segment(
        self,
        idx: list[int] | int,
        col: str = "red",
        fig_size: FigSize = (10, 5),
    ) -> None:
        """Plot segment with given index or multiple segments.

        Args:
            idx: Index of segment or list of indices.
            col: Color for the segment.
            fig_size: Figure size tuple (width, height).
        """
        _plot_segment(obj=self, idx=idx, col=col, fig_size=fig_size)

    def plot_segment_with_trendlines_no_context(
        self,
        idx: int,
        fig_size: FigSize = (10, 5),
    ) -> None:
        """Plot segment with trendlines, without surrounding context.

        Args:
            idx: Index of segment to plot.
            fig_size: Figure size tuple.
        """
        _plot_segment_with_trendlines_no_context(obj=self, idx=idx, fig_size=fig_size)

    def plot_segments(self, fig_size: FigSize = (8, 4)) -> None:
        """Plot all segments with linear trend lines.

        Args:
            fig_size: Figure size tuple.
        """
        _plot_segments(self, fig_size)

    def plot_detrended_signal(self, fig_size: FigSize = (10, 5)) -> None:
        """Plot the detrended signal.

        Args:
            fig_size: Figure size tuple.
        """
        _plot_detrended_signal(
            x=self.x, y_de_trended=self.y_de_trended, fig_size=fig_size
        )

    # ========== Metrics Methods ==========

    def calc_area_outside_trend(self) -> float:
        """Calculate area outside trend.

        This metric measures how well the detected trends fit the data.
        Lower values indicate better fit.

        Returns:
            Normalized sum of absolute deviations from trend lines.
        """
        if self.y_de_trended is None:
            raise ValueError(
                "Must call calculate_segments() before calc_area_outside_trend()"
            )
        return float(np.sum(np.abs(self.y_de_trended)) / np.mean(self.y) / len(self.y))

    # ========== Utility Methods ==========

    @staticmethod
    def describe_reason_for_new_segment(
        is_offset_different: bool, is_slope_different: bool
    ) -> str:
        """Describe reason for creating a new segment.

        Args:
            is_offset_different: Whether offset changed significantly.
            is_slope_different: Whether slope changed significantly.

        Returns:
            Human-readable description of the reason.
        """
        if is_slope_different and is_offset_different:
            return "slope and offset"
        return "slope" if is_slope_different else "offset"

    @staticmethod
    def list_detectors() -> list[str]:
        """List available detector algorithms.

        Returns:
            List of detector names that can be passed to the constructor.
        """
        return list(DETECTOR_REGISTRY.keys())

    def __repr__(self) -> str:
        n_segments = len(self.segments) if self.segments else 0
        return (
            f"Segmenter(detector={self._detector.name!r}, "
            f"data_points={len(self.x)}, segments={n_segments})"
        )
