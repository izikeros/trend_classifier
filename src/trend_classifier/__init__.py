"""Package for automated trend classification.

This package provides tools for detecting and classifying trends in time series
data, with a focus on financial applications.

Basic Usage:
    >>> from trend_classifier import Segmenter
    >>> seg = Segmenter(x=x, y=y, n=40)
    >>> seg.calculate_segments()
    >>> seg.plot_segments()

Advanced Usage (multiple algorithms):
    >>> from trend_classifier import Segmenter
    >>> from trend_classifier.detectors import PELTDetector, get_detector
    >>>
    >>> # Using PELT algorithm
    >>> seg = Segmenter(x=x, y=y, detector="pelt", detector_params={"penalty": 5})
    >>> result = seg.fit_detect()
    >>>
    >>> # Using custom detector
    >>> detector = PELTDetector(model="linear", penalty=3)
    >>> seg = Segmenter(x=x, y=y, detector=detector)
"""

from trend_classifier.configuration import (
    CONFIG_ABS,
    CONFIG_REL,
    CONFIG_REL_SLOPE_ONLY,
    Config,
)
from trend_classifier.detectors import (
    BaseDetector,
    BottomUpDetector,
    DetectionResult,
    SlidingWindowDetector,
    get_detector,
    list_detectors,
)
from trend_classifier.segment import Segment, SegmentList
from trend_classifier.segmentation import Segmenter

# Try to import PELTDetector (requires ruptures)
try:
    from trend_classifier.detectors import PELTDetector
except ImportError:
    PELTDetector = None  # type: ignore

__all__ = [
    "CONFIG_ABS",
    "CONFIG_REL",
    "CONFIG_REL_SLOPE_ONLY",
    "BaseDetector",
    "BottomUpDetector",
    "Config",
    "DetectionResult",
    "PELTDetector",
    "Segment",
    "SegmentList",
    "Segmenter",
    "SlidingWindowDetector",
    "get_detector",
    "list_detectors",
]
__version__ = "0.3.0"
