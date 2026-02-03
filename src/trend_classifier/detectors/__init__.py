"""Pluggable trend detection algorithms.

This module provides multiple algorithms for detecting trend changes
in time series data. All detectors implement the BaseDetector interface.

Available Detectors:
    - SlidingWindowDetector: Original algorithm using sliding window + linear regression
    - PELTDetector: PELT algorithm via ruptures (requires: pip install ruptures)
    - BottomUpDetector: Bottom-up merge segmentation

Example:
    >>> from trend_classifier.detectors import SlidingWindowDetector, PELTDetector
    >>>
    >>> # Use sliding window (legacy)
    >>> detector = SlidingWindowDetector(n=40, alpha=2.0)
    >>> result = detector.fit_detect(x, y)
    >>>
    >>> # Use PELT (recommended for most cases)
    >>> detector = PELTDetector(penalty=5)
    >>> result = detector.fit_detect(x, y)
"""

from trend_classifier.detectors.base import BaseDetector, DetectionResult
from trend_classifier.detectors.bottom_up import BottomUpDetector
from trend_classifier.detectors.sliding_window import SlidingWindowDetector

# Registry of available detectors
DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "sliding_window": SlidingWindowDetector,
    "bottom_up": BottomUpDetector,
}

# Try to register PELT if ruptures is available
try:
    from trend_classifier.detectors.pelt import PELTDetector

    DETECTOR_REGISTRY["pelt"] = PELTDetector
    _PELT_AVAILABLE = True
except ImportError:
    PELTDetector = None  # type: ignore
    _PELT_AVAILABLE = False


def get_detector(name: str, **kwargs) -> BaseDetector:
    """Get a detector instance by name.

    Args:
        name: Detector name (e.g., "sliding_window", "pelt", "bottom_up").
        **kwargs: Parameters to pass to the detector constructor.

    Returns:
        Configured detector instance.

    Raises:
        ValueError: If detector name is not recognized.
        ImportError: If detector requires unavailable dependencies.

    Example:
        >>> detector = get_detector("pelt", penalty=5)
        >>> result = detector.fit_detect(x, y)
    """
    if name not in DETECTOR_REGISTRY:
        available = ", ".join(DETECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown detector '{name}'. Available detectors: {available}")

    return DETECTOR_REGISTRY[name](**kwargs)


def list_detectors() -> list[str]:
    """List available detector names.

    Returns:
        List of registered detector names.
    """
    return list(DETECTOR_REGISTRY.keys())


__all__ = [
    "DETECTOR_REGISTRY",
    "BaseDetector",
    "BottomUpDetector",
    "DetectionResult",
    "PELTDetector",
    "SlidingWindowDetector",
    "get_detector",
    "list_detectors",
]
