"""Package for automated trend classification."""

from trend_classifier.configuration import CONFIG_ABS, CONFIG_REL, Config
from trend_classifier.segment import Segment
from trend_classifier.segmentation import Segmenter

__all__ = ["Segmenter", "Segment", "Config", "CONFIG_ABS", "CONFIG_REL"]
__version__ = "0.1.13"
