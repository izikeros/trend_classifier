"""Package for automated trend classification."""
from trend_classifier.configuration import CONFIG_ABS
from trend_classifier.configuration import CONFIG_REL
from trend_classifier.configuration import Config
from trend_classifier.segment import Segment
from trend_classifier.segmentation import Segmenter
from trend_classifier.types import FigSize

__all__ = ["Segmenter", "Segment", "Config", "CONFIG_ABS", "CONFIG_REL", "FigSize"]
