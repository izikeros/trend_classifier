# Detectors

## Factory Functions

::: trend_classifier.detectors.get_detector

::: trend_classifier.detectors.list_detectors

## Base Classes

::: trend_classifier.detectors.base.BaseDetector
    options:
      show_root_heading: true
      members:
        - name
        - fit
        - detect
        - fit_detect

::: trend_classifier.detectors.base.DetectionResult
    options:
      show_root_heading: true

## Detector Implementations

### SlidingWindowDetector

::: trend_classifier.detectors.sliding_window.SlidingWindowDetector
    options:
      show_root_heading: true
      members:
        - __init__
        - from_config
        - fit
        - detect

### BottomUpDetector

::: trend_classifier.detectors.bottom_up.BottomUpDetector
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - detect

### PELTDetector

::: trend_classifier.detectors.pelt.PELTDetector
    options:
      show_root_heading: true
      members:
        - __init__
        - fit
        - detect
