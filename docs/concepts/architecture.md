# Architecture

Understanding how trend_classifier is designed.

## Overview

trend_classifier uses a **Strategy Pattern** to support multiple detection algorithms. This allows:

- Swapping algorithms at runtime
- Adding new algorithms without changing core code
- Consistent API across all detectors

```
┌─────────────────────────────────────────────────────────────┐
│                        Segmenter                            │
│                     (Facade Class)                          │
├─────────────────────────────────────────────────────────────┤
│  - Handles data input (arrays, DataFrames)                  │
│  - Delegates to selected detector                           │
│  - Provides visualization methods                           │
│  - Maintains backward compatibility                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      BaseDetector                           │
│                    (Abstract Base Class)                    │
├─────────────────────────────────────────────────────────────┤
│  + fit(x, y) → self                                         │
│  + detect() → DetectionResult                               │
│  + fit_detect(x, y) → DetectionResult                       │
│  + name: str                                                │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │SlidingWindow  │ │  BottomUp     │ │    PELT       │
    │   Detector    │ │  Detector     │ │  Detector     │
    ├───────────────┤ ├───────────────┤ ├───────────────┤
    │ Window-based  │ │ Merge-based   │ │ ruptures lib  │
    │ linear fit    │ │ segmentation  │ │ optimal CPD   │
    └───────────────┘ └───────────────┘ └───────────────┘
```

## Key Components

### Segmenter (Facade)

The main entry point that users interact with:

```python
from trend_classifier import Segmenter

# Legacy API (backward compatible)
seg = Segmenter(x=x, y=y, n=40)

# New API with detector selection
seg = Segmenter(x=x, y=y, detector="pelt", detector_params={"penalty": 5})
```

**Responsibilities:**

- Data preprocessing and validation
- Detector instantiation and delegation
- Visualization methods
- Legacy API support

### BaseDetector (Abstract)

Defines the interface all detectors must implement:

```python
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm identifier."""
        pass
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseDetector":
        """Fit to data."""
        pass
    
    @abstractmethod
    def detect(self) -> DetectionResult:
        """Detect segments."""
        pass
```

### DetectionResult

Structured output from detectors:

```python
@dataclass
class DetectionResult:
    segments: SegmentList      # Detected segments
    breakpoints: list[int]     # Change point indices
    metadata: dict             # Algorithm-specific info
```

### Segment

Represents a single detected trend segment:

```python
class Segment(BaseModel):
    start: int          # Start index
    stop: int           # End index
    slope: float        # Trend slope
    offset: float       # Y-intercept
    std: float          # Detrended std dev
    span: float         # Normalized range
    # ... more attributes
```

## Adding a New Detector

To add a custom detector:

```python
from trend_classifier.detectors import BaseDetector, DetectionResult

class MyDetector(BaseDetector):
    name = "my_detector"
    
    def __init__(self, param1=10):
        self.param1 = param1
        self._x = None
        self._y = None
    
    def fit(self, x, y):
        self._x = x
        self._y = y
        return self
    
    def detect(self):
        self._validate_fitted()
        # Your detection logic here
        segments = self._find_segments()
        return DetectionResult(
            segments=segments,
            breakpoints=[s.start for s in segments[1:]],
            metadata={"algorithm": self.name}
        )
```

Register it:

```python
from trend_classifier.detectors import DETECTOR_REGISTRY
DETECTOR_REGISTRY["my_detector"] = MyDetector

# Now usable
seg = Segmenter(x=x, y=y, detector="my_detector")
```

## Data Flow

```
User Input                Processing              Output
─────────────────────────────────────────────────────────────
                              
DataFrame ─┐                                   ┌─ SegmentList
           ├─► Segmenter ─► Detector ─► Result ├─ DataFrame
x, y arrays┘      │              │             └─ Plots
                  │              │
                  ▼              ▼
              Validate      fit_detect()
              Convert       segments
              to numpy      breakpoints
```

## Module Structure

```
trend_classifier/
├── __init__.py          # Public API exports
├── segmentation.py      # Segmenter facade
├── segment.py           # Segment, SegmentList
├── configuration.py     # Config class, presets
├── metrics.py           # Error calculation
├── visuals.py           # Plotting functions
└── detectors/
    ├── __init__.py      # Registry, get_detector()
    ├── base.py          # BaseDetector, DetectionResult
    ├── sliding_window.py
    ├── bottom_up.py
    └── pelt.py
```

## Design Decisions

### Why Strategy Pattern?

1. **Extensibility** - Add algorithms without touching Segmenter
2. **Testing** - Each detector is independently testable
3. **User choice** - Users pick the best algorithm for their data

### Why Facade Pattern for Segmenter?

1. **Simplicity** - Single entry point for users
2. **Backward compatibility** - Legacy API preserved
3. **Convenience** - Handles data conversion, visualization

### Why Pydantic for Segment?

1. **Validation** - Automatic type checking
2. **Serialization** - Easy JSON/dict conversion
3. **Documentation** - Field descriptions as docs
