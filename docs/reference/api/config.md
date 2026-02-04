# Configuration

## Config Class

::: trend_classifier.configuration.Config
    options:
      show_root_heading: true

## Preset Configurations

### CONFIG_REL

Relative error for both slope and offset comparison (default behavior).

```python
from trend_classifier.configuration import CONFIG_REL

CONFIG_REL = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=2,
    beta=2,
    metrics_for_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
    metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
```

### CONFIG_ABS

Absolute error for slope comparison, relative for offset.

```python
from trend_classifier.configuration import CONFIG_ABS

CONFIG_ABS = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=100,
    beta=2,
    metrics_for_alpha=Metrics.ABSOLUTE_ERROR,
    metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
```

### CONFIG_REL_SLOPE_ONLY

Only check slope changes, ignore offset.

```python
from trend_classifier.configuration import CONFIG_REL_SLOPE_ONLY

CONFIG_REL_SLOPE_ONLY = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=2,
    beta=None,  # Disabled
    metrics_for_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
    metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
```

## Metrics Enum

::: trend_classifier.models.Metrics
    options:
      show_root_heading: true
