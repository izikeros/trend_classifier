"""Module with configuration class for Segmenter and sample configurations."""

from __future__ import annotations

from pydantic import BaseModel

from trend_classifier.models import Metrics


class Config(BaseModel):
    """Configuration of the Segmenter."""

    N: int = 60
    overlap_ratio: float = 0.33

    # deviation for slope (RAE -> 2, AE -> 100)
    alpha: float | None = 2
    # deviation for offset (RAE -> 2, AE -> 0.25)
    beta: float | None = 2

    # primary metrics to be used for alpha and beta changes from segment to segment
    metrics_for_alpha: Metrics = Metrics.RELATIVE_ABSOLUTE_ERROR
    metrics_for_beta: Metrics = Metrics.RELATIVE_ABSOLUTE_ERROR


CONFIG_ABS = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=100,
    beta=2,
    metrics_for_alpha=Metrics.ABSOLUTE_ERROR,
    metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
"""Configuration with using absolute error for alpha."""

CONFIG_REL = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=2,
    beta=2,
    metrics_for_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
    metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
"""Configuration with using relative absolute error for alpha."""

CONFIG_REL_SLOPE_ONLY = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=2,
    beta=None,
    metrics_for_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
    metrics_for_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
"""Configuration with using relative absolute error for alpha."""
