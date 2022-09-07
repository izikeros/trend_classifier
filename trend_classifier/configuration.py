""""""
from pydantic import BaseModel
from trend_classifier.models import Metrics


class Config(BaseModel):
    N: int = 60
    overlap_ratio: float = 0.33

    # deviation for slope (RAE -> 2, AE -> 100)
    alpha: float = 2
    # deviation for offset (RAE -> 2, AE -> 0.25)
    beta: float = 2
    metrics_alpha: Metrics = Metrics.RELATIVE_ABSOLUTE_ERROR
    metrics_beta: Metrics = Metrics.RELATIVE_ABSOLUTE_ERROR


CONFIG_ABS = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=100,
    beta=2,
    metrics_alpha=Metrics.ABSOLUTE_ERROR,
    metrics_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
CONFIG_REL = Config(
    N=40,
    overlap_ratio=0.33,
    alpha=2,
    beta=2,
    metrics_alpha=Metrics.RELATIVE_ABSOLUTE_ERROR,
    metrics_beta=Metrics.RELATIVE_ABSOLUTE_ERROR,
)
