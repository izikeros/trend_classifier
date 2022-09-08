"""Module with pydantic models and classes for Enum types."""
from enum import Enum


class Metrics(str, Enum):
    """Enum class for metrics used to calculate deviations."""

    ABSOLUTE_ERROR = "absolute_error"
    RELATIVE_ABSOLUTE_ERROR = "relative_absolute_error"
