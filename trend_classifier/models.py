from enum import Enum


class Metrics(str, Enum):
    ABSOLUTE_ERROR = "absolute_error"
    RELATIVE_ABSOLUTE_ERROR = "relative_absolute_error"
