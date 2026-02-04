"""Error metrics for trend comparison."""

from __future__ import annotations

from trend_classifier.models import Metrics


def calculate_error(
    a: float,
    b: float,
    metrics: Metrics = Metrics.ABSOLUTE_ERROR,
    min_denom: float = 1e-6,
) -> float:
    """Calculate how much two parameters differ.

    Used to compare slopes/offsets between adjacent windows to detect
    trend changes.

    Args:
        a: First parameter (e.g., slope of previous window).
        b: Second parameter (e.g., slope of current window).
        metrics: Error metric to use for comparison.
        min_denom: Minimum denominator for relative error to avoid
            division by near-zero values.

    Returns:
        Measure of difference between the two parameters.

    Raises:
        ValueError: If inputs are not numeric or metrics is unsupported.

    Examples:
        >>> calculate_error(10, 12, Metrics.ABSOLUTE_ERROR)
        2.0
        >>> calculate_error(10, 12, Metrics.RELATIVE_ABSOLUTE_ERROR)
        0.2
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Both 'a' and 'b' must be numeric values.")

    if metrics == Metrics.ABSOLUTE_ERROR:
        return abs(a - b)
    elif metrics == Metrics.RELATIVE_ABSOLUTE_ERROR:
        return abs(a - b) / max(abs(a), min_denom)
    else:
        raise ValueError(f"Unsupported metrics: {metrics}")
