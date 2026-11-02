"""Legacy numerical normalization, missing propagation and ranking."""
import numpy as np


def round_metric(value, digits=2):
    """JSON-compatible number or None for a missing/non-finite metric."""
    if value is None or not np.isfinite(value):
        return None
    return float(value) if digits is None else round(float(value), digits)


def mean_metric(values, digits=2):
    """Mean over the complete population; missing files never disappear silently."""
    values = list(values)
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return None
    return round_metric(np.mean(values), digits)


def rank_metric(value):
    """Undefined scores cannot win a ranking."""
    return float(value) if value is not None and np.isfinite(value) else float("inf")
