"""Compatibility exports; metric ownership is now in src.metrics."""
from src.metrics.numeric import round_metric, mean_metric, rank_metric
from src.metrics.legacy import compute_error_metrics
from src.metrics.evaluation import evaluate_against_ground_truth


def __getattr__(name):
    if name in ('format_metric', 'csv_value'):
        from src.metrics import formatting
        return getattr(formatting, name)
    raise AttributeError(name)

__all__ = ['round_metric', 'mean_metric', 'rank_metric', 'format_metric', 'csv_value',
           'compute_error_metrics', 'evaluate_against_ground_truth']
