"""Statistical analysis and evaluation tools for pitch detection."""
from .threshold import extract_training_distributions, find_gaussian_threshold
from .evaluation import evaluate_against_ground_truth, compute_error_metrics

__all__ = [
    "extract_training_distributions",
    "find_gaussian_threshold",
    "evaluate_against_ground_truth",
    "compute_error_metrics",
]
