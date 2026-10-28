"""Statistical analysis and evaluation tools for pitch detection."""
from .threshold import extract_training_distributions, find_gaussian_threshold
from .unsupervised_threshold import (
    compute_otsu_threshold,
    compute_histogram_valley_threshold,
    compute_median_threshold,
    compute_kmeans_threshold,
    compute_all_unsupervised_thresholds,
    extract_active_speech_frame_metrics,
)
from .advanced_unsupervised import (
    compute_gmm_em_1d,
    compute_zcr,
    cluster_multifeature_2d,
    train_and_decode_2state_hmm,
    compute_sliding_window_adaptive_threshold,
)
from .evaluation import evaluate_against_ground_truth, compute_error_metrics, compute_pitch_contour_metrics

__all__ = [
    "extract_training_distributions",
    "find_gaussian_threshold",
    "compute_otsu_threshold",
    "compute_histogram_valley_threshold",
    "compute_median_threshold",
    "compute_kmeans_threshold",
    "compute_all_unsupervised_thresholds",
    "extract_active_speech_frame_metrics",
    "compute_gmm_em_1d",
    "compute_zcr",
    "cluster_multifeature_2d",
    "train_and_decode_2state_hmm",
    "compute_sliding_window_adaptive_threshold",
    "evaluate_against_ground_truth",
    "compute_error_metrics",
    "compute_pitch_contour_metrics",
]
