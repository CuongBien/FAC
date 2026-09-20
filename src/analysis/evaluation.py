"""Evaluation module to compare algorithm predictions with ground truth from *.lab files."""
from typing import Dict, Optional
import numpy as np

from src.core.lab_parser import parse_lab_file, get_frame_labels


def compute_error_metrics(
    pred_mean: float, pred_std: float, ref_mean: float, ref_std: float
) -> Dict:
    """Compute absolute and relative errors between predicted F0 statistics and reference *.lab statistics."""
    diff_mean = pred_mean - ref_mean
    abs_diff_mean = abs(diff_mean)
    rel_diff_mean_pct = (abs_diff_mean / ref_mean * 100.0) if ref_mean > 0 else 0.0

    diff_std = pred_std - ref_std
    abs_diff_std = abs(diff_std)
    rel_diff_std_pct = (abs_diff_std / ref_std * 100.0) if ref_std > 0 else 0.0

    return {
        "pred_f0_mean": round(pred_mean, 2),
        "ref_f0_mean": round(ref_mean, 2),
        "abs_error_mean": round(abs_diff_mean, 2),
        "rel_error_mean_pct": round(rel_diff_mean_pct, 2),
        "pred_f0_std": round(pred_std, 2),
        "ref_f0_std": round(ref_std, 2),
        "abs_error_std": round(abs_diff_std, 2),
        "rel_error_std_pct": round(rel_diff_std_pct, 2),
    }


def evaluate_against_ground_truth(test_results: Dict, lab_path: str) -> Dict:
    """Compare full test predictions with reference *.lab file.

    Evaluates both F0 numeric statistics (mean, std) and V/UV/Silence classification accuracy.
    """
    lab_data = parse_lab_file(lab_path)
    ref_mean = lab_data["f0_mean"]
    ref_std = lab_data["f0_std"]

    err_metrics = compute_error_metrics(
        pred_mean=test_results["f0_mean"],
        pred_std=test_results["f0_std"],
        ref_mean=ref_mean,
        ref_std=ref_std,
    )

    # Frame-level classification accuracy
    frame_times = test_results["frame_times"]
    pred_labels = test_results["labels"]
    gt_labels = get_frame_labels(frame_times, lab_data["segments"])

    correct = np.sum(pred_labels == gt_labels)
    total = len(pred_labels)
    accuracy = float(correct / total) if total > 0 else 0.0

    # Voiced specific precision & recall
    voiced_gt = gt_labels == "v"
    voiced_pred = pred_labels == "v"
    tp = np.sum(voiced_gt & voiced_pred)
    fp = np.sum((~voiced_gt) & voiced_pred)
    fn = np.sum(voiced_gt & (~voiced_pred))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        **err_metrics,
        "classification_accuracy": round(accuracy * 100.0, 2),
        "voiced_precision": round(precision * 100.0, 2),
        "voiced_recall": round(recall * 100.0, 2),
        "voiced_f1": round(f1_score * 100.0, 2),
        "gt_segments": lab_data["segments"],
    }
