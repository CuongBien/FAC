"""Evaluation module to compare algorithm predictions with ground truth from *.lab files."""
import re
from typing import Dict, Optional
import numpy as np

from src.core.lab_parser import parse_lab_file, get_frame_labels


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


def format_metric(value, spec=".2f"):
    """Format numbers normally and preserve the width of N/A table entries."""
    if value is not None and np.isfinite(value):
        return format(value, spec)
    width = re.match(r"([<>^]?)(\d+)", spec)
    return format("N/A", (width[1] or ">") + width[2]) if width else "N/A"


def csv_value(value):
    return "" if value is None else str(value)


def rank_metric(value):
    """Undefined scores cannot win a ranking."""
    return float(value) if value is not None and np.isfinite(value) else float("inf")


def compute_error_metrics(
    pred_mean: float,
    pred_std: float,
    ref_mean: float,
    ref_std: float,
    pred_num: Optional[float] = None,
    ref_num: Optional[float] = None,
) -> Dict:
    """Preserve absolute error at zero references; undefined MAPE is None.

    Final Score always averages THREE valid components. Missing/zero references
    never produce zero error or a two-component score.
    """
    def errors(prediction, reference):
        if prediction is None or reference is None:
            return None, None
        if not np.isfinite(prediction) or not np.isfinite(reference):
            return None, None
        absolute = abs(prediction - reference)
        return round_metric(absolute), (absolute / reference * 100.0 if reference > 0 else None)

    abs_mean, mape_mean = errors(pred_mean, ref_mean)
    abs_std, mape_std = errors(pred_std, ref_std)
    abs_num, mape_num = errors(pred_num, ref_num)
    score = mean_metric([mape_mean, mape_std, mape_num])
    return {
        "pred_f0_mean": round_metric(pred_mean), "ref_f0_mean": round_metric(ref_mean),
        "abs_error_mean": abs_mean, "rel_error_mean_pct": round_metric(mape_mean),
        "mape_f0": round_metric(mape_mean), "mape_f0_mean": round_metric(mape_mean),
        "mape_mean": round_metric(mape_mean),
        "pred_f0_std": round_metric(pred_std), "ref_f0_std": round_metric(ref_std),
        "abs_error_std": abs_std, "rel_error_std_pct": round_metric(mape_std),
        "mape_f0_std": round_metric(mape_std), "mape_std": round_metric(mape_std),
        "pred_f0_num": int(round(pred_num)) if round_metric(pred_num) is not None else None,
        "ref_f0_num": int(round(ref_num)) if round_metric(ref_num) is not None else None,
        "abs_error_num": int(round(abs_num)) if abs_num is not None else None,
        "mape_f0_num": round_metric(mape_num), "mape_num": round_metric(mape_num),
        "composite_score": score, "final_score": score,
    }


def evaluate_against_ground_truth(test_results: Dict, lab_path: str) -> Dict:
    """Compare full test predictions with reference *.lab file.

    Evaluates both F0 numeric statistics (mean, std, num) and V/UV/Silence classification accuracy.
    """
    lab_data = parse_lab_file(lab_path)
    ref_mean = lab_data["f0_mean"]
    ref_std = lab_data["f0_std"]
    ref_num = lab_data.get("f0_num")

    frame_times = test_results["frame_times"]
    pred_labels = test_results["labels"]
    gt_segments = lab_data["segments"]

    pred_num = test_results.get("num_voiced")
    if pred_num is None:
        pred_num = int(np.sum(pred_labels == "v"))

    has_segments = len(gt_segments) > 0
    if has_segments:
        gt_labels = get_frame_labels(frame_times, gt_segments)
        if ref_num is None:
            ref_num = int(np.sum(gt_labels == "v"))

        correct = np.sum(pred_labels == gt_labels)
        total = len(pred_labels)
        accuracy = float(correct / total) if total > 0 else 0.0

        voiced_gt = gt_labels == "v"
        voiced_pred = pred_labels == "v"
        tp = np.sum(voiced_gt & voiced_pred)
        fp = np.sum((~voiced_gt) & voiced_pred)
        fn = np.sum(voiced_gt & (~voiced_pred))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        accuracy = None
        precision = None
        recall = None
        f1_score = None

    err_metrics = compute_error_metrics(
        pred_mean=test_results["f0_mean"],
        pred_std=test_results["f0_std"],
        ref_mean=ref_mean,
        ref_std=ref_std,
        pred_num=pred_num,
        ref_num=ref_num,
    )

    return {
        **err_metrics,
        "classification_accuracy": round(accuracy * 100.0, 2) if accuracy is not None else None,
        "voiced_precision": round(precision * 100.0, 2) if precision is not None else None,
        "voiced_recall": round(recall * 100.0, 2) if recall is not None else None,
        "voiced_f1": round(f1_score * 100.0, 2) if f1_score is not None else None,
        "gt_segments": gt_segments,
    }
