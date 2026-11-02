"""LAB-reference adapter and prediction invariants for legacy-v1 evaluation."""
from pathlib import Path
from typing import Dict
import numpy as np
from src.core.lab_parser import parse_lab_file, get_frame_labels
from .legacy import compute_error_metrics


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


def evaluate_result(result, wav_path):
    f0 = np.asarray(result['f0_contour'])
    labels = np.asarray(result['labels'])
    voiced = labels == 'v'
    if not (len(f0) == len(labels) == len(result['frame_times'])):
        raise AssertionError('Frame alignment mismatch')
    if not np.all(np.isfinite(f0)) or np.any(f0[~voiced] != 0):
        raise AssertionError('Non-finite F0 or pitch outside voiced labels')
    if np.any((f0[voiced] < 70) | (f0[voiced] > 400)) or int(voiced.sum()) != result['num_voiced']:
        raise AssertionError('Pitch range/count mismatch')
    return {'file': Path(wav_path).name, **evaluate_against_ground_truth(result, str(Path(wav_path).with_suffix('.lab')))}
