"""Evaluation module to compare algorithm predictions with ground truth from *.lab files."""
from typing import Dict, Optional
import numpy as np

from src.core.lab_parser import parse_lab_file, get_frame_labels


def compute_error_metrics(
    pred_mean: float,
    pred_std: float,
    ref_mean: float,
    ref_std: float,
    pred_num: Optional[float] = None,
    ref_num: Optional[float] = None,
) -> Dict:
    """Compute absolute and relative errors (MAPE) between predicted F0 statistics and reference *.lab statistics."""
    diff_mean = pred_mean - ref_mean
    abs_diff_mean = abs(diff_mean)
    mape_f0 = (abs_diff_mean / ref_mean * 100.0) if ref_mean > 0 else 0.0
    rel_diff_mean_pct = mape_f0

    diff_std = pred_std - ref_std
    abs_diff_std = abs(diff_std)
    mape_std = (abs_diff_std / ref_std * 100.0) if ref_std > 0 else 0.0
    rel_diff_std_pct = mape_std

    metrics = {
        "pred_f0_mean": round(pred_mean, 2),
        "ref_f0_mean": round(ref_mean, 2),
        "abs_error_mean": round(abs_diff_mean, 2),
        "rel_error_mean_pct": round(rel_diff_mean_pct, 2),
        "mape_f0": round(mape_f0, 2),
        "mape_f0_mean": round(mape_f0, 2),
        "mape_mean": round(mape_f0, 2),
        "pred_f0_std": round(pred_std, 2),
        "ref_f0_std": round(ref_std, 2),
        "abs_error_std": round(abs_diff_std, 2),
        "rel_error_std_pct": round(rel_diff_std_pct, 2),
        "mape_f0_std": round(mape_std, 2),
        "mape_std": round(mape_std, 2),
    }

    if ref_num is not None and pred_num is not None:
        diff_num = float(pred_num - ref_num)
        abs_diff_num = abs(diff_num)
        mape_num = (abs_diff_num / ref_num * 100.0) if ref_num > 0 else 0.0
        composite_score = (mape_f0 + mape_std + mape_num) / 3.0
        metrics.update({
            "pred_f0_num": int(round(pred_num)),
            "ref_f0_num": int(round(ref_num)),
            "abs_error_num": int(round(abs_diff_num)),
            "mape_f0_num": round(mape_num, 2),
            "mape_num": round(mape_num, 2),
            "composite_score": round(composite_score, 2),
            "final_score": round(composite_score, 2),
        })
    else:
        composite_score = (mape_f0 + mape_std) / 2.0
        metrics.update({
            "composite_score": round(composite_score, 2),
            "final_score": round(composite_score, 2),
        })

    return metrics


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


def compute_pitch_contour_metrics(
    pred_f0: np.ndarray,
    pred_voicing: np.ndarray,
    gt_f0: np.ndarray,
    gt_voicing: np.ndarray,
    tolerance: float = 0.20,
) -> Dict:
    """Compute standard pitch tracking benchmark metrics: VDE, GPE, FFE, FPE.

    Definitions:
        - VDE (Voicing Decision Error, %): Percentage of frames where predicted voicing != ground-truth voicing.
        - GPE (Gross Pitch Error, %): Percentage of shared voiced frames where |pred - gt| / gt > tolerance (default 20%).
        - FFE (F0 Frame Error, %): Percentage of total frames having either a voicing error OR gross pitch error.
        - FPE_MAE (Fine Pitch Error MAE, Hz): Mean absolute error on voiced frames within tolerance.
        - FPE_STD (Fine Pitch Error STD, Hz): Standard deviation of error on voiced frames within tolerance.

    Args:
        pred_f0: 1D array of predicted F0 values.
        pred_voicing: 1D boolean array of predicted voicing flags.
        gt_f0: 1D array of ground-truth F0 values.
        gt_voicing: 1D boolean array of ground-truth voicing flags.
        tolerance: Relative tolerance threshold for Gross Pitch Error (default 0.20 = 20%).

    Returns:
        Dict containing vde, gpe, ffe, fpe_mae, fpe_std, and frame counts.
    """
    total_frames = len(gt_voicing)
    if total_frames == 0:
        return {
            "vde": 0.0,
            "gpe": 0.0,
            "ffe": 0.0,
            "fpe_mae": 0.0,
            "fpe_std": 0.0,
            "total_frames": 0,
            "gt_voiced_frames": 0,
            "pred_voiced_frames": 0,
            "shared_voiced_frames": 0,
        }

    # 1. Voicing Decision Error
    voicing_diff = pred_voicing != gt_voicing
    num_voicing_errors = int(np.sum(voicing_diff))
    vde = (num_voicing_errors / total_frames) * 100.0

    # 2. Gross Pitch Error & Fine Pitch Error
    shared_voiced = pred_voicing & gt_voicing
    num_shared_voiced = int(np.sum(shared_voiced))

    if num_shared_voiced > 0:
        rel_errors = np.abs(pred_f0[shared_voiced] - gt_f0[shared_voiced]) / np.maximum(gt_f0[shared_voiced], 1e-6)
        is_gross = rel_errors > tolerance
        num_gross_errors = int(np.sum(is_gross))
        gpe = (num_gross_errors / num_shared_voiced) * 100.0

        fine_errors = np.abs(pred_f0[shared_voiced][~is_gross] - gt_f0[shared_voiced][~is_gross])
        mape_f0_contour = float(np.mean(rel_errors * 100.0))
        if len(fine_errors) > 0:
            fpe_mae = float(np.mean(fine_errors))
            fpe_std = float(np.std(fine_errors))
        else:
            fpe_mae = 0.0
            fpe_std = 0.0
    else:
        num_gross_errors = 0
        gpe = 0.0
        fpe_mae = 0.0
        fpe_std = 0.0
        mape_f0_contour = 0.0

    # 3. F0 Frame Error (FFE): frames with either voicing decision error or gross pitch error
    total_error_frames = num_voicing_errors + num_gross_errors
    ffe = (total_error_frames / total_frames) * 100.0

    return {
        "vde": round(vde, 2),
        "gpe": round(gpe, 2),
        "ffe": round(ffe, 2),
        "fpe_mae": round(fpe_mae, 2),
        "fpe_std": round(fpe_std, 2),
        "mape_f0_contour": round(mape_f0_contour, 2),
        "total_frames": total_frames,
        "gt_voiced_frames": int(np.sum(gt_voicing)),
        "pred_voiced_frames": int(np.sum(pred_voicing)),
        "shared_voiced_frames": num_shared_voiced,
    }

