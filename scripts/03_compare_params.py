"""Script 03: Khảo sát ảnh hưởng của độ dài khung (Frame Length: 20 ms vs. 30 ms).
So sánh số lượng khung Voiced phát hiện, độ chính xác phân loại V/UV và sai số F0mean/F0std.
"""
import argparse
import json
import os
import sys
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.pitch_detector import PitchDetector
from src.analysis.evaluation import evaluate_against_ground_truth
from src.visualization.plotter import plot_parameter_comparison


def run_comparison(
    wav_path: str = "TinHieuHuanLuyen/phone_M1.wav",
    lab_path: str = "TinHieuHuanLuyen/phone_M1.lab",
    threshold: float = 0.4408,
    output_fig: str = "outputs/figures/03_compare_frame_length.png",
    output_json: str = "outputs/reports/compare_params.json",
):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"File not found: {lab_path}")

    # 1. Khởi tạo 3 bộ detector: 20ms, 25ms (Chuẩn cố định), và 30ms
    detector_20ms = PitchDetector(
        method="acf",
        frame_duration_ms=20.0,
        hop_duration_ms=10.0,
        threshold=threshold,
    )
    detector_25ms = PitchDetector(
        method="acf",
        frame_duration_ms=25.0,
        hop_duration_ms=10.0,
        threshold=threshold,
    )
    detector_30ms = PitchDetector(
        method="acf",
        frame_duration_ms=30.0,
        hop_duration_ms=10.0,
        threshold=threshold,
    )

    # 2. Xử lý tín hiệu
    res_20ms = detector_20ms.process_file(wav_path)
    res_25ms = detector_25ms.process_file(wav_path)
    res_30ms = detector_30ms.process_file(wav_path)

    # 3. Đánh giá sai số so với ground truth *.lab
    eval_20ms = evaluate_against_ground_truth(res_20ms, lab_path)
    eval_25ms = evaluate_against_ground_truth(res_25ms, lab_path)
    eval_30ms = evaluate_against_ground_truth(res_30ms, lab_path)

    wav_name = os.path.basename(wav_path)
    ref_mean = eval_25ms["ref_f0_mean"]
    ref_std = eval_25ms["ref_f0_std"]

    # 4. In kết quả đối sánh định lượng
    print(f"File: {wav_path} (Ground Truth: F0mean={ref_mean}Hz, F0std={ref_std}Hz)")
    print(f"Threshold T = {threshold:.3f}")
    print("-" * 100)
    print(f"{'Metric':<30} | {'Frame 20 ms':<18} | {'Frame 25 ms (Chuẩn)':<22} | {'Frame 30 ms':<18}")
    print("-" * 100)
    print(f"{'Total frames':<30} | {res_20ms['num_frames']:<18d} | {res_25ms['num_frames']:<22d} | {res_30ms['num_frames']:<18d}")
    print(f"{'Detected Voiced frames':<30} | {res_20ms['num_voiced']:<18d} | {res_25ms['num_voiced']:<22d} | {res_30ms['num_voiced']:<18d}")
    print(f"{'Detected Unvoiced frames':<30} | {res_20ms['num_unvoiced']:<18d} | {res_25ms['num_unvoiced']:<22d} | {res_30ms['num_unvoiced']:<18d}")
    print(f"{'Detected Silence frames':<30} | {res_20ms['num_silence']:<18d} | {res_25ms['num_silence']:<22d} | {res_30ms['num_silence']:<18d}")
    print(f"{'F0 mean (Hz)':<30} | {eval_20ms['pred_f0_mean']:<18.2f} | {eval_25ms['pred_f0_mean']:<22.2f} | {eval_30ms['pred_f0_mean']:<18.2f}")
    print(f"{'F0 std (Hz)':<30} | {eval_20ms['pred_f0_std']:<18.2f} | {eval_25ms['pred_f0_std']:<22.2f} | {eval_30ms['pred_f0_std']:<18.2f}")
    print(f"{'Abs Error F0mean (Hz)':<30} | {eval_20ms['abs_error_mean']:<18.2f} | {eval_25ms['abs_error_mean']:<22.2f} | {eval_30ms['abs_error_mean']:<18.2f}")
    print(f"{'Rel Error F0mean (%)':<30} | {eval_20ms['rel_error_mean_pct']:<18.2f} | {eval_25ms['rel_error_mean_pct']:<22.2f} | {eval_30ms['rel_error_mean_pct']:<18.2f}")
    print(f"{'V/UV Accuracy (%)':<30} | {eval_20ms['classification_accuracy']:<18.2f} | {eval_25ms['classification_accuracy']:<22.2f} | {eval_30ms['classification_accuracy']:<18.2f}")
    print(f"{'Voiced F1-Score (%)':<30} | {eval_20ms['voiced_f1']:<18.2f} | {eval_25ms['voiced_f1']:<22.2f} | {eval_30ms['voiced_f1']:<18.2f}")
    print("-" * 100)

    # 5. Vẽ đồ thị so sánh 3 panels
    time_sig = np.arange(len(res_25ms["signal"])) / res_25ms["sample_rate"]
    plot_parameter_comparison(
        time_sig=time_sig,
        signal=res_25ms["signal"],
        results_20ms=res_20ms,
        results_30ms=res_30ms,
        eval_20ms=eval_20ms,
        eval_30ms=eval_30ms,
        results_25ms=res_25ms,
        eval_25ms=eval_25ms,
        wav_name=wav_name,
        save_path=output_fig,
    )
    print(f"Saved figure: {output_fig}")

    # 6. Lưu báo cáo JSON
    report = {
        "file": wav_name,
        "threshold": threshold,
        "frame_20ms": {
            "num_frames": res_20ms["num_frames"],
            "num_voiced": res_20ms["num_voiced"],
            "num_unvoiced": res_20ms["num_unvoiced"],
            "num_silence": res_20ms["num_silence"],
            "f0_mean": eval_20ms["pred_f0_mean"],
            "f0_std": eval_20ms["pred_f0_std"],
            "abs_error_mean": eval_20ms["abs_error_mean"],
            "rel_error_mean_pct": eval_20ms["rel_error_mean_pct"],
            "accuracy": eval_20ms["classification_accuracy"],
            "f1_score": eval_20ms["voiced_f1"],
        },
        "frame_25ms_standard": {
            "num_frames": res_25ms["num_frames"],
            "num_voiced": res_25ms["num_voiced"],
            "num_unvoiced": res_25ms["num_unvoiced"],
            "num_silence": res_25ms["num_silence"],
            "f0_mean": eval_25ms["pred_f0_mean"],
            "f0_std": eval_25ms["pred_f0_std"],
            "abs_error_mean": eval_25ms["abs_error_mean"],
            "rel_error_mean_pct": eval_25ms["rel_error_mean_pct"],
            "accuracy": eval_25ms["classification_accuracy"],
            "f1_score": eval_25ms["voiced_f1"],
        },
        "frame_30ms": {
            "num_frames": res_30ms["num_frames"],
            "num_voiced": res_30ms["num_voiced"],
            "num_unvoiced": res_30ms["num_unvoiced"],
            "num_silence": res_30ms["num_silence"],
            "f0_mean": eval_30ms["pred_f0_mean"],
            "f0_std": eval_30ms["pred_f0_std"],
            "abs_error_mean": eval_30ms["abs_error_mean"],
            "rel_error_mean_pct": eval_30ms["rel_error_mean_pct"],
            "accuracy": eval_30ms["classification_accuracy"],
            "f1_score": eval_30ms["voiced_f1"],
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 20ms vs 30ms frame lengths.")
    parser.add_argument("--wav", type=str, default="TinHieuHuanLuyen/phone_M1.wav", help="Path to .wav file")
    parser.add_argument("--lab", type=str, default="TinHieuHuanLuyen/phone_M1.lab", help="Path to .lab file")
    parser.add_argument("--threshold", type=float, default=0.4408, help="V/UV classification threshold")
    parser.add_argument("--output_fig", type=str, default="outputs/figures/03_compare_frame_length.png", help="Figure path")
    parser.add_argument("--output_json", type=str, default="outputs/reports/compare_params.json", help="Report JSON path")

    args = parser.parse_args()
    run_comparison(
        wav_path=args.wav,
        lab_path=args.lab,
        threshold=args.threshold,
        output_fig=args.output_fig,
        output_json=args.output_json,
    )
