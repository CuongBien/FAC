"""Script 04: Chạy kiểm thử tự động toàn bộ 4 file kiểm thử (TinHieuKiemThu) trong 1 lệnh duy nhất.
Tạo 4 biểu đồ hình ảnh đối sánh và xuất bảng đánh giá định lượng F0mean, F0std, V/UV Accuracy.
Hỗ trợ cả 2 thuật toán: ACF và AMDF.
"""
import argparse
import glob
import json
import os
import sys
from typing import Optional, List, Tuple
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
from src.visualization.plotter import plot_signal_and_f0_contour
from src.plugins import (
    PluginPitchDetector,
    HysteresisPlugin,
    EnergyExtensionPlugin,
    BandpassFilterPlugin,
)


def build_detector_with_plugins(
    plugins_arg: str,
    base_detector: PitchDetector,
    threshold: float,
) -> Tuple[PitchDetector, List[str]]:
    """Build detector configured with requested plugins."""
    if not plugins_arg or plugins_arg.lower() in ("none", "no", "false", "0"):
        return base_detector, []

    parts = [p.strip().lower() for p in plugins_arg.split(",")]
    plugin_instances = []
    applied_names = []

    # Bandpass filter plugin
    if "all" in parts or any(p in ("bandpass", "filter", "bp") for p in parts):
        bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)
        plugin_instances.append(bp)
        applied_names.append(bp.name)

    # Hysteresis thresholding plugin
    if "all" in parts or any(p in ("hysteresis", "schmitt", "hyst") for p in parts):
        if base_detector.method == "amdf":
            t_low = threshold * 0.90
            t_high = threshold
        else:
            t_low = 0.40 if threshold < 0.50 else 0.42
            t_high = threshold
        hyst = HysteresisPlugin(t_high=t_high, t_low=t_low)
        plugin_instances.append(hyst)
        applied_names.append(hyst.name)

    # Energy extension plugin
    if "all" in parts or any(p in ("energy", "ste", "energy_ext", "ext") for p in parts):
        ste_ext = EnergyExtensionPlugin(threshold_discount=0.20)
        plugin_instances.append(ste_ext)
        applied_names.append(ste_ext.name)

    if not plugin_instances:
        return base_detector, []

    detector = PluginPitchDetector(base_detector=base_detector, plugins=plugin_instances)
    return detector, applied_names


def run_testing(
    test_dir: str = "TinHieuKiemThu",
    method: str = "acf",
    threshold_file: Optional[str] = None,
    default_threshold: Optional[float] = None,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    plugins: str = "none",
    output_dir_fig: str = "outputs/figures",
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
):
    method = method.lower()
    p_lower = (plugins or "").lower().strip()
    parts = [p.strip() for p in p_lower.split(",")]
    has_bandpass = (p_lower == "all") or any(p in ("bandpass", "filter", "bp") for p in parts)

    # 1. Tự động chọn file ngưỡng tương ứng
    if threshold_file is None:
        if method == "amdf":
            if has_bandpass:
                threshold_file = "outputs/reports/threshold_amdf_bandpassprefilter.json"
                fallback_t = 0.3734
            else:
                threshold_file = "outputs/reports/threshold_amdf.json"
                fallback_t = 0.4380
        else:
            if has_bandpass:
                threshold_file = "outputs/reports/threshold_acf_bandpassprefilter.json"
                fallback_t = 0.4892
            else:
                threshold_file = "outputs/reports/threshold_acf.json"
                fallback_t = 0.4408
    else:
        if method == "amdf":
            fallback_t = 0.3734 if has_bandpass else 0.4380
        else:
            fallback_t = 0.4892 if has_bandpass else 0.4408

    if default_threshold is not None:
        threshold = default_threshold
    else:
        threshold = fallback_t

    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                threshold = float(t_data.get("threshold_T", threshold))
        except Exception:
            pass

    # Tự động chọn file xuất nếu chưa truyền
    if output_json is None:
        output_json = f"outputs/reports/test_results_{method}.json" if method != "acf" else "outputs/reports/test_results.json"
    if output_csv is None:
        output_csv = f"outputs/reports/test_summary_{method}.csv" if method != "acf" else "outputs/reports/test_summary.csv"

    # 2. Khởi tạo bộ nhận diện PitchDetector
    base_detector = PitchDetector(
        method=method,
        frame_duration_ms=frame_duration_ms,
        hop_duration_ms=hop_duration_ms,
        f0_min=f0_min,
        f0_max=f0_max,
        threshold=threshold,
        ste_silence_ratio=0.008,
        use_median_filter=True,
        median_size=3,
    )

    # 3. Cắm các plugin nếu người dùng yêu cầu
    detector, applied_plugin_names = build_detector_with_plugins(plugins, base_detector, threshold)
    is_enhanced = len(applied_plugin_names) > 0
    plugin_desc = f" | Plugins: [{', '.join(applied_plugin_names)}]" if is_enhanced else " | Mode: Baseline (No plugins)"

    wav_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"Không tìm thấy file WAV nào trong: {test_dir}")

    os.makedirs(output_dir_fig, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    results_all = []

    print("=" * 105)
    print(f"KIỂM THỬ THUẬT TOÁN {method.upper()} TRÊN TẬP TÍN HIỆU KIỂM THỬ ({test_dir})")
    print(f"Tham số: Method={method.upper()}, FrameLen={frame_duration_ms}ms, HopLen={hop_duration_ms}ms, Range=[{f0_min:.0f}, {f0_max:.0f}]Hz, Threshold T={threshold:.4f}{plugin_desc}")
    print("=" * 105)
    header = f"{'Tên file':<16} | {'Ref F0mean':<10} | {'Pred F0mean':<11} | {'|ΔF0| (Hz)':<10} | {'Lệch %':<7} | {'Ref std':<8} | {'Pred std':<8} | {'|Δstd|':<7} | {'V/UV Acc':<8} | {'F1-Score':<8}"
    print(header)
    print("-" * 105)

    for wav_path in wav_files:
        base_name = os.path.basename(wav_path)
        file_id = os.path.splitext(base_name)[0]
        lab_path = os.path.splitext(wav_path)[0] + ".lab"

        # 4. Chạy dự đoán cao độ
        res = detector.process_file(wav_path)

        # Đánh giá sai số so với ground truth *.lab
        eval_res = evaluate_against_ground_truth(res, lab_path)

        row = (
            f"{base_name:<16} | "
            f"{eval_res['ref_f0_mean']:<10.2f} | "
            f"{eval_res['pred_f0_mean']:<11.2f} | "
            f"{eval_res['abs_error_mean']:<10.2f} | "
            f"{eval_res['rel_error_mean_pct']:<5.2f}% | "
            f"{eval_res['ref_f0_std']:<8.2f} | "
            f"{eval_res['pred_f0_std']:<8.2f} | "
            f"{eval_res['abs_error_std']:<7.2f} | "
            f"{eval_res['classification_accuracy']:<7.2f}% | "
            f"{eval_res['voiced_f1']:<7.2f}%"
        )
        print(row)

        # 5. Xuất hình vẽ cho từng file kiểm thử (1 hình / 1 file)
        prefix = f"04_test_enhanced_{method}_" if is_enhanced else (f"04_test_{method}_" if method != "acf" else "04_test_")
        if is_enhanced and method == "acf":
            prefix = "04_test_enhanced_"

        fig_name = f"{prefix}{file_id}.png"
        fig_path = os.path.join(output_dir_fig, fig_name)
        time_sig = np.arange(len(res["signal"])) / res["sample_rate"]
        title_suffix = f" [Enhanced: {', '.join(applied_plugin_names)}]" if is_enhanced else " [Baseline]"
        plot_signal_and_f0_contour(
            time_sig=time_sig,
            signal=res["signal"],
            time_f0=res["frame_times"],
            f0_contour=res["f0_contour"],
            ground_truth_segments=eval_res["gt_segments"],
            f0_mean=eval_res["pred_f0_mean"],
            f0_std=eval_res["pred_f0_std"],
            ref_f0_mean=eval_res["ref_f0_mean"],
            ref_f0_std=eval_res["ref_f0_std"],
            title=f"Đường bao tần số F0 ước lượng bằng {method.upper()}{title_suffix} - {base_name} (Fs={res['sample_rate']}Hz)",
            save_path=fig_path,
        )

        results_all.append({
            "file": base_name,
            "sample_rate": res["sample_rate"],
            "duration": round(res["duration"], 3),
            "ref_f0_mean": eval_res["ref_f0_mean"],
            "pred_f0_mean": eval_res["pred_f0_mean"],
            "abs_error_mean": eval_res["abs_error_mean"],
            "rel_error_mean_pct": eval_res["rel_error_mean_pct"],
            "ref_f0_std": eval_res["ref_f0_std"],
            "pred_f0_std": eval_res["pred_f0_std"],
            "abs_error_std": eval_res["abs_error_std"],
            "rel_error_std_pct": eval_res["rel_error_std_pct"],
            "accuracy": eval_res["classification_accuracy"],
            "voiced_f1": eval_res["voiced_f1"],
            "figure_path": fig_path,
        })

    # 6. Tính trung bình sai số toàn tập kiểm thử
    avg_abs_err_mean = np.mean([r["abs_error_mean"] for r in results_all])
    avg_rel_err_mean = np.mean([r["rel_error_mean_pct"] for r in results_all])
    avg_acc = np.mean([r["accuracy"] for r in results_all])
    avg_f1 = np.mean([r["voiced_f1"] for r in results_all])

    print("-" * 105)
    print(f"{'TRUNG BÌNH TOÀN TẬP':<16} | {'-':<10} | {'-':<11} | {avg_abs_err_mean:<10.2f} | {avg_rel_err_mean:<6.2f}% | {'-':<8} | {'-':<8} | {'-':<7} | {avg_acc:<7.2f}% | {avg_f1:<7.2f}%")
    print("=" * 105)

    # 7. Lưu kết quả ra file JSON
    summary_data = {
        "params": {
            "method": method,
            "frame_duration_ms": frame_duration_ms,
            "hop_duration_ms": hop_duration_ms,
            "threshold_T": threshold,
            "f0_min": f0_min,
            "f0_max": f0_max,
        },
        "average_metrics": {
            "mean_abs_error_f0": round(float(avg_abs_err_mean), 2),
            "mean_rel_error_pct": round(float(avg_rel_err_mean), 2),
            "mean_v_uv_accuracy": round(float(avg_acc), 2),
            "mean_voiced_f1": round(float(avg_f1), 2),
        },
        "files": results_all,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Saved report JSON: {output_json}")

    # 8. Lưu kết quả ra file CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("File,Fs_Hz,Ref_F0mean,Pred_F0mean,AbsErr_F0mean,RelErr_Pct,Ref_F0std,Pred_F0std,AbsErr_F0std,Accuracy_Pct,F1_Pct\n")
        for r in results_all:
            f.write(f"{r['file']},{r['sample_rate']},{r['ref_f0_mean']},{r['pred_f0_mean']},{r['abs_error_mean']},{r['rel_error_mean_pct']},{r['ref_f0_std']},{r['pred_f0_std']},{r['abs_error_std']},{r['accuracy']},{r['voiced_f1']}\n")
    print(f"Saved summary CSV: {output_csv}")
    print(f"Generated 4 figures in: {output_dir_fig}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pitch detection and evaluation on test signals.")
    parser.add_argument("--test_dir", type=str, default="TinHieuKiemThu", help="Thư mục tín hiệu kiểm thử")
    parser.add_argument("--method", type=str, default="acf", choices=["acf", "amdf"], help="Thuật toán (acf hoặc amdf)")
    parser.add_argument("--threshold_file", type=str, default=None, help="File JSON chứa ngưỡng T (mặc định tự chọn)")
    parser.add_argument("--threshold", type=float, default=None, help="Ngưỡng dự phòng nếu không có file JSON")
    parser.add_argument("--frame_len", type=float, default=25.0, help="Độ dài khung (ms)")
    parser.add_argument("--hop_len", type=float, default=10.0, help="Độ dịch khung (ms)")
    parser.add_argument("--plugins", type=str, default="none", help="Plugins: 'none', 'all', 'bandpass', 'hysteresis', 'energy_ext'.")
    parser.add_argument("--out_fig_dir", type=str, default="outputs/figures", help="Thư mục lưu hình")
    parser.add_argument("--out_json", type=str, default=None, help="File JSON kết quả (mặc định tự chọn theo method)")
    parser.add_argument("--out_csv", type=str, default=None, help="File CSV kết quả (mặc định tự chọn theo method)")

    args = parser.parse_args()
    run_testing(
        test_dir=args.test_dir,
        method=args.method,
        threshold_file=args.threshold_file,
        default_threshold=args.threshold,
        frame_duration_ms=args.frame_len,
        hop_duration_ms=args.hop_len,
        plugins=args.plugins,
        output_dir_fig=args.out_fig_dir,
        output_json=args.out_json,
        output_csv=args.out_csv,
    )
