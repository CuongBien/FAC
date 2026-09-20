"""Script 04: Xử lý toàn bộ 4 file tín hiệu kiểm thử trong TinHieuKiemThu trong 1 lần chạy duy nhất:
1. phone_F2.wav
2. phone_M2.wav
3. studio_F2.wav
4. studio_M2.wav

Tính toán F0mean, F0std, độ lệch tuyệt đối và phần trăm sai số so với *.lab.
Xuất 4 hình ảnh (mỗi file 1 hình gồm waveform + F0 contour đồng bộ thời gian) sẵn sàng trình chiếu cho giảng viên.
"""
import argparse
import glob
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
):
    """Build a detector instance, optionally wrapped with selected plugins."""
    if not plugins_arg or plugins_arg.lower() in ("none", "no", "false", "0"):
        return base_detector, []

    plugins_list = []
    p_lower = plugins_arg.lower().strip()

    if p_lower == "all":
        plugins_list = [
            BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
            HysteresisPlugin(t_high=threshold, t_low=0.40),
            EnergyExtensionPlugin(threshold_discount=0.20),
        ]
    else:
        parts = [p.strip() for p in p_lower.split(",")]
        for p in parts:
            if p in ("bandpass", "filter", "bp"):
                plugins_list.append(BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0))
            elif p in ("hysteresis", "hys", "schmitt"):
                plugins_list.append(HysteresisPlugin(t_high=threshold, t_low=0.40))
            elif p in ("energy", "energy_ext", "edge"):
                plugins_list.append(EnergyExtensionPlugin(threshold_discount=0.20))

    if not plugins_list:
        return base_detector, []

    plugin_detector = PluginPitchDetector(base_detector=base_detector, plugins=plugins_list)
    return plugin_detector, [p.name for p in plugins_list]


def run_testing(
    test_dir: str = "TinHieuKiemThu",
    threshold_file: str = "outputs/reports/threshold_acf.json",
    default_threshold: float = 0.462,
    frame_duration_ms: float = 30.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    plugins: str = "none",
    output_dir_fig: str = "outputs/figures",
    output_json: str = "outputs/reports/test_results.json",
    output_csv: str = "outputs/reports/test_summary.csv",
):
    # 1. Tải ngưỡng tối ưu T từ báo cáo huấn luyện (nếu có)
    threshold = default_threshold
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                threshold = float(t_data.get("threshold_T", default_threshold))
        except Exception:
            pass

    # 2. Khởi tạo bộ nhận diện PitchDetector (Baseline)
    base_detector = PitchDetector(
        method="acf",
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
    print(f"KIỂM THỬ THUẬT TOÁN ACF TRÊN TẬP TÍN HIỆU KIỂM THỬ ({test_dir})")
    print(f"Tham số: FrameLen={frame_duration_ms}ms, HopLen={hop_duration_ms}ms, Range=[{f0_min:.0f}, {f0_max:.0f}]Hz, Threshold T={threshold:.4f}{plugin_desc}")
    print("=" * 105)
    header = f"{'Tên file':<16} | {'Ref F0mean':<10} | {'Pred F0mean':<11} | {'|ΔF0| (Hz)':<10} | {'Lệch %':<7} | {'Ref std':<8} | {'Pred std':<8} | {'|Δstd|':<7} | {'V/UV Acc':<8} | {'F1-Score':<8}"
    print(header)
    print("-" * 105)

    for wav_path in wav_files:
        base_name = os.path.basename(wav_path)
        file_id = os.path.splitext(base_name)[0]
        lab_path = os.path.splitext(wav_path)[0] + ".lab"

        # 3. Xử lý nhận diện F0
        res = detector.process_file(wav_path)

        # 4. Đánh giá sai số so với *.lab
        eval_res = evaluate_against_ground_truth(res, lab_path)

        row = (
            f"{base_name:<16} | "
            f"{eval_res['ref_f0_mean']:<10.2f} | "
            f"{eval_res['pred_f0_mean']:<11.2f} | "
            f"{eval_res['abs_error_mean']:<10.2f} | "
            f"{eval_res['rel_error_mean_pct']:<6.2f}% | "
            f"{eval_res['ref_f0_std']:<8.2f} | "
            f"{eval_res['pred_f0_std']:<8.2f} | "
            f"{eval_res['abs_error_std']:<7.2f} | "
            f"{eval_res['classification_accuracy']:<7.2f}% | "
            f"{eval_res['voiced_f1']:<7.2f}%"
        )
        print(row)

        # 5. Xuất hình vẽ cho từng file kiểm thử (1 hình / 1 file)
        fig_name = f"04_test_enhanced_{file_id}.png" if is_enhanced else f"04_test_{file_id}.png"
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
            title=f"Đường bao tần số F0 ước lượng bằng ACF{title_suffix} - {base_name} (Fs={res['sample_rate']}Hz)",
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
            "method": "acf",
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
    parser.add_argument("--threshold_file", type=str, default="outputs/reports/threshold_acf.json", help="File JSON chứa ngưỡng T")
    parser.add_argument("--threshold", type=float, default=0.462, help="Ngưỡng dự phòng nếu không có file JSON")
    parser.add_argument("--frame_len", type=float, default=30.0, help="Độ dài khung (ms)")
    parser.add_argument("--hop_len", type=float, default=10.0, help="Độ dịch khung (ms)")
    parser.add_argument("--plugins", type=str, default="none", help="Plugins: 'none', 'all', 'bandpass', 'hysteresis', 'energy_ext' (or comma-separated).")
    parser.add_argument("--out_fig_dir", type=str, default="outputs/figures", help="Thư mục lưu hình")
    parser.add_argument("--out_json", type=str, default="outputs/reports/test_results.json", help="File JSON kết quả")
    parser.add_argument("--out_csv", type=str, default="outputs/reports/test_summary.csv", help="File CSV kết quả")

    args = parser.parse_args()
    run_testing(
        test_dir=args.test_dir,
        threshold_file=args.threshold_file,
        default_threshold=args.threshold,
        frame_duration_ms=args.frame_len,
        hop_duration_ms=args.hop_len,
        plugins=args.plugins,
        output_dir_fig=args.out_fig_dir,
        output_json=args.out_json,
        output_csv=args.out_csv,
    )
