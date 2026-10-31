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

from src import config
from src.core.pitch_detector import PitchDetector
from src.analysis.evaluation import (
    evaluate_against_ground_truth, round_metric, mean_metric, format_metric, csv_value,
)
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
        bp = BandpassFilterPlugin(low_cutoff=config.BANDPASS_LOW, high_cutoff=config.BANDPASS_HIGH)
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
        ste_ext = EnergyExtensionPlugin(threshold_discount=config.ENERGY_THRESHOLD_DISCOUNT)
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

    # Saved JSON is an explicit historical source, with a shared fallback.
    fallback_t = config.threshold_for_plugins(method, ("BP",) if has_bandpass else ())
    if threshold_file is None:
        suffix = "_bandpassprefilter" if has_bandpass else ""
        threshold_file = f"outputs/reports/threshold_{method}{suffix}.json"

    if default_threshold is not None:
        threshold = default_threshold
    else:
        threshold = fallback_t
    threshold_source = "explicit_fallback" if default_threshold is not None else config.THRESHOLD_SOURCE

    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                if "threshold_T" in t_data:
                    threshold = float(t_data["threshold_T"])
                    threshold_source = f"json:{threshold_file}"
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

    print("=" * 160)
    print(f"KIỂM THỬ THUẬT TOÁN {method.upper()} TRÊN TẬP TÍN HIỆU KIỂM THỬ ({test_dir})")
    print(f"Tham số: Method={method.upper()}, FrameLen={frame_duration_ms}ms, HopLen={hop_duration_ms}ms, Range=[{format_metric(f0_min, '.0f')}, {format_metric(f0_max, '.0f')}]Hz, Threshold T={format_metric(threshold, '.4f')}{plugin_desc}")
    print("=" * 160)
    header = f"{'Tên file':<14} | {'Ref F0':<8} | {'Pred F0':<8} | {'ΔF0':<6} | {'MAPE F0':<8} | {'Ref Std':<8} | {'Pred Std':<8} | {'ΔStd':<6} | {'MAPE Std':<9} | {'Ref Num':<8} | {'Pred Num':<8} | {'ΔNum':<5} | {'MAPE Num':<9} | {'SCORE':<7} | {'V/UV Acc':<8} | {'F1':<6}"
    print(header)
    print("-" * 160)

    for wav_path in wav_files:
        base_name = os.path.basename(wav_path)
        file_id = os.path.splitext(base_name)[0]
        lab_path = os.path.splitext(wav_path)[0] + ".lab"

        # 4. Chạy dự đoán cao độ
        res = detector.process_file(wav_path)

        # Đánh giá sai số so với ground truth *.lab
        eval_res = evaluate_against_ground_truth(res, lab_path)

        ref_num = eval_res.get("ref_f0_num", 0)
        pred_num = eval_res.get("pred_f0_num", 0)
        abs_err_num = eval_res.get("abs_error_num", 0)
        mape_num = eval_res.get("mape_num", 0.0)
        score = eval_res.get("final_score", eval_res.get("composite_score", 0.0))
        acc_str = f"{format_metric(eval_res['classification_accuracy'], '5.2f')}%" if eval_res['classification_accuracy'] is not None else " N/A"
        f1_str = f"{format_metric(eval_res['voiced_f1'], '5.2f')}%" if eval_res['voiced_f1'] is not None else " N/A"

        row = (
            f"{base_name:<14} | "
            f"{format_metric(eval_res['ref_f0_mean'], '<8.2f')} | "
            f"{format_metric(eval_res['pred_f0_mean'], '<8.2f')} | "
            f"{format_metric(eval_res['abs_error_mean'], '<6.2f')} | "
            f"{format_metric(eval_res['mape_f0'], '<7.2f')}% | "
            f"{format_metric(eval_res['ref_f0_std'], '<8.2f')} | "
            f"{format_metric(eval_res['pred_f0_std'], '<8.2f')} | "
            f"{format_metric(eval_res['abs_error_std'], '<6.2f')} | "
            f"{format_metric(eval_res['mape_std'], '<8.2f')}% | "
            f"{format_metric(ref_num, '<8.0f')} | "
            f"{format_metric(pred_num, '<8.0f')} | "
            f"{format_metric(abs_err_num, '<5.0f')} | "
            f"{format_metric(mape_num, '<8.2f')}% | "
            f"{format_metric(score, '<6.2f')}% | "
            f"{acc_str:<8} | "
            f"{f1_str:<6}"
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
            "duration": round_metric(res["duration"], 3),
            "ref_f0_mean": eval_res["ref_f0_mean"],
            "pred_f0_mean": eval_res["pred_f0_mean"],
            "abs_error_mean": eval_res["abs_error_mean"],
            "rel_error_mean_pct": eval_res["rel_error_mean_pct"],
            "mape_f0": eval_res["mape_f0"],
            "ref_f0_std": eval_res["ref_f0_std"],
            "pred_f0_std": eval_res["pred_f0_std"],
            "abs_error_std": eval_res["abs_error_std"],
            "rel_error_std_pct": eval_res["rel_error_std_pct"],
            "mape_std": eval_res["mape_std"],
            "ref_f0_num": ref_num,
            "pred_f0_num": pred_num,
            "abs_error_num": abs_err_num,
            "mape_num": mape_num,
            "composite_score": score,
            "final_score": score,
            "accuracy": eval_res["classification_accuracy"],
            "voiced_f1": eval_res["voiced_f1"],
            "figure_path": fig_path,
        })

    # 6. Tính trung bình sai số toàn tập kiểm thử
    avg_abs_err_mean = mean_metric([r["abs_error_mean"] for r in results_all])
    avg_mape_f0 = mean_metric([r["mape_f0"] for r in results_all])
    avg_abs_err_std = mean_metric([r["abs_error_std"] for r in results_all])
    avg_mape_std = mean_metric([r["mape_std"] for r in results_all])
    avg_abs_err_num = mean_metric([r["abs_error_num"] for r in results_all])
    avg_mape_num = mean_metric([r["mape_num"] for r in results_all])
    avg_score = mean_metric([r["final_score"] for r in results_all])

    acc_valid = [r["accuracy"] for r in results_all if r["accuracy"] is not None]
    f1_valid = [r["voiced_f1"] for r in results_all if r["voiced_f1"] is not None]
    avg_acc = mean_metric(acc_valid)
    avg_f1 = mean_metric(f1_valid)

    print("-" * 160)
    print(
        f"{'TRUNG BÌNH TOÀN TẬP':<14} | {'-':<8} | {'-':<8} | {format_metric(avg_abs_err_mean, '<6.2f')} | {format_metric(avg_mape_f0, '<7.2f')}% | "
        f"{'-':<8} | {'-':<8} | {format_metric(avg_abs_err_std, '<6.2f')} | {format_metric(avg_mape_std, '<8.2f')}% | "
        f"{'-':<8} | {'-':<8} | {format_metric(avg_abs_err_num, '<5.1f')} | {format_metric(avg_mape_num, '<8.2f')}% | "
        f"{format_metric(avg_score, '<6.2f')}% | {format_metric(avg_acc, '<7.2f')}% | {format_metric(avg_f1, '<6.2f')}%"
    )
    print("=" * 160)

    # 7. Lưu kết quả ra file JSON
    summary_data = {
        "params": {
            "method": method,
            "frame_duration_ms": frame_duration_ms,
            "hop_duration_ms": hop_duration_ms,
            "threshold_T": threshold,
            "threshold_source": threshold_source,
            "f0_min": f0_min,
            "f0_max": f0_max,
        },
        "average_metrics": {
            "mean_abs_error_f0": round_metric(avg_abs_err_mean, 2),
            "mean_mape_f0_pct": round_metric(avg_mape_f0, 2),
            "mean_abs_error_std": round_metric(avg_abs_err_std, 2),
            "mean_mape_std_pct": round_metric(avg_mape_std, 2),
            "mean_abs_error_num": round_metric(avg_abs_err_num, 2),
            "mean_mape_num_pct": round_metric(avg_mape_num, 2),
            "final_composite_score_pct": round_metric(avg_score, 2),
            "mean_v_uv_accuracy": round_metric(avg_acc, 2) if acc_valid else None,
            "mean_voiced_f1": round_metric(avg_f1, 2) if f1_valid else None,
        },
        "files": results_all,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Saved report JSON: {output_json}")

    # 8. Lưu kết quả ra file CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("File,Fs_Hz,Ref_F0mean,Pred_F0mean,AbsErr_F0mean,MAPE_F0_Pct,Ref_F0std,Pred_F0std,AbsErr_F0std,MAPE_Std_Pct,Ref_F0num,Pred_F0num,AbsErr_F0num,MAPE_Num_Pct,Final_Score_Pct,Accuracy_Pct,F1_Pct\n")
        for r in results_all:
            f.write(f"{csv_value(r['file'])},{csv_value(r['sample_rate'])},{csv_value(r['ref_f0_mean'])},{csv_value(r['pred_f0_mean'])},{csv_value(r['abs_error_mean'])},{csv_value(r['mape_f0'])},{csv_value(r['ref_f0_std'])},{csv_value(r['pred_f0_std'])},{csv_value(r['abs_error_std'])},{csv_value(r['mape_std'])},{csv_value(r['ref_f0_num'])},{csv_value(r['pred_f0_num'])},{csv_value(r['abs_error_num'])},{csv_value(r['mape_num'])},{csv_value(r['final_score'])},{csv_value(r['accuracy'])},{csv_value(r['voiced_f1'])}\n")
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
