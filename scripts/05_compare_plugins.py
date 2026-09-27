"""Script 05: Đánh giá và đối sánh TOÀN BỘ 8 TỔ HỢP PLUGINS trên tập kiểm thử TinHieuKiemThu:
1. Baseline (Không dùng plugin)
2. Plugin 1 (Hysteresis)
3. Plugin 2 (Energy Extension)
4. Plugin 3 (Bandpass Filter)
5. Tổ hợp [Hysteresis + Energy Extension]
6. Tổ hợp [Bandpass + Hysteresis]
7. Tổ hợp [Bandpass + Energy Extension]
8. Tổ hợp [Cả 3 Plugins: Bandpass + Hysteresis + Energy Extension]
"""
import argparse
import json
import os
import sys
from typing import Optional
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
from src.plugins import (
    PluginPitchDetector,
    HysteresisPlugin,
    EnergyExtensionPlugin,
    BandpassFilterPlugin,
)
from src.analysis.evaluation import evaluate_against_ground_truth
from src.visualization.plotter import plot_plugin_contour_comparison, plot_combinations_ranking


def run_all_combinations(
    test_dir: str = "TinHieuKiemThu",
    method: str = "acf",
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_ranking_fig: Optional[str] = None,
):
    method = method.lower()
    files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]

    if output_json is None:
        output_json = f"outputs/reports/compare_plugins_{method}.json" if method != "acf" else "outputs/reports/compare_plugins.json"
    if output_csv is None:
        output_csv = f"outputs/reports/compare_plugins_{method}.csv" if method != "acf" else "outputs/reports/compare_plugins.csv"
    if output_ranking_fig is None:
        output_ranking_fig = f"outputs/figures/05_all_combinations_ranking_{method}.png" if method != "acf" else "outputs/figures/05_all_combinations_ranking.png"

    # 1. Tải ngưỡng tối ưu tương ứng: Baseline (chưa lọc) vs Bandpass (đã lọc)
    if method == "amdf":
        t_base_file = "outputs/reports/threshold_amdf.json"
        t_bp_file = "outputs/reports/threshold_amdf_bandpassprefilter.json"
        t_base = 0.4380
        t_bp = 0.3734
        hyst_t_low_base = 0.380
        hyst_t_low_bp = 0.320
    else:
        t_base_file = "outputs/reports/threshold_acf.json"
        t_bp_file = "outputs/reports/threshold_acf_bandpassprefilter.json"
        t_base = 0.4408
        t_bp = 0.4892
        hyst_t_low_base = 0.400
        hyst_t_low_bp = 0.420

    if os.path.exists(t_base_file):
        try:
            with open(t_base_file, "r", encoding="utf-8") as f:
                t_base = float(json.load(f).get("threshold_T", t_base))
        except Exception:
            pass

    if os.path.exists(t_bp_file):
        try:
            with open(t_bp_file, "r", encoding="utf-8") as f:
                t_bp = float(json.load(f).get("threshold_T", t_bp))
        except Exception:
            pass

    # Định nghĩa toàn bộ 8 tổ hợp (2^3) theo đúng phân phối ngưỡng
    configs = {
        "1. Baseline (Gốc)": PitchDetector(method=method, threshold=t_base),
        "2. [Hysteresis]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_base),
            plugins=[HysteresisPlugin(t_high=t_base, t_low=hyst_t_low_base)],
        ),
        "3. [Energy Ext]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_base),
            plugins=[EnergyExtensionPlugin(threshold_discount=0.20)],
        ),
        "4. [Bandpass Filter]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)],
        ),
        "5. [Hysteresis + Energy Ext]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_base),
            plugins=[
                HysteresisPlugin(t_high=t_base, t_low=hyst_t_low_base),
                EnergyExtensionPlugin(threshold_discount=0.20),
            ],
        ),
        "6. [Bandpass + Hysteresis]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[
                BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
                HysteresisPlugin(t_high=t_bp, t_low=hyst_t_low_bp),
            ],
        ),
        "7. [Bandpass + Energy Ext]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[
                BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
                EnergyExtensionPlugin(threshold_discount=0.20),
            ],
        ),
        "8. [Cả 3 Plugins]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[
                BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
                HysteresisPlugin(t_high=t_bp, t_low=hyst_t_low_bp),
                EnergyExtensionPlugin(threshold_discount=0.20),
            ],
        ),
    }

    print("=" * 125)
    print(f"       BẢNG ĐỐI SÁNH TOÀN BỘ 8 TỔ HỢP PLUGINS TRÊN TẬP KIỂM THỬ (METHOD = {method.upper()})")
    print("=" * 125)
    header = f"{'Cấu hình':<32} | {'phone_F2 (|Δ|)':<16} | {'phone_M2 (|Δ|)':<16} | {'studio_F2 (|Δ|)':<16} | {'studio_M2 (|Δ|)':<16} | {'Sai số TB':<10} | {'F1 TB':<8} | {'Acc TB':<8}"
    print(header)
    print("-" * 125)

    summary_records = []

    for name, detector in configs.items():
        errs = []
        f1s = []
        accs = []
        file_metrics = {}

        for f_id in files:
            wav_path = os.path.join(test_dir, f"{f_id}.wav")
            lab_path = os.path.join(test_dir, f"{f_id}.lab")

            res = detector.process_file(wav_path)
            ev = evaluate_against_ground_truth(res, lab_path)

            file_metrics[f_id] = {
                "pred_f0_mean": ev["pred_f0_mean"],
                "abs_error_mean": ev["abs_error_mean"],
                "rel_error_mean_pct": ev["rel_error_mean_pct"],
                "accuracy": ev["classification_accuracy"],
                "voiced_f1": ev["voiced_f1"],
            }
            errs.append(ev["abs_error_mean"])
            f1s.append(ev["voiced_f1"])
            accs.append(ev["classification_accuracy"])

        avg_err = float(np.mean(errs))
        avg_f1 = float(np.mean(f1s))
        avg_acc = float(np.mean(accs))

        row = (
            f"{name:<32} | "
            f"{file_metrics['phone_F2']['abs_error_mean']:5.2f}Hz ({file_metrics['phone_F2']['rel_error_mean_pct']:4.1f}%) | "
            f"{file_metrics['phone_M2']['abs_error_mean']:5.2f}Hz ({file_metrics['phone_M2']['rel_error_mean_pct']:4.1f}%) | "
            f"{file_metrics['studio_F2']['abs_error_mean']:5.2f}Hz ({file_metrics['studio_F2']['rel_error_mean_pct']:4.1f}%) | "
            f"{file_metrics['studio_M2']['abs_error_mean']:5.2f}Hz ({file_metrics['studio_M2']['rel_error_mean_pct']:4.1f}%) | "
            f"{avg_err:5.2f} Hz    | "
            f"{avg_f1:5.2f}% | "
            f"{avg_acc:5.2f}%"
        )
        print(row)

        summary_records.append({
            "config_name": name,
            "average_error_hz": round(avg_err, 2),
            "average_accuracy_pct": round(avg_acc, 2),
            "average_voiced_f1_pct": round(avg_f1, 2),
            "files": file_metrics,
        })

    print("=" * 125)

    # Tìm tổ hợp tốt nhất
    best_err_cfg = min(summary_records, key=lambda x: x["average_error_hz"])
    best_f1_cfg = max(summary_records, key=lambda x: x["average_voiced_f1_pct"])

    print(f"[*] Cấu hình có SAI SỐ THẤP NHẤT: {best_err_cfg['config_name']} (Sai số TB: {best_err_cfg['average_error_hz']} Hz)")
    print(f"[*] Cấu hình có F1-SCORE CAO NHẤT : {best_f1_cfg['config_name']} (F1 TB: {best_f1_cfg['average_voiced_f1_pct']}%, Acc: {best_f1_cfg['average_accuracy_pct']}%)")
    print("-" * 125)

    # 1. Vẽ biểu đồ xếp hạng toàn bộ 8 tổ hợp
    plot_combinations_ranking(summary_records, save_path=output_ranking_fig)
    print(f"Saved ranking figure: {output_ranking_fig}")

    # 2. Xuất biểu đồ trực quan đối sánh Baseline vs Tổ hợp tốt nhất (Cả 3 Plugins)
    fig_dir = "outputs/figures"
    detector_base = configs["1. Baseline (Gốc)"]
    detector_best = configs["8. [Cả 3 Plugins]"]

    print("\nĐang xuất 4 biểu đồ đối sánh trực quan (Baseline vs. Best Combination)...")
    for f_id in files:
        wav_path = os.path.join(test_dir, f"{f_id}.wav")
        lab_path = os.path.join(test_dir, f"{f_id}.lab")

        res_base = detector_base.process_file(wav_path)
        res_best = detector_best.process_file(wav_path)

        ev_base = evaluate_against_ground_truth(res_base, lab_path)
        ev_best = evaluate_against_ground_truth(res_best, lab_path)

        fig_prefix = f"05_compare_plugin_{method}_{f_id}.png" if method != "acf" else f"05_compare_plugin_{f_id}.png"
        fig_path = os.path.join(fig_dir, fig_prefix)
        time_sig = np.arange(len(res_base["signal"])) / res_base["sample_rate"]

        plot_plugin_contour_comparison(
            time_sig=time_sig,
            signal=res_base["signal"],
            time_f0=res_base["frame_times"],
            f0_baseline=res_base["f0_contour"],
            f0_enhanced=res_best["f0_contour"],
            eval_baseline=ev_base,
            eval_enhanced=ev_best,
            ground_truth_segments=ev_base["gt_segments"],
            file_name=f"{f_id}.wav",
            plugin_names_str=f"{method.upper()} (Bandpass + Hysteresis + EnergyExt)",
            save_path=fig_path,
        )

    # Lưu JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {output_json}")

    # Lưu CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Config,phone_F2_Err,phone_M2_Err,studio_F2_Err,studio_M2_Err,Avg_Err_Hz,Avg_Accuracy_Pct,Avg_F1_Pct\n")
        for rec in summary_records:
            f.write(f"{rec['config_name']},{rec['files']['phone_F2']['abs_error_mean']},{rec['files']['phone_M2']['abs_error_mean']},{rec['files']['studio_F2']['abs_error_mean']},{rec['files']['studio_M2']['abs_error_mean']},{rec['average_error_hz']},{rec['average_accuracy_pct']},{rec['average_voiced_f1_pct']}\n")
    print(f"Saved report: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all 8 plugin combinations.")
    parser.add_argument("--test_dir", type=str, default="TinHieuKiemThu", help="Test directory")
    parser.add_argument("--method", type=str, default="acf", choices=["acf", "amdf"], help="Pitch detection method (acf or amdf)")
    parser.add_argument("--out_json", type=str, default=None, help="JSON output")
    parser.add_argument("--out_csv", type=str, default=None, help="CSV output")
    parser.add_argument("--out_fig", type=str, default=None, help="Ranking plot")

    args = parser.parse_args()
    run_all_combinations(
        test_dir=args.test_dir,
        method=args.method,
        output_json=args.out_json,
        output_csv=args.out_csv,
        output_ranking_fig=args.out_fig,
    )
