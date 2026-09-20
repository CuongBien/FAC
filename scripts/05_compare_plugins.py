"""Script 05: Đối sánh hiệu quả của từng Plugin và các tổ hợp cải tiến:
1. Baseline (Không dùng plugin)
2. Plugin 1 (HysteresisPlugin - Ngưỡng trễ kép)
3. Plugin 2 (EnergyExtensionPlugin - Bù trễ biên theo năng lượng)
4. Plugin 3 (BandpassFilterPlugin - Lọc thông dải 70-900 Hz)
5. Kết hợp cả 3 Plugins
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
from src.plugins import (
    PluginPitchDetector,
    HysteresisPlugin,
    EnergyExtensionPlugin,
    BandpassFilterPlugin,
)
from src.analysis.evaluation import evaluate_against_ground_truth


def run_plugin_comparison(
    test_dir: str = "TinHieuKiemThu",
    output_json: str = "outputs/reports/compare_plugins.json",
    output_csv: str = "outputs/reports/compare_plugins.csv",
):
    files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]

    configs = {
        "1. Baseline (Gốc)": PitchDetector(threshold=0.462),
        "2. Plugin 1 (Hysteresis)": PluginPitchDetector(plugins=[HysteresisPlugin(t_high=0.462, t_low=0.40)]),
        "3. Plugin 2 (Energy Extension)": PluginPitchDetector(plugins=[EnergyExtensionPlugin(threshold_discount=0.20)]),
        "4. Plugin 3 (Bandpass Filter)": PluginPitchDetector(plugins=[BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)]),
        "5. Kết hợp cả 3 Plugins": PluginPitchDetector(plugins=[
            BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
            HysteresisPlugin(t_high=0.462, t_low=0.40),
            EnergyExtensionPlugin(threshold_discount=0.20),
        ]),
    }

    print("=" * 115)
    print("                 BẢNG ĐỐI SÁNH CÁC GIẢI PHÁP PLUGIN TRÊN TẬP KIỂM THỬ")
    print("=" * 115)
    header = f"{'Cấu hình':<32} | {'phone_F2 (|Δ|)':<16} | {'phone_M2 (|Δ|)':<16} | {'studio_F2 (|Δ|)':<16} | {'studio_M2 (|Δ|)':<16} | {'Sai số TB':<10} | {'F1 TB':<8}"
    print(header)
    print("-" * 115)

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
            f"{avg_f1:5.2f}%"
        )
        print(row)

        summary_records.append({
            "config_name": name,
            "average_error_hz": round(avg_err, 2),
            "average_accuracy_pct": round(avg_acc, 2),
            "average_voiced_f1_pct": round(avg_f1, 2),
            "files": file_metrics,
        })

    print("=" * 115)

    # 2. Xuất biểu đồ trực quan đối sánh Baseline vs Enhanced (Combined Plugins) cho cả 4 file
    fig_dir = "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)
    detector_base = configs["1. Baseline (Gốc)"]
    detector_enhanced = configs["5. Kết hợp cả 3 Plugins"]

    print("\nĐang xuất biểu đồ trực quan đối sánh Baseline vs Plugins...")
    generated_figs = []
    for f_id in files:
        wav_path = os.path.join(test_dir, f"{f_id}.wav")
        lab_path = os.path.join(test_dir, f"{f_id}.lab")

        res_base = detector_base.process_file(wav_path)
        res_enh = detector_enhanced.process_file(wav_path)

        ev_base = evaluate_against_ground_truth(res_base, lab_path)
        ev_enh = evaluate_against_ground_truth(res_enh, lab_path)

        fig_path = os.path.join(fig_dir, f"05_compare_plugin_{f_id}.png")
        time_sig = np.arange(len(res_base["signal"])) / res_base["sample_rate"]

        from src.visualization.plotter import plot_plugin_contour_comparison
        plot_plugin_contour_comparison(
            time_sig=time_sig,
            signal=res_base["signal"],
            time_f0=res_base["frame_times"],
            f0_baseline=res_base["f0_contour"],
            f0_enhanced=res_enh["f0_contour"],
            eval_baseline=ev_base,
            eval_enhanced=ev_enh,
            ground_truth_segments=ev_base["gt_segments"],
            file_name=f"{f_id}.wav",
            plugin_names_str="Bandpass + Hysteresis + EnergyExt",
            save_path=fig_path,
        )
        generated_figs.append(fig_path)

    # Lưu JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_json}")

    # Lưu CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Config,phone_F2_Err,phone_M2_Err,studio_F2_Err,studio_M2_Err,Avg_Err_Hz,Avg_Accuracy_Pct,Avg_F1_Pct\n")
        for rec in summary_records:
            f.write(f"{rec['config_name']},{rec['files']['phone_F2']['abs_error_mean']},{rec['files']['phone_M2']['abs_error_mean']},{rec['files']['studio_F2']['abs_error_mean']},{rec['files']['studio_M2']['abs_error_mean']},{rec['average_error_hz']},{rec['average_accuracy_pct']},{rec['average_voiced_f1_pct']}\n")
    print(f"Saved: {output_csv}")
    print(f"Generated {len(generated_figs)} comparison figures in: {fig_dir}/")
    for fp in generated_figs:
        print(f"  - {fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all plugin configurations.")
    parser.add_argument("--test_dir", type=str, default="TinHieuKiemThu", help="Test directory")
    parser.add_argument("--out_json", type=str, default="outputs/reports/compare_plugins.json", help="JSON output")
    parser.add_argument("--out_csv", type=str, default="outputs/reports/compare_plugins.csv", help="CSV output")

    args = parser.parse_args()
    run_plugin_comparison(
        test_dir=args.test_dir,
        output_json=args.out_json,
        output_csv=args.out_csv,
    )
