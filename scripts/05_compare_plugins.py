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
    CenterClippingPlugin,
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

    # 1. Tải ngưỡng tối ưu cho 4 nhóm tiền xử lý: Raw, Bandpass, CenterClip, Bandpass+CenterClip
    if method == "amdf":
        t_raw = 0.4380
        t_bp = 0.3734
        t_clip = 0.6488
        t_bp_clip = 0.5628
    else:
        t_raw = 0.4408
        t_bp = 0.4892
        t_clip = 0.3352
        t_bp_clip = 0.3953

    thresh_file = "outputs/reports/threshold_all_preprocessors.json"
    if os.path.exists(thresh_file):
        try:
            with open(thresh_file, "r", encoding="utf-8") as f:
                t_all = json.load(f)
                if method in t_all:
                    t_raw = float(t_all[method]["raw"]["t_opt"])
                    t_bp = float(t_all[method]["bp"]["t_opt"])
                    t_clip = float(t_all[method]["clip"]["t_opt"])
                    t_bp_clip = float(t_all[method]["bp_clip"]["t_opt"])
        except Exception:
            pass

    # Helper tạo plugin cho các nhóm
    def make_bp():
        return BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)

    def make_clip():
        return CenterClippingPlugin(clipping_ratio=0.40, mode="standard")

    def make_hyst(t):
        return HysteresisPlugin(t_high=t, t_low=t * 0.90)

    def make_energy():
        return EnergyExtensionPlugin(threshold_discount=0.20)

    # Định nghĩa toàn bộ 16 tổ hợp (2^4) theo đúng phân phối ngưỡng tương ứng
    configs = {
        "01. Baseline (Gốc)": PitchDetector(method=method, threshold=t_raw),
        "02. [Hysteresis]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_raw),
            plugins=[make_hyst(t_raw)],
        ),
        "03. [Energy Ext]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_raw),
            plugins=[make_energy()],
        ),
        "04. [Center Clip]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_clip),
            plugins=[make_clip()],
        ),
        "05. [Bandpass Filter]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[make_bp()],
        ),
        "06. [Hyst + Energy Ext]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_raw),
            plugins=[make_hyst(t_raw), make_energy()],
        ),
        "07. [Center Clip + Hyst]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_clip),
            plugins=[make_clip(), make_hyst(t_clip)],
        ),
        "08. [Center Clip + Energy Ext]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_clip),
            plugins=[make_clip(), make_energy()],
        ),
        "09. [Bandpass + Hyst]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[make_bp(), make_hyst(t_bp)],
        ),
        "10. [Bandpass + Energy Ext]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[make_bp(), make_energy()],
        ),
        "11. [Bandpass + Center Clip]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp_clip),
            plugins=[make_bp(), make_clip()],
        ),
        "12. [Center Clip + Hyst + Energy]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_clip),
            plugins=[make_clip(), make_hyst(t_clip), make_energy()],
        ),
        "13. [Bandpass + Hyst + Energy]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp),
            plugins=[make_bp(), make_hyst(t_bp), make_energy()],
        ),
        "14. [Bandpass + Center Clip + Hyst]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp_clip),
            plugins=[make_bp(), make_clip(), make_hyst(t_bp_clip)],
        ),
        "15. [Bandpass + Center Clip + Energy]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp_clip),
            plugins=[make_bp(), make_clip(), make_energy()],
        ),
        "16. [Cả 4 Plugins]": PluginPitchDetector(
            base_detector=PitchDetector(method=method, threshold=t_bp_clip),
            plugins=[make_bp(), make_clip(), make_hyst(t_bp_clip), make_energy()],
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

    # 1. Vẽ biểu đồ xếp hạng toàn bộ 16 tổ hợp
    plot_combinations_ranking(summary_records, save_path=output_ranking_fig, title=f"Đánh giá toàn bộ 16 tổ hợp Plugins ({method.upper()}) trên tập Kiểm thử")
    print(f"Saved ranking figure: {output_ranking_fig}")

    # 2. Xuất biểu đồ trực quan đối sánh Baseline vs Tổ hợp tốt nhất
    fig_dir = "outputs/figures"
    detector_base = configs["01. Baseline (Gốc)"]
    best_name = best_err_cfg["config_name"]
    detector_best = configs[best_name]

    print(f"\nĐang xuất 4 biểu đồ đối sánh trực quan (Baseline vs. {best_name})...")
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
            plugin_names_str=f"{method.upper()}: {best_name}",
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
