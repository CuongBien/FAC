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
import itertools
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
    ViterbiTrackingPlugin,
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

    def make_viterbi():
        return ViterbiTrackingPlugin(w_freq=3.0, w_octave=2.0)

    plugin_defs = [
        ("BP", "Bandpass Filter"),
        ("Clip", "Center Clip"),
        ("Hyst", "Hysteresis"),
        ("Energy", "Energy Ext"),
        ("Viterbi", "Viterbi Tracking"),
    ]

    # Sinh tự động toàn bộ 32 tổ hợp (2^5) theo bậc k = 0..5
    configs = {}
    combo_idx = 1
    for k in range(6):
        for combo in itertools.combinations(plugin_defs, k):
            short_names = [p[0] for p in combo]
            has_bp = "BP" in short_names
            has_clip = "Clip" in short_names
            has_hyst = "Hyst" in short_names
            has_energy = "Energy" in short_names
            has_viterbi = "Viterbi" in short_names

            # Chọn ngưỡng tối ưu theo phân phối tiền xử lý
            if has_bp and has_clip:
                t = t_bp_clip
            elif has_bp:
                t = t_bp
            elif has_clip:
                t = t_clip
            else:
                t = t_raw

            plugins = []
            if has_bp:
                plugins.append(make_bp())
            if has_clip:
                plugins.append(make_clip())
            if has_hyst:
                plugins.append(make_hyst(t))
            if has_energy:
                plugins.append(make_energy())
            if has_viterbi:
                plugins.append(make_viterbi())

            if k == 0:
                name = f"{combo_idx:02d}. Baseline (Gốc)"
                detector = PitchDetector(method=method, threshold=t)
            elif k == 5:
                name = f"{combo_idx:02d}. [Cả 5 Plugins]"
                detector = PluginPitchDetector(
                    base_detector=PitchDetector(method=method, threshold=t),
                    plugins=plugins,
                )
            elif k == 1:
                name = f"{combo_idx:02d}. [{combo[0][1]}]"
                detector = PluginPitchDetector(
                    base_detector=PitchDetector(method=method, threshold=t),
                    plugins=plugins,
                )
            else:
                name = f"{combo_idx:02d}. [{' + '.join(short_names)}]"
                detector = PluginPitchDetector(
                    base_detector=PitchDetector(method=method, threshold=t),
                    plugins=plugins,
                )

            configs[name] = (detector, t)
            combo_idx += 1

    print("=" * 148)
    print(f"       BẢNG ĐỐI SÁNH TOÀN BỘ 32 TỔ HỢP PLUGINS TRÊN TẬP KIỂM THỬ (METHOD = {method.upper()})")
    print("=" * 148)
    header = f"{'Cấu hình':<44} | {'phone_F2 (|Δ|)':<16} | {'phone_M2 (|Δ|)':<16} | {'studio_F2 (|Δ|)':<16} | {'studio_M2 (|Δ|)':<16} | {'Sai số TB':<10} | {'Std TB':<8} | {'F1 TB':<8} | {'Acc TB':<8}"
    print(header)
    print("-" * 148)

    summary_records = []

    for name, (detector, t_val) in configs.items():
        errs = []
        std_errs = []
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
                "pred_f0_std": ev["pred_f0_std"],
                "abs_error_std": ev["abs_error_std"],
                "rel_error_std_pct": ev["rel_error_std_pct"],
                "accuracy": ev["classification_accuracy"],
                "voiced_f1": ev["voiced_f1"],
            }
            errs.append(ev["abs_error_mean"])
            std_errs.append(ev["abs_error_std"])
            f1s.append(ev["voiced_f1"])
            accs.append(ev["classification_accuracy"])

        avg_err = float(np.mean(errs))
        avg_std_err = float(np.mean(std_errs))
        avg_f1 = float(np.mean(f1s))
        avg_acc = float(np.mean(accs))

        row = (
            f"{name:<44} | "
            f"{file_metrics['phone_F2']['abs_error_mean']:5.2f}Hz ({file_metrics['phone_F2']['rel_error_mean_pct']:4.1f}%) | "
            f"{file_metrics['phone_M2']['abs_error_mean']:5.2f}Hz ({file_metrics['phone_M2']['rel_error_mean_pct']:4.1f}%) | "
            f"{file_metrics['studio_F2']['abs_error_mean']:5.2f}Hz ({file_metrics['studio_F2']['rel_error_mean_pct']:4.1f}%) | "
            f"{file_metrics['studio_M2']['abs_error_mean']:5.2f}Hz ({file_metrics['studio_M2']['rel_error_mean_pct']:4.1f}%) | "
            f"{avg_err:5.2f} Hz    | "
            f"{avg_std_err:5.2f}   | "
            f"{avg_f1:5.2f}% | "
            f"{avg_acc:5.2f}%"
        )
        print(row)

        summary_records.append({
            "config_name": name,
            "threshold": round(float(t_val), 4),
            "average_error_hz": round(avg_err, 2),
            "average_std_error": round(avg_std_err, 2),
            "average_accuracy_pct": round(avg_acc, 2),
            "average_voiced_f1_pct": round(avg_f1, 2),
            "files": file_metrics,
        })

    print("=" * 148)

    # Tìm tổ hợp tốt nhất
    best_err_cfg = min(summary_records, key=lambda x: x["average_error_hz"])
    best_f1_cfg = max(summary_records, key=lambda x: x["average_voiced_f1_pct"])

    print(f"[*] Cấu hình có SAI SỐ THẤP NHẤT: {best_err_cfg['config_name']} (Sai số TB: {best_err_cfg['average_error_hz']} Hz)")
    print(f"[*] Cấu hình có F1-SCORE CAO NHẤT : {best_f1_cfg['config_name']} (F1 TB: {best_f1_cfg['average_voiced_f1_pct']}%, Acc: {best_f1_cfg['average_accuracy_pct']}%)")
    print("-" * 137)

    # 1. Vẽ biểu đồ xếp hạng toàn bộ các cấu hình
    plot_combinations_ranking(summary_records, save_path=output_ranking_fig, title=f"Đánh giá toàn bộ 32 tổ hợp Plugins ({method.upper()}) trên tập Kiểm thử")
    print(f"Saved ranking figure: {output_ranking_fig}")

    # 2. Xuất biểu đồ trực quan đối sánh Baseline vs Tổ hợp tốt nhất
    fig_dir = "outputs/figures"
    detector_base = configs["01. Baseline (Gốc)"][0]
    best_name = best_err_cfg["config_name"]
    detector_best = configs[best_name][0]

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
        f.write("Config,Threshold,phone_F2_Err,phone_M2_Err,studio_F2_Err,studio_M2_Err,Avg_Err_Hz,Avg_Std_Err,Avg_Accuracy_Pct,Avg_F1_Pct\n")
        for rec in summary_records:
            t_str = f"{rec.get('threshold', 0.0):.4f}"
            f.write(f"{rec['config_name']},{t_str},{rec['files']['phone_F2']['abs_error_mean']},{rec['files']['phone_M2']['abs_error_mean']},{rec['files']['studio_F2']['abs_error_mean']},{rec['files']['studio_M2']['abs_error_mean']},{rec['average_error_hz']},{rec.get('average_std_error', 0.0)},{rec['average_accuracy_pct']},{rec['average_voiced_f1_pct']}\n")
    print(f"Saved report: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all 32 plugin combinations.")
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
