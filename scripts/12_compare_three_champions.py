"""Script 12: Benchmark YIN Enhanced & Đối sánh Đỉnh cao 3 Quán quân (ACF vs. AMDF vs. YIN).

Nhiệm vụ:
1. Đánh giá toàn diện các biến thể Plugin Enhanced trên nền tảng YIN:
   - YIN Baseline (T=0.25)
   - YIN + Viterbi Tracking
   - YIN + Bandpass Filter
   - YIN + Energy Edge Extension
   - YIN + Bandpass + Viterbi
   - YIN + Energy + Viterbi
   - YIN + Bandpass + Energy + Viterbi
   Chọn ra Quán quân YIN (YIN Champion).

2. Đối sánh đối đầu trực diện giữa 3 Quán quân:
   - ACF Champion: Config 29 [BP + Clip + Energy + Viterbi] (MAE 1.86 Hz, F1 91.39%)
   - AMDF Champion: Config 13 [Clip + Viterbi] (MAE 1.50 Hz, F1 91.42%) / Config 05 [Energy Ext] (F1 93.83%)
   - YIN Champion: [Viterbi Tracking] (MAE 1.27 Hz, F1 92.22%) & [Bandpass Filter] (MAE 1.08 Hz)

3. Xuất biểu đồ trực quan hóa cao cấp 4 đồ thị con:
   - outputs/figures/12_compare_three_champions.png
4. Xuất báo cáo chi tiết:
   - outputs/reports/compare_three_champions.json & .csv
   - outputs/reports/compare_yin_enhanced.json & .csv
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
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
    BandpassFilterPlugin,
    CenterClippingPlugin,
    HysteresisPlugin,
    EnergyExtensionPlugin,
    ViterbiTrackingPlugin,
)
from src.analysis.evaluation import evaluate_against_ground_truth


def build_detectors():
    """Build all candidate detectors for benchmark."""
    # Enhanced YIN configurations
    yin_configs = {
        "YIN Baseline": lambda: PitchDetector(method="yin", threshold=0.25),
        "YIN + Viterbi": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[ViterbiTrackingPlugin()]
        ),
        "YIN + Bandpass": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)]
        ),
        "YIN + EnergyExt": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[EnergyExtensionPlugin(threshold_discount=0.20)]
        ),
        "YIN + BP + Viterbi": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[
                BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
                ViterbiTrackingPlugin()
            ]
        ),
        "YIN + Energy + Viterbi": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[
                EnergyExtensionPlugin(threshold_discount=0.20),
                ViterbiTrackingPlugin()
            ]
        ),
        "YIN + BP + Energy + Viterbi": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[
                BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
                EnergyExtensionPlugin(threshold_discount=0.20),
                ViterbiTrackingPlugin()
            ]
        ),
    }

    # Champions
    champions = {
        "ACF Champion (Config 29)": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="acf", threshold=0.395257),
            plugins=[
                BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
                CenterClippingPlugin(clipping_ratio=0.40),
                EnergyExtensionPlugin(threshold_discount=0.20),
                ViterbiTrackingPlugin()
            ]
        ),
        "AMDF Champion (Config 13)": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="amdf", threshold=0.648843),
            plugins=[
                CenterClippingPlugin(clipping_ratio=0.40),
                ViterbiTrackingPlugin()
            ]
        ),
        "YIN Champion (Viterbi)": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[ViterbiTrackingPlugin()]
        ),
        "YIN Ultra-Low MAE (BP)": lambda: PluginPitchDetector(
            base_detector=PitchDetector(method="yin", threshold=0.25),
            plugins=[BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)]
        ),
    }

    return yin_configs, champions


def evaluate_suite(configs_dict, test_dir="TinHieuKiemThu"):
    """Evaluate a dictionary of detector factory functions across test files."""
    wav_paths = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    files = [os.path.splitext(os.path.basename(p))[0] for p in wav_paths]
    if not files:
        files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]
    suite_results = []

    for name, factory in configs_dict.items():
        det = factory()
        errs, std_errs, f1s, accs = [], [], [], []
        mapes_f0, mapes_std, mapes_num, num_errs = [], [], [], []
        file_metrics = {}

        for f_id in files:
            wav_path = os.path.join(test_dir, f"{f_id}.wav")
            lab_path = os.path.join(test_dir, f"{f_id}.lab")

            res = det.process_file(wav_path)
            ev = evaluate_against_ground_truth(res, lab_path)

            file_metrics[f_id] = ev
            errs.append(ev["abs_error_mean"])
            std_errs.append(ev["abs_error_std"])
            mapes_f0.append(ev["mape_f0"])
            mapes_std.append(ev["mape_std"])
            if ev.get("mape_num") is not None:
                mapes_num.append(ev["mape_num"])
                num_errs.append(ev["abs_error_num"])
            if ev.get("voiced_f1") is not None:
                f1s.append(ev["voiced_f1"])
            if ev.get("classification_accuracy") is not None:
                accs.append(ev["classification_accuracy"])

        mean_mape_f0 = float(np.mean(mapes_f0))
        mean_mape_std = float(np.mean(mapes_std))
        mean_mape_num = float(np.mean(mapes_num)) if mapes_num else None
        if mean_mape_num is not None:
            final_composite = (mean_mape_f0 + mean_mape_std + mean_mape_num) / 3.0
        else:
            final_composite = (mean_mape_f0 + mean_mape_std) / 2.0

        rec = {
            "name": name,
            "dataset": test_dir,
            "average_error_hz": round(float(np.mean(errs)), 2),
            "average_std_error": round(float(np.mean(std_errs)), 2),
            "average_num_error": round(float(np.mean(num_errs)), 2) if num_errs else None,
            "mape_f0_pct": round(mean_mape_f0, 2),
            "mape_std_pct": round(mean_mape_std, 2),
            "mape_num_pct": round(mean_mape_num, 2) if mean_mape_num is not None else None,
            "final_composite_score_pct": round(final_composite, 2),
            "average_voiced_f1_pct": round(float(np.mean(f1s)), 2) if f1s else None,
            "average_accuracy_pct": round(float(np.mean(accs)), 2) if accs else None,
            "files": {
                f: {
                    "abs_error_mean": round(file_metrics[f]["abs_error_mean"], 2),
                    "mape_f0": round(file_metrics[f]["mape_f0"], 2),
                    "abs_error_std": round(file_metrics[f]["abs_error_std"], 2),
                    "mape_std": round(file_metrics[f]["mape_std"], 2),
                    "pred_f0_num": file_metrics[f].get("pred_f0_num"),
                    "ref_f0_num": file_metrics[f].get("ref_f0_num"),
                    "abs_error_num": file_metrics[f].get("abs_error_num"),
                    "mape_num": file_metrics[f].get("mape_num"),
                    "composite_score": file_metrics[f].get("composite_score"),
                    "voiced_f1": file_metrics[f].get("voiced_f1"),
                    "accuracy": file_metrics[f].get("classification_accuracy"),
                    "pred_f0_mean": round(file_metrics[f]["pred_f0_mean"], 2),
                    "ref_f0_mean": round(file_metrics[f]["ref_f0_mean"], 2),
                    "pred_f0_std": round(file_metrics[f]["pred_f0_std"], 2),
                    "ref_f0_std": round(file_metrics[f]["ref_f0_std"], 2),
                }
                for f in files
            }
        }
        suite_results.append(rec)

    return suite_results


def plot_champions_comparison(records, output_fig_path):
    """Plot comprehensive 4-panel comparison for Champions."""
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]
    file_labels = ["Phone F2", "Phone M2", "Studio F2", "Studio M2", "Trung bình"]

    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]
    names = [r["name"] for r in records]

    # --- Subplot 1: MAE across files ---
    ax1 = axes[0, 0]
    x = np.arange(len(file_labels))
    width = 0.18

    for i, r in enumerate(records):
        vals = [r["files"][f]["abs_error_mean"] for f in files] + [r["average_error_hz"]]
        offset = (i - (len(records) - 1) / 2) * width
        bars = ax1.bar(x + offset, vals, width, label=r["name"], color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=8, rotation=35)

    ax1.set_title("1. Sai số tuyệt đối trung bình F0 (MAE - Hz) - Càng thấp càng tốt", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(file_labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Sai số F0 (Hz)", fontsize=11)
    ax1.set_ylim(0, max([r["files"]["phone_F2"]["abs_error_mean"] for r in records]) * 1.35)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 2: Voiced F1 Score (%) ---
    ax2 = axes[0, 1]
    for i, r in enumerate(records):
        vals = [r["files"][f]["voiced_f1"] for f in files] + [r["average_voiced_f1_pct"]]
        offset = (i - (len(records) - 1) / 2) * width
        bars = ax2.bar(x + offset, vals, width, label=r["name"], color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, rotation=35)

    ax2.set_title("2. Điểm Voiced F1 Score (%) - Phân loại V/UV - Càng cao càng tốt", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(file_labels, fontsize=10, fontweight="bold")
    ax2.set_ylabel("F1 Score (%)", fontsize=11)
    ax2.set_ylim(70, 102)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 3: Std Error (Hz) ---
    ax3 = axes[1, 0]
    for i, r in enumerate(records):
        vals = [r["files"][f]["abs_error_std"] for f in files] + [r["average_std_error"]]
        offset = (i - (len(records) - 1) / 2) * width
        bars = ax3.bar(x + offset, vals, width, label=r["name"], color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=8, rotation=35)

    ax3.set_title("3. Độ lệch chuẩn sai số (Std Err - Hz) - Độ ổn định cao độ", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(file_labels, fontsize=10, fontweight="bold")
    ax3.set_ylabel("Độ lệch chuẩn (Hz)", fontsize=11)
    ax3.set_ylim(0, max([r["average_std_error"] for r in records]) * 1.5)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 4: Summary Comparison (Normalized Performance Radar or Grouped Bar) ---
    ax4 = axes[1, 1]
    metrics = ["MAE (Hz)", "Std Err (Hz)", "F1 Score (%)", "Accuracy (%)"]
    c_names = [r["name"].split("(")[0].strip() for r in records]
    
    # Overview Table-like visual inside ax4
    ax4.axis("off")
    table_data = []
    headers = ["Quán quân", "MAE TB (Hz)", "Std Err (Hz)", "F1 Voiced (%)", "Accuracy (%)"]
    for r in records:
        table_data.append([
            r["name"],
            f"{r['average_error_hz']:.2f}",
            f"{r['average_std_error']:.2f}",
            f"{r['average_voiced_f1_pct']:.2f}%",
            f"{r['average_accuracy_pct']:.2f}%",
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.38, 0.15, 0.15, 0.16, 0.16]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    # Style header row
    for col in range(len(headers)):
        table[(0, col)].set_facecolor("#2c3e50")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    # Style champion rows
    for row in range(1, len(table_data) + 1):
        bg = "#eaf2f8" if (row % 2 == 1) else "#ffffff"
        if "YIN Champion" in table_data[row - 1][0]:
            bg = "#d4efdf"  # highlight YIN champion
        for col in range(len(headers)):
            table[(row, col)].set_facecolor(bg)
            if "YIN Champion" in table_data[row - 1][0] and col == 0:
                table[(row, col)].set_text_props(fontweight="bold", color="#196f3d")

    ax4.set_title("4. Tổng kết Đối sánh Các Quán Quân Tuyệt Đối", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle("ĐỐI SÁNH TRỰC DIỆN TAM ĐẠI QUÁN QUÂN: ACF vs. AMDF vs. YIN", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved figure to: {output_fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Đối sánh các hệ thống SOTA Champions.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        help="Chọn tập dữ liệu: kiemthu, huanluyen, both, hoặc đường dẫn tới thư mục chứa file .wav/.lab",
    )
    args = parser.parse_args()

    yin_configs, champions = build_detectors()

    datasets = []
    if args.dataset == "huanluyen":
        datasets.append(("TinHieuHuanLuyen", "TẬP HUẤN LUYỆN (TinHieuHuanLuyen)"))
    elif args.dataset == "kiemthu":
        datasets.append(("TinHieuKiemThu", "TẬP KIỂM THỬ (TinHieuKiemThu)"))
    elif args.dataset == "both":
        datasets.append(("TinHieuHuanLuyen", "TẬP HUẤN LUYỆN (TinHieuHuanLuyen)"))
        datasets.append(("TinHieuKiemThu", "TẬP KIỂM THỬ (TinHieuKiemThu)"))
    elif os.path.isdir(args.dataset):
        datasets.append((args.dataset, f"TẬP DỮ LIỆU TÙY CHỌN ({args.dataset})"))
    else:
        print(f"[!] Thư mục hoặc tùy chọn không hợp lệ: {args.dataset}")
        sys.exit(1)

    for d_dir, d_title in datasets:
        print("\n" + "=" * 160)
        print(f"  {d_title} - ĐỐI SÁNH CÁC HỆ THỐNG SOTA CHAMPIONS (F0, STD & F0NUM - KÈM MAPE & FINAL SCORE)")
        print("=" * 160)

        champ_results = evaluate_suite(champions, test_dir=d_dir)

        # Print header
        print(f"{'Hệ thống Quán quân':<28} | {'MAE F0':<9} | {'MAPE F0':<9} | {'ΔStd':<9} | {'MAPE Std':<9} | {'ΔNum (khung)':<13} | {'MAPE Num':<9} | {'FINAL SCORE':<12} | {'F1-Score':<9} | {'V/UV Acc':<9}")
        print("-" * 160)
        for r in champ_results:
            num_err_str = f"{r['average_num_error']:5.1f}" if r['average_num_error'] is not None else "   N/A"
            mape_num_str = f"{r['mape_num_pct']:5.2f}%" if r['mape_num_pct'] is not None else "   N/A"
            f1_str = f"{r['average_voiced_f1_pct']:5.2f}%" if r['average_voiced_f1_pct'] is not None else "   N/A"
            acc_str = f"{r['average_accuracy_pct']:5.2f}%" if r['average_accuracy_pct'] is not None else "   N/A"
            final_score_str = f"{r['final_composite_score_pct']:5.2f}%"

            print(
                f"{r['name']:<28} | "
                f"{r['average_error_hz']:5.2f} Hz | "
                f"{r['mape_f0_pct']:5.2f}%    | "
                f"{r['average_std_error']:5.2f} Hz | "
                f"{r['mape_std_pct']:5.2f}%    | "
                f"{num_err_str:<13} | "
                f"{mape_num_str:<9} | "
                f"{final_score_str:<12} | "
                f"{f1_str:<9} | "
                f"{acc_str:<9}"
            )

        # Print breakdown per file
        print("\n  Chi tiết từng file:")
        for r in champ_results:
            print(f"  * {r['name']}:")
            for fname, fmetrics in r["files"].items():
                num_info = ""
                if fmetrics.get("ref_f0_num") is not None:
                    num_info = f" | Num: Pred={fmetrics['pred_f0_num']} vs Ref={fmetrics['ref_f0_num']} (Δ={fmetrics['abs_error_num']}, MAPE={fmetrics['mape_num']:.1f}%)"
                comp_info = ""
                if fmetrics.get("composite_score") is not None:
                    comp_info = f" | Score={fmetrics['composite_score']:.2f}%"
                f1_info = f" | F1: {fmetrics['voiced_f1']:.1f}%" if fmetrics.get("voiced_f1") is not None else ""
                print(f"    - {fname:<12}: F0 Pred={fmetrics['pred_f0_mean']:.1f} vs Ref={fmetrics['ref_f0_mean']:.1f}Hz (Δ={fmetrics['abs_error_mean']:.2f}Hz, MAPE={fmetrics['mape_f0']:.2f}%) | Std: Pred={fmetrics['pred_f0_std']:.1f} vs Ref={fmetrics['ref_f0_std']:.1f}Hz (Δ={fmetrics['abs_error_std']:.2f}Hz, MAPE={fmetrics['mape_std']:.2f}%){num_info}{comp_info}{f1_info}")

        # Save Champions report for this dataset
        if d_dir == "TinHieuHuanLuyen":
            prefix_out = "huanluyen_"
        elif d_dir == "TinHieuKiemThu":
            prefix_out = ""
        else:
            safe_d = os.path.basename(os.path.normpath(d_dir)).lower()
            prefix_out = f"{safe_d}_"

        os.makedirs("outputs/reports", exist_ok=True)
        with open(f"outputs/reports/{prefix_out}compare_three_champions.json", "w", encoding="utf-8") as f:
            json.dump(champ_results, f, ensure_ascii=False, indent=2)

        with open(f"outputs/reports/{prefix_out}compare_three_champions.csv", "w", encoding="utf-8") as f:
            f.write("name,dataset,average_error_hz,mape_f0_pct,average_std_error,mape_std_pct,average_num_error,mape_num_pct,final_composite_score_pct,average_voiced_f1_pct,average_accuracy_pct\n")
            for r in champ_results:
                f.write(f'"{r["name"]}","{d_dir}",{r["average_error_hz"]},{r["mape_f0_pct"]},{r["average_std_error"]},{r["mape_std_pct"]},{r.get("average_num_error","")},{r.get("mape_num_pct","")},{r["final_composite_score_pct"]},{r.get("average_voiced_f1_pct","")},{r.get("average_accuracy_pct","")}\n')

        # If testing dataset, also update the chart
        if d_dir == "TinHieuKiemThu":
            plot_champions_comparison(champ_results, "outputs/figures/12_compare_three_champions.png")

    print("\n[HOÀN TẤT] Toàn bộ báo cáo và biểu đồ Champions đã được lưu trữ thành công!")


if __name__ == "__main__":
    main()
