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
    files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]
    suite_results = []

    for name, factory in configs_dict.items():
        det = factory()
        errs, std_errs, f1s, accs = [], [], [], []
        file_metrics = {}

        for f_id in files:
            wav_path = os.path.join(test_dir, f"{f_id}.wav")
            lab_path = os.path.join(test_dir, f"{f_id}.lab")

            res = det.process_file(wav_path)
            ev = evaluate_against_ground_truth(res, lab_path)

            file_metrics[f_id] = ev
            errs.append(ev["abs_error_mean"])
            std_errs.append(ev["abs_error_std"])
            f1s.append(ev["voiced_f1"])
            accs.append(ev["classification_accuracy"])

        rec = {
            "name": name,
            "average_error_hz": round(float(np.mean(errs)), 2),
            "average_std_error": round(float(np.mean(std_errs)), 2),
            "average_voiced_f1_pct": round(float(np.mean(f1s)), 2),
            "average_accuracy_pct": round(float(np.mean(accs)), 2),
            "files": {
                f: {
                    "abs_error_mean": round(file_metrics[f]["abs_error_mean"], 2),
                    "abs_error_std": round(file_metrics[f]["abs_error_std"], 2),
                    "voiced_f1": round(file_metrics[f]["voiced_f1"], 2),
                    "accuracy": round(file_metrics[f]["classification_accuracy"], 2),
                    "pred_f0_mean": round(file_metrics[f]["pred_f0_mean"], 2),
                    "ref_f0_mean": round(file_metrics[f]["ref_f0_mean"], 2),
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
    yin_configs, champions = build_detectors()

    print("=" * 115)
    print("PHẦN 1: KHẢO SÁT & ĐÁNH GIÁ CÁC CẤU HÌNH ENHANCED TRÊN NỀN TẢNG YIN")
    print("=" * 115)
    yin_results = evaluate_suite(yin_configs)

    print(f"{'Cấu hình YIN':<30} | {'phone_F2':<10} | {'phone_M2':<10} | {'studio_F2':<10} | {'studio_M2':<10} | {'MAE TB':<8} | {'Std TB':<8} | {'F1 TB':<8} | {'Acc TB':<8}")
    print("-" * 125)
    for r in yin_results:
        f = r["files"]
        print(
            f"{r['name']:<30} | "
            f"{f['phone_F2']['abs_error_mean']:5.2f} Hz  | "
            f"{f['phone_M2']['abs_error_mean']:5.2f} Hz  | "
            f"{f['studio_F2']['abs_error_mean']:5.2f} Hz  | "
            f"{f['studio_M2']['abs_error_mean']:5.2f} Hz  | "
            f"{r['average_error_hz']:5.2f} Hz | "
            f"{r['average_std_error']:5.2f} Hz | "
            f"{r['average_voiced_f1_pct']:5.2f}% | "
            f"{r['average_accuracy_pct']:5.2f}%"
        )

    # Save YIN Enhanced report
    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/compare_yin_enhanced.json", "w", encoding="utf-8") as f:
        json.dump(yin_results, f, ensure_ascii=False, indent=2)

    with open("outputs/reports/compare_yin_enhanced.csv", "w", encoding="utf-8") as f:
        f.write("name,phone_F2_mae,phone_M2_mae,studio_F2_mae,studio_M2_mae,average_error_hz,average_std_error,average_voiced_f1_pct,average_accuracy_pct\n")
        for r in yin_results:
            f_m = r["files"]
            f.write(f'"{r["name"]}",{f_m["phone_F2"]["abs_error_mean"]},{f_m["phone_M2"]["abs_error_mean"]},{f_m["studio_F2"]["abs_error_mean"]},{f_m["studio_M2"]["abs_error_mean"]},{r["average_error_hz"]},{r["average_std_error"]},{r["average_voiced_f1_pct"]},{r["average_accuracy_pct"]}\n')

    print("\n" + "=" * 115)
    print("PHẦN 2: ĐỐI SÁNH TRỰC DIỆN 3 QUÁN QUÂN (ACF vs. AMDF vs. YIN)")
    print("=" * 115)
    champ_results = evaluate_suite(champions)

    print(f"{'Quán quân':<30} | {'phone_F2':<10} | {'phone_M2':<10} | {'studio_F2':<10} | {'studio_M2':<10} | {'MAE TB':<8} | {'Std TB':<8} | {'F1 TB':<8} | {'Acc TB':<8}")
    print("-" * 125)
    for r in champ_results:
        f = r["files"]
        print(
            f"{r['name']:<30} | "
            f"{f['phone_F2']['abs_error_mean']:5.2f} Hz  | "
            f"{f['phone_M2']['abs_error_mean']:5.2f} Hz  | "
            f"{f['studio_F2']['abs_error_mean']:5.2f} Hz  | "
            f"{f['studio_M2']['abs_error_mean']:5.2f} Hz  | "
            f"{r['average_error_hz']:5.2f} Hz | "
            f"{r['average_std_error']:5.2f} Hz | "
            f"{r['average_voiced_f1_pct']:5.2f}% | "
            f"{r['average_accuracy_pct']:5.2f}%"
        )

    # Save Champions report
    with open("outputs/reports/compare_three_champions.json", "w", encoding="utf-8") as f:
        json.dump(champ_results, f, ensure_ascii=False, indent=2)

    with open("outputs/reports/compare_three_champions.csv", "w", encoding="utf-8") as f:
        f.write("name,phone_F2_mae,phone_M2_mae,studio_F2_mae,studio_M2_mae,average_error_hz,average_std_error,average_voiced_f1_pct,average_accuracy_pct\n")
        for r in champ_results:
            f_m = r["files"]
            f.write(f'"{r["name"]}",{f_m["phone_F2"]["abs_error_mean"]},{f_m["phone_M2"]["abs_error_mean"]},{f_m["studio_F2"]["abs_error_mean"]},{f_m["studio_M2"]["abs_error_mean"]},{r["average_error_hz"]},{r["average_std_error"]},{r["average_voiced_f1_pct"]},{r["average_accuracy_pct"]}\n')

    # Generate Chart
    plot_champions_comparison(champ_results, "outputs/figures/12_compare_three_champions.png")
    print("\n[HOÀN TẤT] Toàn bộ báo cáo và biểu đồ đã được lưu trữ thành công!")


if __name__ == "__main__":
    main()
