"""Script 11: Benchmark và đối sánh thuật toán YIN với ACF và AMDF.

Khảo sát Tam giác Thuật toán miền Thời gian:
1. ACF (Autocorrelation Function - Đỉnh tương quan)
2. AMDF (Average Magnitude Difference Function - Đáy hiệu độ lớn)
3. YIN (Squared Difference + CMNDF + Absolute Threshold + Parabolic Interpolation)
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
from src.analysis.evaluation import evaluate_against_ground_truth


def run_yin_benchmark(
    test_dir: str = "TinHieuKiemThu",
    output_json: str = "outputs/reports/compare_yin.json",
    output_csv: str = "outputs/reports/compare_yin.csv",
    output_fig: str = "outputs/figures/11_compare_acf_amdf_yin.png",
):
    files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]

    # Khởi tạo 3 detectors đại diện cho 3 thuật toán nền tảng
    detectors = {
        "ACF (Baseline)": PitchDetector(method="acf", threshold=0.4408),
        "AMDF (Baseline)": PitchDetector(method="amdf", threshold=0.4380),
        "YIN (Baseline, T=0.25)": PitchDetector(method="yin", threshold=0.25),
    }

    results = {}
    print("=" * 115)
    print("       ĐỐI SÁNH HIỆU NĂNG TAM GIÁC THUẬT TOÁN MIỀN THỜI GIAN: ACF vs. AMDF vs. YIN")
    print("=" * 115)
    print(f"{'Thuật toán':<24} | {'phone_F2':<12} | {'phone_M2':<12} | {'studio_F2':<12} | {'studio_M2':<12} | {'Sai số TB':<10} | {'F1 TB':<8} | {'Acc TB':<8}")
    print("-" * 115)

    summary_records = []

    for name, det in detectors.items():
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

        avg_err = float(np.mean(errs))
        avg_std = float(np.mean(std_errs))
        avg_f1 = float(np.mean(f1s))
        avg_acc = float(np.mean(accs))

        print(
            f"{name:<24} | "
            f"{errs[0]:5.2f} Hz    | "
            f"{errs[1]:5.2f} Hz    | "
            f"{errs[2]:5.2f} Hz    | "
            f"{errs[3]:5.2f} Hz    | "
            f"{avg_err:5.2f} Hz  | "
            f"{avg_f1:5.2f}% | "
            f"{avg_acc:5.2f}%"
        )

        rec = {
            "algorithm": name,
            "threshold": det.threshold,
            "average_error_hz": round(avg_err, 2),
            "average_std_error": round(avg_std, 2),
            "average_voiced_f1_pct": round(avg_f1, 2),
            "average_accuracy_pct": round(avg_acc, 2),
            "files": {
                f: {
                    "pred_mean": file_metrics[f]["pred_f0_mean"],
                    "abs_err_mean": file_metrics[f]["abs_error_mean"],
                    "rel_err_pct": file_metrics[f]["rel_error_mean_pct"],
                    "pred_std": file_metrics[f]["pred_f0_std"],
                    "abs_err_std": file_metrics[f]["abs_error_std"],
                    "accuracy": file_metrics[f]["classification_accuracy"],
                    "f1": file_metrics[f]["voiced_f1"],
                }
                for f in files
            },
        }
        summary_records.append(rec)

    print("=" * 115)

    # 1. Lưu JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {output_json}")

    # 2. Lưu CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Algorithm,Threshold,phone_F2_Err,phone_M2_Err,studio_F2_Err,studio_M2_Err,Avg_Err_Hz,Avg_Std_Err,Avg_F1_Pct,Avg_Acc_Pct\n")
        for r in summary_records:
            f.write(
                f"{r['algorithm']},{r['threshold']},"
                f"{r['files']['phone_F2']['abs_err_mean']},{r['files']['phone_M2']['abs_err_mean']},"
                f"{r['files']['studio_F2']['abs_err_mean']},{r['files']['studio_M2']['abs_err_mean']},"
                f"{r['average_error_hz']},{r['average_std_error']},"
                f"{r['average_voiced_f1_pct']},{r['average_accuracy_pct']}\n"
            )
    print(f"Saved report: {output_csv}")

    # 3. Vẽ biểu đồ đối sánh trực quan 3 thuật toán
    alg_names = [r["algorithm"] for r in summary_records]
    err_vals = [r["average_error_hz"] for r in summary_records]
    std_vals = [r["average_std_error"] for r in summary_records]
    f1_vals = [r["average_voiced_f1_pct"] for r in summary_records]
    acc_vals = [r["average_accuracy_pct"] for r in summary_records]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("So sánh Định lượng Tam giác Thuật toán Miền thời gian: ACF vs. AMDF vs. YIN", fontsize=14, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    x = np.arange(len(alg_names))
    width = 0.55

    # Subplot 1: MAE F0
    bars1 = ax1.bar(x, err_vals, color=colors, width=width, edgecolor="gray", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(alg_names, fontweight="bold", fontsize=10)
    ax1.set_ylabel("Sai số tuyệt đối trung bình |ΔF0| (Hz)", fontweight="bold")
    ax1.set_title("1. Sai số F0 (Hz) [Thấp hơn là tốt hơn]", fontweight="bold", fontsize=11)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.2f} Hz", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontweight="bold", fontsize=10)

    # Subplot 2: MAE Std
    bars2 = ax2.bar(x, std_vals, color=colors, width=width, edgecolor="gray", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(alg_names, fontweight="bold", fontsize=10)
    ax2.set_ylabel("Sai số độ lệch chuẩn |Δstd|", fontweight="bold")
    ax2.set_title("2. Sai số Độ lệch chuẩn [Thấp hơn là tốt hơn]", fontweight="bold", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontweight="bold", fontsize=10)

    # Subplot 3: F1-Score
    bars3 = ax3.bar(x, f1_vals, color=colors, width=width, edgecolor="gray", alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(alg_names, fontweight="bold", fontsize=10)
    ax3.set_ylabel("Voiced F1-Score (%)", fontweight="bold")
    ax3.set_title("3. Voiced F1-Score (%) [Cao hơn là tốt hơn]", fontweight="bold", fontsize=11)
    ax3.set_ylim(80, 100)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.6)
    for b in bars3:
        h = b.get_height()
        ax3.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontweight="bold", fontsize=10)

    # Subplot 4: Accuracy
    bars4 = ax4.bar(x, acc_vals, color=colors, width=width, edgecolor="gray", alpha=0.9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(alg_names, fontweight="bold", fontsize=10)
    ax4.set_ylabel("V/UV Classification Accuracy (%)", fontweight="bold")
    ax4.set_title("4. V/UV Accuracy (%) [Cao hơn là tốt hơn]", fontweight="bold", fontsize=11)
    ax4.set_ylim(75, 100)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.6)
    for b in bars4:
        h = b.get_height()
        ax4.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YIN vs ACF vs AMDF.")
    parser.add_argument("--test_dir", type=str, default="TinHieuKiemThu", help="Test directory")
    args = parser.parse_args()
    run_yin_benchmark(test_dir=args.test_dir)
