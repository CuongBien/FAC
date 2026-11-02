"""Benchmark the preserved original ACF-based Pitch-Shift consistency baseline."""
import csv
import glob
import json
import os
import sys
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
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

from src.core.audio import load_wav
from src.core.lab_parser import parse_lab_file
from src.core.processing import smooth_voiced_segments
from src.analysis.ssl_pitch_shift import analyze_signal_ssl_pitch_shift


SSL_METHODS = {"ssl_pitch_shift": "Pitch-Shift original (ACF)"}


def run_ssl_analysis(method_key: str, signal: np.ndarray, fs: int) -> Dict:
    if method_key != 'ssl_pitch_shift':
        raise ValueError(f'Unknown Pitch-Shift method: {method_key}')
    return analyze_signal_ssl_pitch_shift(signal, fs, frame_duration_ms=30., hop_duration_ms=10.)


def evaluate_test_dataset(test_dir="TinHieuKiemThu"):
    """Evaluate all SSL methods on the test dataset."""
    wav_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {test_dir}")

    benchmark_records = {}

    for method_key, method_name in SSL_METHODS.items():
        file_metrics = []
        for wav_path in wav_files:
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            lab_path = os.path.join(test_dir, f"{base_name}.lab")
            gt = parse_lab_file(lab_path)

            fs, signal, _ = load_wav(wav_path)
            res = run_ssl_analysis(method_key, signal, fs)

            # Apply segment-wise median filter to smooth pitch contour
            smoothed_f0 = smooth_voiced_segments(res["f0_contour"], labels=res["labels"], size=5)
            voiced_pitches = smoothed_f0[res["labels"] == "v"]
            valid_pitches = voiced_pitches[voiced_pitches > 0]

            f0_mean = float(np.mean(valid_pitches)) if len(valid_pitches) else 0.0
            f0_std = float(np.std(valid_pitches)) if len(valid_pitches) else 0.0
            f0_num = len(voiced_pitches)

            gt_mean = gt["f0_mean"]
            gt_std = gt["f0_std"]
            gt_num = gt["f0_num"]

            diff_mean = abs(f0_mean - gt_mean)
            diff_std = abs(f0_std - gt_std)
            diff_num = abs(f0_num - gt_num)

            mape_mean = (diff_mean / gt_mean) * 100.0 if gt_mean > 0 else 0.0
            mape_std = (diff_std / gt_std) * 100.0 if gt_std > 0 else 0.0
            mape_num = (diff_num / gt_num) * 100.0 if gt_num > 0 else 0.0
            final_score = (mape_mean + mape_std + mape_num) / 3.0

            file_metrics.append({
                "file": base_name,
                "f0_mean": f0_mean, "f0_std": f0_std, "f0_num": f0_num,
                "diff_f0": diff_mean, "mape_f0": mape_mean,
                "diff_std": diff_std, "mape_std": mape_std,
                "diff_num": diff_num, "mape_num": mape_num,
                "final_score": final_score,
            })

        avg_diff_f0 = float(np.mean([m["diff_f0"] for m in file_metrics]))
        avg_mape_f0 = float(np.mean([m["mape_f0"] for m in file_metrics]))
        avg_diff_std = float(np.mean([m["diff_std"] for m in file_metrics]))
        avg_mape_std = float(np.mean([m["mape_std"] for m in file_metrics]))
        avg_diff_num = float(np.mean([m["diff_num"] for m in file_metrics]))
        avg_mape_num = float(np.mean([m["mape_num"] for m in file_metrics]))
        avg_final_score = float(np.mean([m["final_score"] for m in file_metrics]))

        benchmark_records[method_key] = {
            "name": method_name,
            "avg_diff_f0": avg_diff_f0,
            "avg_mape_f0": avg_mape_f0,
            "avg_diff_std": avg_diff_std,
            "avg_mape_std": avg_mape_std,
            "avg_diff_num": avg_diff_num,
            "avg_mape_num": avg_mape_num,
            "avg_final_score": avg_final_score,
            "files": file_metrics,
        }

    return benchmark_records


def print_comparison_table(records: Dict):
    """Print the standardized 8-column benchmark table."""
    print("\n" + "=" * 105)
    print(" BẢNG ĐỐI SÁNH KẾT QUẢ CÁC GIẢI PHÁP SELF-SUPERVISED LEARNING (SSL) TRÊN TINHIEUKIEMTHU")
    print(" (Tất cả giải pháp dùng ranh giới xác suất tự nhiên Bayes P >= 0.5 | Không dùng nhãn ground-truth)")
    print("=" * 105)
    header = (
        f"{'Phương pháp SSL':<35} | "
        f"{'ΔF0 (Hz)':<8} | {'MAPE F0':<8} | "
        f"{'ΔStd(Hz)':<8} | {'MAPE Std':<9} | "
        f"{'ΔNum':<6} | {'MAPE Num':<9} | "
        f"{'FINAL SCORE':<11}"
    )
    print(header)
    print("-" * 105)

    sorted_keys = sorted(records.keys(), key=lambda k: records[k]["avg_final_score"])
    for rank, key in enumerate(sorted_keys, 1):
        r = records[key]
        medal = "🥇 " if rank == 1 else "🥈 " if rank == 2 else "🥉 " if rank == 3 else f"{rank}. "
        line = (
            f"{medal + r['name']:<35} | "
            f"{r['avg_diff_f0']:>8.2f} | {r['avg_mape_f0']:>7.2f}% | "
            f"{r['avg_diff_std']:>8.2f} | {r['avg_mape_std']:>8.2f}% | "
            f"{r['avg_diff_num']:>6.1f} | {r['avg_mape_num']:>8.2f}% | "
            f"{r['avg_final_score']:>10.3f}%"
        )
        print(line)
    print("=" * 105)


def export_and_plot_results(records: Dict, output_dir="outputs"):
    """Export benchmark JSON, CSV, and summary figure."""
    reports_dir = os.path.join(output_dir, "reports")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Export JSON
    json_path = os.path.join(reports_dir, "27_ssl_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # 2. Export CSV
    csv_path = os.path.join(reports_dir, "27_ssl_benchmark_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method_ID", "Method_Name", "Diff_F0_Hz", "MAPE_F0_Pct",
            "Diff_Std_Hz", "MAPE_Std_Pct", "Diff_Num_Frames", "MAPE_Num_Pct", "Final_Score_Pct"
        ])
        for key, r in records.items():
            writer.writerow([
                key, r["name"], round(r["avg_diff_f0"], 3), round(r["avg_mape_f0"], 3),
                round(r["avg_diff_std"], 3), round(r["avg_mape_std"], 3),
                round(r["avg_diff_num"], 3), round(r["avg_mape_num"], 3),
                round(r["avg_final_score"], 3),
            ])

    # 3. Create High-Quality Matplotlib Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sorted_keys = sorted(records.keys(), key=lambda k: records[k]["avg_final_score"])
    names = [records[k]["name"].split(". ")[-1] for k in sorted_keys]
    final_scores = [records[k]["avg_final_score"] for k in sorted_keys]
    mape_nums = [records[k]["avg_mape_num"] for k in sorted_keys]
    mape_f0s = [records[k]["avg_mape_f0"] for k in sorted_keys]

    colors = ["#27ae60", "#2980b9", "#8e44ad", "#e67e22", "#d35400"]

    # Bar chart 1: Final Score
    bars = ax1.barh(names, final_scores, color=colors[:len(names)], height=0.55)
    ax1.set_xlabel("FINAL SCORE (%) [Càng thấp càng chính xác]", fontsize=11, fontweight="bold")
    ax1.set_title("Xếp Hạng Final Score Các Giải Pháp Self-Supervised Learning", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=":", alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=10, fontweight="bold")

    # Bar chart 2: Component Breakdown
    y = np.arange(len(names))
    height = 0.35
    b1 = ax2.bar(y - height / 2, mape_f0s, height, label="MAPE F0 (Tần số cơ bản)", color="#3498db")
    b2 = ax2.bar(y + height / 2, mape_nums, height, label="MAPE Num (Số khung Voiced)", color="#e74c3c")
    ax2.set_xticks(y)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    ax2.set_ylabel("Sai số thành phần (%)", fontsize=11, fontweight="bold")
    ax2.set_title("So Sánh Chi Tiết Sai Số F0 vs Sai Số Phân Loại Số Khung V/UV", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "27_ssl_benchmark_summary.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\n[+] Đã lưu báo cáo JSON: {json_path}")
    print(f"[+] Đã lưu bảng số liệu CSV: {csv_path}")
    print(f"[+] Đã lưu biểu đồ trực quan 300 DPI: {fig_path}")


def main():
    print("=" * 80)
    print(" ĐANG CHẠY BENCHMARK ĐỐI SÁNH TẤT CẢ GIẢI PHÁP SELF-SUPERVISED LEARNING (SSL)")
    print("=" * 80)
    records = evaluate_test_dataset("TinHieuKiemThu")
    print_comparison_table(records)
    export_and_plot_results(records, output_dir="outputs")


if __name__ == "__main__":
    main()
