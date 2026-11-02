"""Compare native Champion with Pitch-Shift using each algorithm on both views."""
import csv
import json
import os
import sys
from pathlib import Path
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
from src.pipelines import build_champion
from src.pipeline.champion import extract_file_features
from src.analysis.champion_unsupervised import experiment_configuration
from src.analysis.ssl_pitch_shift import analyze_signal_ssl_pitch_shift
from src.analysis.ssl_pitch_shift_methods import analyze_signal_pitch_shift_by_method


SSL_STRATEGIES = {
    'champion_native': 'Champion gốc (Ngưỡng tĩnh)',
    'ssl_pitch_shift': 'Champion + Pitch-Shift theo thuật toán',
}


def get_ssl_decisions(signal: np.ndarray, fs: int, *, method: str) -> Dict[str, np.ndarray]:
    return {'ssl_pitch_shift': analyze_signal_pitch_shift_by_method(signal, fs, method)['labels']}


def evaluate_champion_ssl_matrix(test_dir="TinHieuKiemThu") -> Dict:
    """Evaluate all Champion x SSL combinations across test audio files."""
    test_paths = sorted(Path(test_dir).glob("*.wav"))
    if not test_paths:
        raise FileNotFoundError(f"No WAV files found in {test_dir}")

    settings = experiment_configuration()
    matrix_results = {}
    ssl_by_file = {}

    for method in ("acf", "amdf", "cepstrum"):
        matrix_results[method] = {}
        detector = build_champion(method)

        for strategy_key, strategy_name in SSL_STRATEGIES.items():
            file_records = []

            for path in test_paths:
                lab_path = path.with_suffix(".lab")
                gt = parse_lab_file(str(lab_path))
                fs, signal, _ = load_wav(str(path))
                fd = extract_file_features(path, method, settings)

                if strategy_key == "champion_native":
                    # Original frozen Champion detector
                    res = detector.process_file(str(path))
                else:
                    # Run SSL decision extraction on signal
                    key = (path, method)
                    if key not in ssl_by_file:
                        ssl_by_file[key] = get_ssl_decisions(signal, fs, method=method)
                    target_labels = ssl_by_file[key][strategy_key]
                    # Apply SSL labels through Champion's original Energy/Viterbi processing
                    res = detector.process_feature_labels(fd, target_labels)

                v_mask = res["labels"] == "v"
                voiced_pitches = res["f0_contour"][v_mask]
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

                file_records.append({
                    "file": path.stem,
                    "diff_f0": diff_mean, "mape_f0": mape_mean,
                    "diff_std": diff_std, "mape_std": mape_std,
                    "diff_num": diff_num, "mape_num": mape_num,
                    "final_score": final_score,
                })

            avg_score = float(np.mean([r["final_score"] for r in file_records]))
            matrix_results[method][strategy_key] = {
                "name": strategy_name,
                "avg_diff_f0": float(np.mean([r["diff_f0"] for r in file_records])),
                "avg_mape_f0": float(np.mean([r["mape_f0"] for r in file_records])),
                "avg_diff_std": float(np.mean([r["diff_std"] for r in file_records])),
                "avg_mape_std": float(np.mean([r["mape_std"] for r in file_records])),
                "avg_diff_num": float(np.mean([r["diff_num"] for r in file_records])),
                "avg_mape_num": float(np.mean([r["mape_num"] for r in file_records])),
                "avg_final_score": avg_score,
                "files": file_records,
            }

    return matrix_results


def print_matrix_tables(matrix_results: Dict):
    """Print clean 8-column tables for ACF, AMDF, and Cepstrum Champions."""
    for method, strategies in matrix_results.items():
        print("\n" + "=" * 105)
        print(f" KẾT QUẢ ĐỐI SÁNH: {method.upper()} CHAMPION KẾT HỢP VỚI CÁC GIẢI PHÁP SSL (TINHIEUKIEMTHU)")
        print("=" * 105)
        header = (
            f"{'Cấu hình Champion + Quyết định':<35} | "
            f"{'ΔF0 (Hz)':<8} | {'MAPE F0':<8} | "
            f"{'ΔStd(Hz)':<8} | {'MAPE Std':<9} | "
            f"{'ΔNum':<6} | {'MAPE Num':<9} | "
            f"{'FINAL SCORE':<11}"
        )
        print(header)
        print("-" * 105)

        sorted_keys = sorted(strategies.keys(), key=lambda k: strategies[k]["avg_final_score"])
        for rank, key in enumerate(sorted_keys, 1):
            r = strategies[key]
            tag = "★ " if key == "champion_native" else f"{rank}. "
            if rank == 1:
                tag = "🥇 "
            line = (
                f"{tag + r['name']:<35} | "
                f"{r['avg_diff_f0']:>8.2f} | {r['avg_mape_f0']:>7.2f}% | "
                f"{r['avg_diff_std']:>8.2f} | {r['avg_mape_std']:>8.2f}% | "
                f"{r['avg_diff_num']:>6.1f} | {r['avg_mape_num']:>8.2f}% | "
                f"{r['avg_final_score']:>10.3f}%"
            )
            print(line)
        print("=" * 105)


def export_matrix_reports(matrix_results: Dict, output_dir="outputs"):
    """Save JSON, CSV and comparison figure."""
    reports_dir = os.path.join(output_dir, "reports")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    json_path = os.path.join(reports_dir, "28_champion_ssl_matrix_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(reports_dir, "28_champion_ssl_matrix_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "Strategy_Key", "Strategy_Name", "Diff_F0_Hz", "MAPE_F0_Pct",
            "Diff_Std_Hz", "MAPE_Std_Pct", "Diff_Num_Frames", "MAPE_Num_Pct", "Final_Score_Pct"
        ])
        for method, strategies in matrix_results.items():
            for key, r in strategies.items():
                writer.writerow([
                    method.upper(), key, r["name"], round(r["avg_diff_f0"], 3), round(r["avg_mape_f0"], 3),
                    round(r["avg_diff_std"], 3), round(r["avg_mape_std"], 3),
                    round(r["avg_diff_num"], 3), round(r["avg_mape_num"], 3),
                    round(r["avg_final_score"], 3),
                ])

    # Plot Matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    methods = ("acf", "amdf", "cepstrum")
    colors = ["#2c3e50", "#27ae60", "#2980b9", "#e67e22", "#9b59b6", "#e74c3c"]

    for ax, method in zip(axes, methods):
        strategies = matrix_results[method]
        sorted_keys = sorted(strategies.keys(), key=lambda k: strategies[k]["avg_final_score"])
        names = [strategies[k]["name"].replace("Champion + ", "").replace("Champion gốc (Ngưỡng tĩnh)", "Gốc (Tĩnh)") for k in sorted_keys]
        scores = [strategies[k]["avg_final_score"] for k in sorted_keys]

        bars = ax.barh(names, scores, color=colors[:len(names)], height=0.55)
        ax.set_title(f"Champion {method.upper()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("FINAL SCORE (%) [Càng thấp càng tốt]", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, linestyle=":", alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9.5, fontweight="bold")

    fig.suptitle("Ma Trận Đối Sánh: 3 Champion Pipelines Kết Hợp Các Quyết Định Self-Supervised Learning", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "28_champion_ssl_matrix.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\n[+] Đã xuất báo cáo JSON: {json_path}")
    print(f"[+] Đã xuất bảng số liệu CSV: {csv_path}")
    print(f"[+] Đã xuất biểu đồ so sánh 300 DPI: {fig_path}")


def main():
    print("=" * 80)
    print(" ĐANG CHẠY TOÀN BỘ MA TRẬN 3 CHAMPIONS x PITCH-SHIFT TRÊN TINHIEUKIEMTHU")
    print("=" * 80)
    matrix_results = evaluate_champion_ssl_matrix("TinHieuKiemThu")
    print_matrix_tables(matrix_results)
    export_matrix_reports(matrix_results, output_dir="outputs")


if __name__ == "__main__":
    main()
