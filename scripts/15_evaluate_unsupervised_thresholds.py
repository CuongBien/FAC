"""Script 15: Benchmark toàn diện 4 giải pháp học không giám sát tìm ngưỡng Voiced/Unvoiced.
Đáp ứng yêu cầu mở rộng của Giảng viên:
- Tìm ngưỡng tối ưu phân biệt Voiced / Unvoiced hoàn toàn không dùng nhãn ground-truth .lab
- Khai thác giả định tiên nghiệm tỉ lệ khung Voiced và Unvoiced xấp xỉ 50/50 (P(V) ≈ P(UV) ≈ 0.5)
- Áp dụng trên cả 3 thuật toán cốt lõi Bài Tập 2: ACF, AMDF, Cepstrum.
- Đánh giá bằng bảng chuẩn 8 cột trên tập Kiểm thử (TinHieuKiemThu).
"""
import argparse
import csv
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

from src.analysis.unsupervised_threshold import (
    compute_otsu_threshold,
    compute_histogram_valley_threshold,
    compute_median_threshold,
    compute_kmeans_threshold,
)
from src.analysis.evaluation import (
    evaluate_against_ground_truth, round_metric, mean_metric, format_metric,
)
from src import config
from src.core.audio import load_wav
from src.pipeline.features import extract_raw_features
from src.core.lab_parser import parse_lab_file
from src.core.pitch_detector import PitchDetector


STRATEGY_LABELS = {
    "median": "Median (Tiên nghiệm 50/50)",
    "otsu": "Otsu (Cực đại phương sai)",
    "kmeans": "K-Means (Phân cụm 1D)",
    "valley": "Đáy Thung Lũng (Valley)",
    "supervised": "Chuẩn đối sánh (tham số lịch sử)",
}


def extract_all_and_active_metrics(
    wav_paths: List[str],
    method: str,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    ste_silence_ratio: float = 0.008,
) -> Dict:
    """Extract periodicity metric values across all frames and active speech frames."""
    method = method.lower()
    all_values = []
    active_values = []
    total_frames = 0
    total_ref_voiced = 0

    for wav_path in wav_paths:
        sr, signal, _ = load_wav(wav_path)
        features = extract_raw_features(signal, sr, method, frame_duration_ms, hop_duration_ms,
                                        f0_min, f0_max, ste_silence_ratio, profile='raw_all')

        lab_path = wav_path.rsplit(".", 1)[0] + ".lab"
        if os.path.exists(lab_path):
            lab_data = parse_lab_file(lab_path)
            if lab_data.get("f0_num") is not None:
                total_ref_voiced += int(lab_data["f0_num"])

        total_frames += features['num_frames']
        all_values.extend(features['periodicity_vals'])
        active_values.extend(features['periodicity_vals'][features['is_active']])

    return {
        "all_values": np.array(all_values, dtype=np.float64),
        "active_values": np.array(active_values, dtype=np.float64),
        "total_frames": total_frames,
        "total_ref_voiced": total_ref_voiced,
    }


def evaluate_pipeline_across_files(
    wav_files: List[str],
    method: str,
    threshold: float,
) -> Dict[str, float]:
    """Run full pitch detector pipeline across test WAV files with a specific threshold.
    Returns averaged standard 8-column metrics.
    """
    detector = PitchDetector(
        method=method,
        threshold=threshold,
        frame_duration_ms=25.0,
        hop_duration_ms=10.0,
        f0_min=70.0,
        f0_max=400.0,
        ste_silence_ratio=0.008,
        use_median_filter=True,
        median_size=3,
    )

    all_diff_mean = []
    all_mape_f0 = []
    all_diff_std = []
    all_mape_std = []
    all_diff_num = []
    all_mape_num = []
    all_final_score = []

    for wav_path in wav_files:
        lab_path = wav_path.rsplit(".", 1)[0] + ".lab"
        res = detector.process_file(wav_path)
        eval_res = evaluate_against_ground_truth(res, lab_path)

        all_diff_mean.append(eval_res["abs_error_mean"])
        all_mape_f0.append(eval_res["mape_f0"])
        all_diff_std.append(eval_res["abs_error_std"])
        all_mape_std.append(eval_res["mape_std"])
        all_diff_num.append(eval_res["abs_error_num"])
        all_mape_num.append(eval_res["mape_num"])
        all_final_score.append(eval_res["final_score"])

    return {
        "diff_f0": mean_metric(all_diff_mean, 2),
        "mape_f0": mean_metric(all_mape_f0, 2),
        "diff_std": mean_metric(all_diff_std, 2),
        "mape_std": mean_metric(all_mape_std, 2),
        "diff_num": mean_metric(all_diff_num, 2),
        "mape_num": mean_metric(all_mape_num, 2),
        "final_score": mean_metric(all_final_score, 2),
    }


def plot_unsupervised_comparison(
    all_values: np.ndarray,
    thresholds: Dict[str, float],
    pipeline_eval: Dict[str, Dict[str, float]],
    method: str,
    save_path: str,
):
    """Plot distribution and comparative performance bar chart."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2.2, 1.1]})

    n_bins = 70
    min_val, max_val = np.min(all_values), np.max(all_values)
    bins = np.linspace(min_val, max_val, n_bins)

    metric_name = {
        "acf": "Cực đại tự tương quan chuẩn hóa R_max trong [0, 1]",
        "amdf": "Cực tiểu hiệu biên độ chuẩn hóa D_min trong [0, 1]",
        "cepstrum": "Đỉnh biên độ quefrency c[q_0]",
    }.get(method, "Đặc trưng chu kỳ")

    # Top Subplot: Empirical Histogram of All Frames
    counts, edges, patches = ax1.hist(
        all_values,
        bins=bins,
        alpha=0.65,
        color="#2980b9",
        density=True,
        edgecolor="white",
        linewidth=0.6,
        label=f"Toàn bộ phân bố tín hiệu (N = {len(all_values)} khung)",
    )

    # Color palette & styles for thresholds
    styles = [("#d35400", "-"), ("#8e44ad", "--"), ("#16a085", "-."),
              ("#27ae60", ":"), ("#2c3e50", "-")]
    for (strategy, label), (color, style) in zip(STRATEGY_LABELS.items(), styles):
        score = pipeline_eval[strategy]["final_score"]
        ax1.axvline(thresholds[strategy], color=color, linestyle=style, linewidth=2.2,
                    label=f"{label}: T = {format_metric(thresholds[strategy], '.4f')} [Score = {format_metric(score)}%]")

    direction_note = "Voiced nếu D_min ≤ T (Hõm cực tiểu)" if method == "amdf" else "Voiced nếu R_max ≥ T (Đỉnh chu kỳ)"
    ax1.set_title(
        f"Phân Bố Đặc Trưng Không Giám Sát & Ngưỡng Phân Tách Voiced / Unvoiced - Thuật Toán {method.upper()}\n"
        f"({direction_note} | Khai thác tiên nghiệm P(Voiced) ≈ 50%)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel(f"Giá trị đặc trưng: {metric_name}", fontsize=11)
    ax1.set_ylabel("Mật độ tần suất (Probability Density)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom Subplot: Bar chart comparing FINAL SCORE and MAPE Num
    strategies = list(STRATEGY_LABELS)
    scores = np.asarray([pipeline_eval[name]["final_score"] for name in strategies], dtype=float)
    mape_nums = np.asarray([pipeline_eval[name]["mape_num"] for name in strategies], dtype=float)

    x = np.arange(len(strategies))
    bar_width = 0.35

    bars1 = ax2.bar(x - bar_width / 2, scores, bar_width, label="FINAL SCORE (%) [Càng thấp càng tốt]", color="#e74c3c")
    bars2 = ax2.bar(x + bar_width / 2, mape_nums, bar_width, label="MAPE Số khung Voiced (%)", color="#3498db")

    for bar in bars1:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, (yval + 0.6 if np.isfinite(yval) else 0.6), f"{format_metric(yval, '.2f')}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, (yval + 0.6 if np.isfinite(yval) else 0.6), f"{format_metric(yval, '.2f')}%", ha="center", va="bottom", fontsize=8.5)

    ax2.set_ylabel("Sai số (%)", fontsize=10)
    ax2.set_title("So sánh Sai số Toàn chu trình Pitch (Final Score & MAPE Khung Voiced) trên TinHieuKiemThu", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=9.5)
    finite_values = np.r_[scores, mape_nums]
    finite_values = finite_values[np.isfinite(finite_values)]
    max_y = float(np.max(finite_values)) * 1.18 if len(finite_values) else 20.0
    ax2.set_ylim(0, max(max_y, 20))
    ax2.legend(loc="upper left", fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")


def write_benchmark_results(all_results, output_dir):
    """Use canonical strategy IDs for JSON, CSV and the thresholds actually used."""
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "15_unsupervised_thresholds_results.json")
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(all_results, stream, indent=2, ensure_ascii=False, allow_nan=False)
    csv_path = os.path.join(reports_dir, "15_unsupervised_thresholds_results.csv")
    keys = ["diff_f0", "mape_f0", "diff_std", "mape_std", "diff_num", "mape_num", "final_score"]
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Algorithm", "Strategy", "Strategy_Label", "Threshold_T", "Diff_F0_Hz",
                         "MAPE_F0_Pct", "Diff_Std_Hz", "MAPE_Std_Pct", "Diff_Num_Frames", "MAPE_Num_Pct", "Final_Score_Pct"])
        for method, result in all_results.items():
            for strategy, metrics in result["pipeline_eval"].items():
                writer.writerow([method.upper(), strategy, STRATEGY_LABELS[strategy],
                                 result["thresholds"][strategy]] + [metrics[key] for key in keys])
    return json_path, csv_path


def run_benchmark(output_dir="outputs"):
    train_wavs = sorted(glob.glob(os.path.join("TinHieuHuanLuyen", "*.wav")))
    test_wavs = sorted(glob.glob(os.path.join("TinHieuKiemThu", "*.wav")))
    all_results = {}
    for method in ("acf", "amdf", "cepstrum"):
        stats = extract_all_and_active_metrics(train_wavs, method=method)
        values = stats["all_values"]
        thresholds = {
            "median": compute_median_threshold(values),
            "otsu": compute_otsu_threshold(values),
            "kmeans": compute_kmeans_threshold(values),
            "valley": compute_histogram_valley_threshold(values),
            "supervised": config.baseline_threshold(method),
        }
        print(f"\n{method.upper()}: fit trên {stats['total_frames']} khung train (có silence).")
        print(f"Đối chứng dùng {config.THRESHOLD_SOURCE}")
        pipeline_eval = {}
        for strategy, label in STRATEGY_LABELS.items():
            metrics = evaluate_pipeline_across_files(test_wavs, method, thresholds[strategy])
            pipeline_eval[strategy] = metrics
            print(f"  {label}: T={format_metric(thresholds[strategy], '.6f')}; Final Score={format_metric(metrics['final_score'])}%")
        figure_path = os.path.join(output_dir, "figures", f"15_unsupervised_thresholds_{method}.png")
        plot_unsupervised_comparison(values, thresholds, pipeline_eval, method, figure_path)
        ratio = stats["total_ref_voiced"] / stats["total_frames"] * 100 if stats["total_frames"] else None
        all_results[method] = {
            "distribution_stats": {"total_frames": stats["total_frames"],
                                   "total_ref_voiced": stats["total_ref_voiced"],
                                   "voiced_ratio_pct": round_metric(ratio),
                                   "non_voiced_ratio_pct": round_metric(100-ratio) if ratio is not None else None},
            "thresholds": thresholds,
            "strategy_labels": dict(STRATEGY_LABELS),
            "pipeline_eval": pipeline_eval,
            "threshold_source": config.THRESHOLD_SOURCE,
            "protocol": {"unsupervised_fit": "train_all_frames_including_silence",
                         "evaluation": "test_with_shared_silence_gate_and_segment_median"},
            "figure_path": figure_path,
        }
    json_path, csv_path = write_benchmark_results(all_results, output_dir)
    print(f"Saved: {json_path}\nSaved: {csv_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs", help="Root for figures/ and reports/")
    run_benchmark(parser.parse_args().output_dir)
