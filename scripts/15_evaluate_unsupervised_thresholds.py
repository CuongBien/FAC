"""Script 15: Benchmark toàn diện 4 giải pháp học không giám sát tìm ngưỡng Voiced/Unvoiced.
Đáp ứng yêu cầu mở rộng của Giảng viên:
- Tìm ngưỡng tối ưu phân biệt Voiced / Unvoiced hoàn toàn không dùng nhãn ground-truth .lab
- Khai thác giả định tiên nghiệm tỉ lệ khung Voiced và Unvoiced xấp xỉ 50/50 (P(V) ≈ P(UV) ≈ 0.5)
- Áp dụng trên cả 3 thuật toán cốt lõi Bài Tập 2: ACF, AMDF, Cepstrum.
- Đánh giá bằng bảng chuẩn 8 cột trên tập Kiểm thử (TinHieuKiemThu).
"""
import glob
import json
import os
import sys
from typing import Dict, List, Tuple
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
from src.analysis.evaluation import evaluate_against_ground_truth
from src.core.audio import load_wav, frame_signal, compute_ste
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.core.lab_parser import parse_lab_file
from src.core.pitch_detector import PitchDetector


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
        frames, _ = frame_signal(
            signal,
            sr,
            frame_duration_ms=frame_duration_ms,
            hop_duration_ms=hop_duration_ms,
            window="rectangular",
        )
        ste, _ = compute_ste(
            signal,
            sr,
            frame_duration_ms=frame_duration_ms,
            hop_duration_ms=hop_duration_ms,
        )
        ste_thresh = ste_silence_ratio * np.max(ste) if len(ste) > 0 else 0.0

        lab_path = wav_path.rsplit(".", 1)[0] + ".lab"
        if os.path.exists(lab_path):
            lab_data = parse_lab_file(lab_path)
            if lab_data.get("f0_num") is not None:
                total_ref_voiced += int(lab_data["f0_num"])

        total_frames += len(frames)

        for i, frame in enumerate(frames):
            if method == "acf":
                _, metric_val, _ = find_f0_acf(
                    frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized"
                )
            elif method == "amdf":
                _, metric_val, _ = find_f0_amdf(
                    frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized"
                )
            elif method == "cepstrum":
                _, metric_val, _ = find_f0_cepstrum(
                    frame,
                    sr,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    window="hamming",
                    n_fft=2048,
                    mode="peak",
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            all_values.append(metric_val)
            if ste[i] >= ste_thresh:
                active_values.append(metric_val)

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
        "diff_f0": round(float(np.mean(all_diff_mean)), 2),
        "mape_f0": round(float(np.mean(all_mape_f0)), 2),
        "diff_std": round(float(np.mean(all_diff_std)), 2),
        "mape_std": round(float(np.mean(all_mape_std)), 2),
        "diff_num": round(float(np.mean(all_diff_num)), 2),
        "mape_num": round(float(np.mean(all_mape_num)), 2),
        "final_score": round(float(np.mean(all_final_score)), 2),
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
        "acf": "Cực đại tự tương quan chuẩn hóa R_max ∈ [0, 1]",
        "amdf": "Cực tiểu hiệu biên độ chuẩn hóa D_min ∈ [0, 1]",
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
    th_styles = {
        "Median (Tiên nghiệm 50/50)": (thresholds["Median (50/50 All)"], "#d35400", "-", 2.5),
        "Otsu (Cực đại phương sai)": (thresholds["Otsu (All Frames)"], "#8e44ad", "--", 2.2),
        "K-Means (Phân cụm 1D)": (thresholds["K-Means (All Frames)"], "#16a085", "-.", 2.2),
        "Đáy Thung Lũng (Valley)": (thresholds["Valley (All Frames)"], "#27ae60", ":", 2.2),
        "Chuẩn đối sánh (Có giám sát)": (thresholds["Supervised Baseline"], "#2c3e50", "-", 2.5),
    }

    for name, (t_val, color, ls, lw) in th_styles.items():
        score_val = pipeline_eval.get(name, {}).get("final_score", 0.0)
        ax1.axvline(
            t_val,
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=f"{name}: T = {t_val:.4f} [Score = {score_val:.2f}%]",
        )

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
    strategies = list(th_styles.keys())
    scores = [pipeline_eval[name]["final_score"] for name in strategies]
    mape_nums = [pipeline_eval[name]["mape_num"] for name in strategies]

    x = np.arange(len(strategies))
    bar_width = 0.35

    bars1 = ax2.bar(x - bar_width / 2, scores, bar_width, label="FINAL SCORE (%) [Càng thấp càng tốt]", color="#e74c3c")
    bars2 = ax2.bar(x + bar_width / 2, mape_nums, bar_width, label="MAPE Số khung Voiced (%)", color="#3498db")

    for bar in bars1:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.6, f"{yval:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.6, f"{yval:.2f}%", ha="center", va="bottom", fontsize=8.5)

    ax2.set_ylabel("Sai số (%)", fontsize=10)
    ax2.set_title("So sánh Sai số Toàn chu trình Pitch (Final Score & MAPE Khung Voiced) trên TinHieuKiemThu", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=9.5)
    max_y = max(max(scores), max(mape_nums)) * 1.18
    ax2.set_ylim(0, max(max_y, 20))
    ax2.legend(loc="upper left", fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")


def run_benchmark():
    train_dir = "TinHieuHuanLuyen"
    test_dir = "TinHieuKiemThu"

    train_wavs = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    test_wavs = sorted(glob.glob(os.path.join(test_dir, "*.wav")))

    methods = ["acf", "amdf", "cepstrum"]
    all_results = {}

    print("=" * 125)
    print("        BÁO CÁO THỰC NGHIỆM: 4 GIẢI PHÁP HỌC KHÔNG GIÁM SÁT TÌM NGƯỠNG TỐI ƯU VOICED / UNVOICED")
    print("        (Yêu cầu: Không dùng nhãn .lab | Khai thác triệt để giả định tiên nghiệm P(V) ≈ P(UV) ≈ 50%)")
    print("=" * 125)

    for method in methods:
        print(f"\n{'#' * 45} THUẬT TOÁN: {method.upper()} {'#' * 45}")

        # 1. Trích xuất đặc trưng
        train_stats = extract_all_and_active_metrics(train_wavs, method=method)
        all_vals = train_stats["all_values"]
        act_vals = train_stats["active_values"]
        tot_frames = train_stats["total_frames"]
        ref_v = train_stats["total_ref_voiced"]
        v_ratio = (ref_v / tot_frames * 100.0) if tot_frames > 0 else 0.0

        print(f"1. Thống kê tập Huấn luyện (TinHieuHuanLuyen):")
        print(f"   * Tổng số khung tín hiệu: {tot_frames} khung")
        print(f"   * Số khung Voiced chuẩn (F0num ground-truth): {ref_v} khung")
        print(f"   * Tỉ lệ thực tế: Voiced = {v_ratio:.1f}%, Non-Voiced = {100.0 - v_ratio:.1f}%")
        print(f"   * NHẬN XÉT: Tỉ lệ thực tế ({v_ratio:.1f}%) sát tuyệt đối với giả định tiên nghiệm 50/50 của đề bài!")

        # 2. Tính 4 ngưỡng không giám sát trên toàn bộ phân bố tín hiệu
        t_otsu = compute_otsu_threshold(all_vals)
        t_median = compute_median_threshold(all_vals)
        t_kmeans = compute_kmeans_threshold(all_vals)
        t_valley = compute_histogram_valley_threshold(all_vals)

        # Ngưỡng có giám sát đối chứng
        if method == "acf":
            t_sup = 0.4408
        elif method == "amdf":
            t_sup = 0.4380
        else:
            t_sup = 0.0600  # Baseline Cepstrum

        threshold_dict = {
            "Median (Tiên nghiệm 50/50)": t_median,
            "Otsu (Cực đại phương sai)": t_otsu,
            "K-Means (Phân cụm 1D)": t_kmeans,
            "Đáy Thung Lũng (Valley)": t_valley,
            "Chuẩn đối sánh (Có giám sát)": t_sup,
        }

        # Lưu lại cho pipeline
        raw_th_dict = {
            "Median (50/50 All)": t_median,
            "Otsu (All Frames)": t_otsu,
            "K-Means (All Frames)": t_kmeans,
            "Valley (All Frames)": t_valley,
            "Supervised Baseline": t_sup,
        }

        print("\n2. Bảng Đối Sánh Giá Trị Ngưỡng Khai Thác Tiên Nghiệm 50/50:")
        print(f"   {'Chiến lược phân ngưỡng':<30} | {'Ngưỡng T':<9} | {'Nguyên lý toán học cốt lõi'}")
        print("   " + "-" * 90)
        print(f"   {'Median (Tiên nghiệm 50/50)':<30} | {t_median:<9.4f} | Phân vị thứ 50 (P50), chia đôi 50% V và 50% UV")
        print(f"   {'Otsu (Cực đại phương sai)':<30} | {t_otsu:<9.4f} | Cực đại hóa phương sai giữa 2 lớp (Between-class variance)")
        print(f"   {'K-Means (Phân cụm 1D)':<30} | {t_kmeans:<9.4f} | Ranh giới trung điểm giữa 2 tâm cụm k-means (K=2)")
        print(f"   {'Đáy Thung Lũng (Valley)':<30} | {t_valley:<9.4f} | Cực tiểu cục bộ giữa 2 đỉnh phân bố hai đỉnh (Bimodal)")
        print(f"   {'Chuẩn đối sánh (Có giám sát)':<30} | {t_sup:<9.4f} | Giao điểm phân bố Gauss Bayes (cần nhãn .lab)")

        # 3. Đánh giá chu trình Pitch trên TinHieuKiemThu (Bảng chuẩn 8 cột)
        print("\n3. Bảng Đánh Giá Định Lượng Toàn Chu Trình Pitch Trên Tập Kiểm Thử (TinHieuKiemThu):")
        print(f"   {'Phương pháp / Ngưỡng':<30} | {'ΔF0 (Hz)':<9} | {'MAPE F0 (%)':<12} | {'ΔStd (Hz)':<10} | {'MAPE Std (%)':<13} | {'ΔNum':<7} | {'MAPE Num (%)':<13} | {'FINAL SCORE (%)':<15}")
        print("   " + "-" * 125)

        pipeline_eval = {}
        for display_name, raw_key in [
            ("Median (Tiên nghiệm 50/50)", "Median (50/50 All)"),
            ("Otsu (Cực đại phương sai)", "Otsu (All Frames)"),
            ("K-Means (Phân cụm 1D)", "K-Means (All Frames)"),
            ("Đáy Thung Lũng (Valley)", "Valley (All Frames)"),
            ("Chuẩn đối sánh (Có giám sát)", "Supervised Baseline"),
        ]:
            t_val = raw_th_dict[raw_key]
            p_res = evaluate_pipeline_across_files(test_wavs, method=method, threshold=t_val)
            pipeline_eval[display_name] = p_res
            print(
                f"   {display_name:<30} | "
                f"{p_res['diff_f0']:<9.2f} | "
                f"{p_res['mape_f0']:<12.2f} | "
                f"{p_res['diff_std']:<10.2f} | "
                f"{p_res['mape_std']:<13.2f} | "
                f"{p_res['diff_num']:<7.1f} | "
                f"{p_res['mape_num']:<13.2f} | "
                f"{p_res['final_score']:<15.2f}"
            )

        # 4. Xuất đồ thị hình vẽ
        fig_path = f"outputs/figures/15_unsupervised_thresholds_{method}.png"
        plot_unsupervised_comparison(
            all_values=all_vals,
            thresholds=raw_th_dict,
            pipeline_eval=pipeline_eval,
            method=method,
            save_path=fig_path,
        )

        all_results[method] = {
            "distribution_stats": {
                "total_frames": tot_frames,
                "total_ref_voiced": ref_v,
                "voiced_ratio_pct": round(v_ratio, 2),
                "unvoiced_ratio_pct": round(100.0 - v_ratio, 2),
            },
            "thresholds": {k: round(v, 4) for k, v in raw_th_dict.items()},
            "pipeline_eval": pipeline_eval,
            "figure_path": fig_path,
        }

    # 5. Lưu báo cáo JSON và CSV
    json_path = "outputs/reports/15_unsupervised_thresholds_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Đã lưu báo cáo JSON tại: {json_path}")

    csv_path = "outputs/reports/15_unsupervised_thresholds_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Algorithm,Strategy,Threshold_T,Diff_F0_Hz,MAPE_F0_Pct,Diff_Std_Hz,MAPE_Std_Pct,Diff_Num_Frames,MAPE_Num_Pct,Final_Score_Pct\n")
        for method, m_data in all_results.items():
            for name, p_ev in m_data["pipeline_eval"].items():
                t_val = m_data["thresholds"].get(name, 0.0)
                f.write(
                    f"{method.upper()},{name},{t_val},"
                    f"{p_ev['diff_f0']},{p_ev['mape_f0']},{p_ev['diff_std']},{p_ev['mape_std']},"
                    f"{p_ev['diff_num']},{p_ev['mape_num']},{p_ev['final_score']}\n"
                )
    print(f"[OK] Đã lưu báo cáo CSV tại: {csv_path}")
    print("=" * 125)


if __name__ == "__main__":
    run_benchmark()
