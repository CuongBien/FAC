"""Plotting routines using Matplotlib for speech signals, F0 contours, and ACF/AMDF."""
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_frame_acf_comparison(
    voiced_frame: np.ndarray,
    voiced_acf: np.ndarray,
    voiced_f0: float,
    voiced_lag: int,
    voiced_peak: float,
    unvoiced_frame: np.ndarray,
    unvoiced_acf: np.ndarray,
    unvoiced_lag: int,
    unvoiced_peak: float,
    sample_rate: int,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    save_path: Optional[str] = None,
    title: str = "So sánh Hàm Tự Tương Quan (ACF): Khung Hữu Thanh vs Vô Thanh",
):
    """Plot a comprehensive 2x2 comparison between a Voiced frame and an Unvoiced frame.

    - Top Left: Voiced Frame Waveform (shows quasi-periodicity).
    - Bottom Left: Voiced Frame Normalized ACF (shows prominent peak at pitch period T0).
    - Top Right: Unvoiced Frame Waveform (noise-like, non-periodic).
    - Bottom Right: Unvoiced Frame Normalized ACF (rapid decay, no dominant peak).
    """
    n_v = len(voiced_frame)
    n_uv = len(unvoiced_frame)

    t_v = np.arange(n_v) / sample_rate * 1000.0  # ms
    t_uv = np.arange(n_uv) / sample_rate * 1000.0  # ms

    lag_v_ms = np.arange(len(voiced_acf)) / sample_rate * 1000.0  # ms
    lag_uv_ms = np.arange(len(unvoiced_acf)) / sample_rate * 1000.0  # ms

    lag_min = int(round(sample_rate / f0_max))
    lag_max = int(round(sample_rate / f0_min))
    lag_min_ms = lag_min / sample_rate * 1000.0
    lag_max_ms = lag_max / sample_rate * 1000.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    # Subplot 1: Voiced Waveform
    ax1 = axes[0, 0]
    ax1.plot(t_v, voiced_frame, color="#1f77b4", linewidth=1.5, label="Voiced signal")
    ax1.set_title("1. Tín hiệu Khung Hữu Thanh (Voiced Frame)", fontsize=12, fontweight="bold", color="#1f77b4")
    ax1.set_xlabel("Thời gian (ms)")
    ax1.set_ylabel("Biên độ chuẩn hóa")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Subplot 2: Voiced ACF
    ax2 = axes[1, 0]
    ax2.plot(lag_v_ms, voiced_acf, color="#1f77b4", linewidth=1.5, label="Normalized ACF R(τ)")
    # Highlight pitch search region [lag_min, lag_max]
    ax2.axvspan(lag_min_ms, lag_max_ms, color="green", alpha=0.15, label=f"Vùng tìm F0 [{f0_min:.0f}-{f0_max:.0f} Hz]")
    # Highlight pitch peak
    t0_ms = voiced_lag / sample_rate * 1000.0
    ax2.plot(t0_ms, voiced_peak, "ro", markersize=8, label=f"Đỉnh T0 = {t0_ms:.2f} ms\nF0 = {voiced_f0:.1f} Hz (Peak: {voiced_peak:.3f})")
    ax2.axvline(t0_ms, color="red", linestyle="--", alpha=0.7)
    ax2.set_title(f"ACF Khung Hữu Thanh -> F0 = {voiced_f0:.1f} Hz (Xác định)", fontsize=12, fontweight="bold", color="darkgreen")
    ax2.set_xlabel("Độ trễ lag τ (ms)")
    ax2.set_ylabel("ACF chuẩn hóa R(τ)/R(0)")
    ax2.set_ylim(-0.6, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    # Subplot 3: Unvoiced Waveform
    ax3 = axes[0, 1]
    ax3.plot(t_uv, unvoiced_frame, color="#d62728", linewidth=1.2, label="Unvoiced signal")
    ax3.set_title("2. Tín hiệu Khung Vô Thanh (Unvoiced Frame)", fontsize=12, fontweight="bold", color="#d62728")
    ax3.set_xlabel("Thời gian (ms)")
    ax3.set_ylabel("Biên độ chuẩn hóa")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")

    # Subplot 4: Unvoiced ACF
    ax4 = axes[1, 1]
    ax4.plot(lag_uv_ms, unvoiced_acf, color="#d62728", linewidth=1.5, label="Normalized ACF R(τ)")
    ax4.axvspan(lag_min_ms, lag_max_ms, color="orange", alpha=0.15, label=f"Vùng tìm F0 [{f0_min:.0f}-{f0_max:.0f} Hz]")
    unvoiced_t0_ms = unvoiced_lag / sample_rate * 1000.0
    ax4.plot(unvoiced_t0_ms, unvoiced_peak, "kx", markersize=8, markeredgewidth=2, label=f"Cực đại cục bộ ({unvoiced_peak:.3f} < Ngưỡng)")
    ax4.set_title("ACF Khung Vô Thanh -> F0 không xác định (Undefined)", fontsize=12, fontweight="bold", color="darkred")
    ax4.set_xlabel("Độ trễ lag τ (ms)")
    ax4.set_ylabel("ACF chuẩn hóa R(τ)/R(0)")
    ax4.set_ylim(-0.6, 1.05)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_signal_and_f0_contour(
    time_sig: np.ndarray,
    signal: np.ndarray,
    time_f0: np.ndarray,
    f0_contour: np.ndarray,
    ground_truth_segments: Optional[list] = None,
    f0_mean: Optional[float] = None,
    f0_std: Optional[float] = None,
    ref_f0_mean: Optional[float] = None,
    ref_f0_std: Optional[float] = None,
    title: str = "Tín hiệu âm thanh và đường bao tần số F0 (F0 Contour)",
    save_path: Optional[str] = None,
):
    """Plot waveform on top and aligned F0 contour on bottom with segmentation shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Plot waveform
    ax1.plot(time_sig, signal, color="#333333", linewidth=0.8, label="Dạng sóng tín hiệu")
    ax1.set_ylabel("Biên độ")
    ax1.set_title("Dạng sóng tín hiệu (Waveform)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Shading ground truth segments if provided
    if ground_truth_segments:
        color_map = {"v": ("#2ca02c", 0.15, "Hữu thanh (V)"), "uv": ("#ff7f0e", 0.15, "Vô thanh (UV)"), "sil": ("#7f7f7f", 0.1, "Khoảng lặng (Sil)")}
        used_labels = set()
        for start, end, lbl in ground_truth_segments:
            if lbl in color_map:
                c, alpha, name = color_map[lbl]
                lbl_name = name if lbl not in used_labels else ""
                ax1.axvspan(start, end, color=c, alpha=alpha, label=lbl_name)
                ax2.axvspan(start, end, color=c, alpha=alpha)
                used_labels.add(lbl)

    ax1.legend(loc="upper right")

    # Plot F0 contour (mask 0 values for voiced contour representation)
    f0_voiced = np.where(f0_contour > 0, f0_contour, np.nan)
    ax2.plot(time_f0, f0_voiced, "b.-", markersize=5, linewidth=1.5, label="F0 ước lượng (Hz)")

    if ref_f0_mean is not None and ref_f0_mean > 0:
        ax2.axhline(ref_f0_mean, color="red", linestyle="--", linewidth=1.5, label=f"Ground Truth F0mean = {ref_f0_mean:.1f} Hz")

    stats_str = ""
    if f0_mean is not None:
        stats_str += f"F0mean ước lượng = {f0_mean:.1f} Hz, F0std = {f0_std:.1f} Hz"
    if ref_f0_mean is not None:
        stats_str += f"\nTham chiếu Lab   = {ref_f0_mean:.1f} Hz, F0std = {ref_f0_std:.1f} Hz"

    if stats_str:
        ax2.text(0.02, 0.88, stats_str, transform=ax2.transAxes,
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))

    ax2.set_xlabel("Thời gian (giây)", fontsize=11)
    ax2.set_ylabel("Tần số F0 (Hz)", fontsize=11)
    ax2.set_ylim(50, 450)
    ax2.set_title("Đường bao tần số F0 (F0 Contour)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_distributions_and_threshold(
    values_v: np.ndarray,
    values_u: np.ndarray,
    mean_v: float,
    std_v: float,
    mean_u: float,
    std_u: float,
    threshold: float,
    method: str = "ACF",
    save_path: Optional[str] = None,
    title: str = "Phân bố biên độ cực trị Voiced/Unvoiced và Ngưỡng phân tách tối ưu T",
):
    """Plot histograms and fitted Gaussian curves for Voiced and Unvoiced frames, marking threshold T."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Range of values
    all_vals = np.concatenate([values_v, values_u])
    x_min = max(-0.2, float(np.min(all_vals)) - 0.1)
    x_max = min(1.2, float(np.max(all_vals)) + 0.1)
    x_grid = np.linspace(x_min, x_max, 500)

    # Gaussian PDF formula: 1 / (std * sqrt(2*pi)) * exp(-0.5 * ((x - mean)/std)^2)
    pdf_v = (1.0 / (std_v * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_grid - mean_v) / std_v) ** 2)
    pdf_u = (1.0 / (std_u * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_grid - mean_u) / std_u) ** 2)

    # Histograms (density=True to match PDF scale)
    bins = np.linspace(x_min, x_max, 35)
    ax.hist(values_u, bins=bins, density=True, alpha=0.45, color="#ff7f0e", edgecolor="white", label=f"Vô thanh UV (N={len(values_u)})")
    ax.hist(values_v, bins=bins, density=True, alpha=0.45, color="#2ca02c", edgecolor="white", label=f"Hữu thanh V (N={len(values_v)})")

    # Fitted Gaussian lines
    ax.plot(x_grid, pdf_u, color="#d95f02", linewidth=2.5, linestyle="-", label=f"Gaussian UV: μ={mean_u:.3f}, σ={std_u:.3f}")
    ax.plot(x_grid, pdf_v, color="#1b9e77", linewidth=2.5, linestyle="-", label=f"Gaussian V: μ={mean_v:.3f}, σ={std_v:.3f}")

    # Threshold line
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2.2, label=f"Ngưỡng tối ưu T = {threshold:.3f}")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Biên độ đỉnh cực đại chuẩn hóa của {method.upper()} [R(τ*)/R(0)]", fontsize=11)
    ax.set_ylabel("Mật độ xác suất (Probability Density)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Annotation box for summary
    summary_text = (
        f"Thuật toán: {method.upper()}\n"
        f"Khung Hữu thanh: μ_V = {mean_v:.3f}, σ_V = {std_v:.3f}\n"
        f"Khung Vô thanh : μ_U = {mean_u:.3f}, σ_U = {std_u:.3f}\n"
        f"Ngưỡng phân tách T = {threshold:.3f}"
    )
    ax.text(0.03, 0.92, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_parameter_comparison(
    time_sig: np.ndarray,
    signal: np.ndarray,
    results_20ms: dict,
    results_30ms: dict,
    eval_20ms: dict,
    eval_30ms: dict,
    results_25ms: Optional[dict] = None,
    eval_25ms: Optional[dict] = None,
    wav_name: str = "",
    save_path: Optional[str] = None,
    title: str = "Khảo sát ảnh hưởng của độ dài khung (20 ms vs 25 ms [Chuẩn] vs 30 ms)",
):
    """Plot comprehensive 3-panel comparison between 20ms, 25ms, and 30ms frame lengths:

    1. Waveform.
    2. Overlaid F0 contours (20ms, 25ms, 30ms) with Ground Truth reference.
    3. Bar chart of key metrics (Mean error, Voiced frame count, Classification Accuracy).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [1, 1.3, 1]})
    fig.suptitle(f"{title}\nFile: {wav_name}", fontsize=14, fontweight="bold")

    # 1. Waveform
    ax1.plot(time_sig, signal, color="#444444", linewidth=0.8, label="Dạng sóng tín hiệu")
    ax1.set_title("1. Dạng sóng tín hiệu (Waveform)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Biên độ")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # 2. Overlaid F0 contours
    t_20 = results_20ms["frame_times"]
    f0_20 = np.where(results_20ms["f0_contour"] > 0, results_20ms["f0_contour"], np.nan)

    t_30 = results_30ms["frame_times"]
    f0_30 = np.where(results_30ms["f0_contour"] > 0, results_30ms["f0_contour"], np.nan)

    ref_mean = eval_30ms.get("ref_f0_mean", 0.0)

    ax2.plot(t_20, f0_20, "o--", color="#ff7f0e", markersize=3, linewidth=1.1, alpha=0.7,
             label=f"Khung 20 ms (F0mean={eval_20ms['pred_f0_mean']} Hz, Err={eval_20ms['abs_error_mean']} Hz)")

    if results_25ms is not None and eval_25ms is not None:
        t_25 = results_25ms["frame_times"]
        f0_25 = np.where(results_25ms["f0_contour"] > 0, results_25ms["f0_contour"], np.nan)
        ax2.plot(t_25, f0_25, "d-", color="#2ca02c", markersize=4, linewidth=1.8, alpha=0.95,
                 label=f"Khung 25 ms [Chuẩn] (F0mean={eval_25ms['pred_f0_mean']} Hz, Err={eval_25ms['abs_error_mean']} Hz)")

    ax2.plot(t_30, f0_30, "s--", color="#1f77b4", markersize=3, linewidth=1.1, alpha=0.7,
             label=f"Khung 30 ms (F0mean={eval_30ms['pred_f0_mean']} Hz, Err={eval_30ms['abs_error_mean']} Hz)")

    if ref_mean > 0:
        ax2.axhline(ref_mean, color="black", linestyle=":", linewidth=2.0, label=f"Ground Truth F0mean = {ref_mean} Hz")

    ax2.set_title("2. So sánh đường bao tần số F0 (F0 Contours)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("F0 (Hz)")
    ax2.set_ylim(50, 420)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    # 3. Bar chart of comparison metrics
    labels_bar = ["Sai số F0mean (|ΔF0| Hz)", "Sai số F0std (|Δstd| Hz)", "Số khung Voiced phát hiện", "Độ chính xác V/UV (%)"]
    vals_20 = [
        eval_20ms["abs_error_mean"],
        eval_20ms["abs_error_std"],
        results_20ms["num_voiced"],
        eval_20ms["classification_accuracy"],
    ]
    vals_30 = [
        eval_30ms["abs_error_mean"],
        eval_30ms["abs_error_std"],
        results_30ms["num_voiced"],
        eval_30ms["classification_accuracy"],
    ]

    x = np.arange(len(labels_bar))

    if results_25ms is not None and eval_25ms is not None:
        vals_25 = [
            eval_25ms["abs_error_mean"],
            eval_25ms["abs_error_std"],
            results_25ms["num_voiced"],
            eval_25ms["classification_accuracy"],
        ]
        width = 0.25
        rects1 = ax3.bar(x - width, vals_20, width, label="Khung 20 ms", color="#ff7f0e", alpha=0.85, edgecolor="gray")
        rects_mid = ax3.bar(x, vals_25, width, label="Khung 25 ms (Chuẩn)", color="#2ca02c", alpha=0.9, edgecolor="black")
        rects2 = ax3.bar(x + width, vals_30, width, label="Khung 30 ms", color="#1f77b4", alpha=0.85, edgecolor="gray")
        all_rects = [rects1, rects_mid, rects2]
    else:
        width = 0.35
        rects1 = ax3.bar(x - width / 2, vals_20, width, label="Khung 20 ms", color="#ff7f0e", alpha=0.85, edgecolor="gray")
        rects2 = ax3.bar(x + width / 2, vals_30, width, label="Khung 30 ms", color="#1f77b4", alpha=0.85, edgecolor="gray")
        all_rects = [rects1, rects2]

    ax3.set_title("3. Bảng số liệu đối sánh định lượng", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_bar, fontsize=10)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    # Value labels on bars
    for rect_group in all_rects:
        for r in rect_group:
            h = r.get_height()
            ax3.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_plugin_contour_comparison(
    time_sig: np.ndarray,
    signal: np.ndarray,
    time_f0: np.ndarray,
    f0_baseline: np.ndarray,
    f0_enhanced: np.ndarray,
    eval_baseline: dict,
    eval_enhanced: dict,
    ground_truth_segments: Optional[list] = None,
    file_name: str = "",
    plugin_names_str: str = "Combined Plugins",
    save_path: Optional[str] = None,
):
    """Plot a 3-panel comparison between Baseline and Plugin-enhanced F0 contours."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"So sánh đường bao F0: Baseline vs. Cải tiến Plugin ({file_name})", fontsize=14, fontweight="bold")

    # 1. Waveform
    ax1.plot(time_sig, signal, color="#444444", linewidth=0.8, label="Dạng sóng tín hiệu")
    ax1.set_ylabel("Biên độ")
    ax1.set_title("1. Dạng sóng tín hiệu (Waveform)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Shading ground truth segments
    if ground_truth_segments:
        color_map = {
            "v": ("#2ca02c", 0.15, "Hữu thanh (V)"),
            "uv": ("#ff7f0e", 0.15, "Vô thanh (UV)"),
            "sil": ("#7f7f7f", 0.10, "Khoảng lặng (Sil)"),
        }
        used_labels = set()
        for start, end, lbl in ground_truth_segments:
            if lbl in color_map:
                c, alpha, name = color_map[lbl]
                lbl_name = name if lbl not in used_labels else ""
                ax1.axvspan(start, end, color=c, alpha=alpha, label=lbl_name)
                ax2.axvspan(start, end, color=c, alpha=alpha)
                ax3.axvspan(start, end, color=c, alpha=alpha)
                used_labels.add(lbl)
    ax1.legend(loc="upper right")

    ref_mean = eval_baseline.get("ref_f0_mean", 0.0)

    # 2. Baseline F0 contour
    f0_base_v = np.where(f0_baseline > 0, f0_baseline, np.nan)
    ax2.plot(time_f0, f0_base_v, ".-", color="#d62728", markersize=5, linewidth=1.3, label="Baseline F0 (ACF gốc)")
    if ref_mean > 0:
        ax2.axhline(ref_mean, color="black", linestyle=":", linewidth=1.5, label=f"Ground Truth = {ref_mean:.1f} Hz")
    ax2.set_ylabel("F0 (Hz)")
    ax2.set_ylim(50, 420)
    ax2.set_title(
        f"2. Baseline (Chưa dùng Plugin) -> F0mean = {eval_baseline['pred_f0_mean']:.1f} Hz (Sai số: {eval_baseline['abs_error_mean']:.2f} Hz | F1: {eval_baseline['voiced_f1']:.1f}%)",
        fontsize=11, fontweight="bold", color="#d62728",
    )
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # 3. Enhanced F0 contour
    f0_enh_v = np.where(f0_enhanced > 0, f0_enhanced, np.nan)
    ax3.plot(time_f0, f0_enh_v, ".-", color="#1f77b4", markersize=5, linewidth=1.5, label=f"Enhanced F0 ({plugin_names_str})")
    if ref_mean > 0:
        ax3.axhline(ref_mean, color="black", linestyle=":", linewidth=1.5, label=f"Ground Truth = {ref_mean:.1f} Hz")
    ax3.set_xlabel("Thời gian (giây)", fontsize=11)
    ax3.set_ylabel("F0 (Hz)")
    ax3.set_ylim(50, 420)
    ax3.set_title(
        f"3. Enhanced (Áp dụng {plugin_names_str}) -> F0mean = {eval_enhanced['pred_f0_mean']:.1f} Hz (Sai số: {eval_enhanced['abs_error_mean']:.2f} Hz | F1: {eval_enhanced['voiced_f1']:.1f}%)",
        fontsize=11, fontweight="bold", color="#1f77b4",
    )
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_combinations_ranking(
    summary_records: list,
    save_path: Optional[str] = None,
    title: str = "Đánh giá toàn bộ 8 tổ hợp Plugins trên tập Kiểm thử",
):
    """Plot horizontal bar charts comparing Mean Absolute Error and F1-Score across all plugin combinations."""
    names = [rec["config_name"] for rec in summary_records]
    errs = [rec["average_error_hz"] for rec in summary_records]
    f1s = [rec["average_voiced_f1_pct"] for rec in summary_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    y_pos = np.arange(len(names))

    # Panel 1: Error (Lower is better)
    min_err = min(errs)
    colors_err = ["#2ca02c" if e == min_err else "#1f77b4" for e in errs]
    bars1 = ax1.barh(y_pos, errs, color=colors_err, alpha=0.85, edgecolor="gray")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()  # top-down
    ax1.set_xlabel("Sai số trung bình |ΔF0| (Hz) - [Thấp hơn là tốt hơn]", fontsize=11, fontweight="bold")
    ax1.set_title("1. So sánh Sai số tuyệt đối trung bình (|ΔF0|)", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f"{w:.2f} Hz", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold")

    # Panel 2: F1-Score (Higher is better)
    max_f1 = max(f1s)
    colors_f1 = ["#2ca02c" if f == max_f1 else "#ff7f0e" for f in f1s]
    bars2 = ax2.barh(y_pos, f1s, color=colors_f1, alpha=0.85, edgecolor="gray")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # hide duplicate labels
    ax2.invert_yaxis()
    ax2.set_xlim(85, 95)
    ax2.set_xlabel("Voiced F1-Score (%) - [Cao hơn là tốt hơn]", fontsize=11, fontweight="bold")
    ax2.set_title("2. So sánh Voiced F1-Score (%)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w:.2f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig
