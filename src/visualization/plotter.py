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


def plot_frame_amdf_comparison(
    voiced_frame: np.ndarray,
    voiced_amdf: np.ndarray,
    voiced_f0: float,
    voiced_lag: int,
    voiced_dip: float,
    unvoiced_frame: np.ndarray,
    unvoiced_amdf: np.ndarray,
    unvoiced_lag: int,
    unvoiced_dip: float,
    sample_rate: int,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "So sánh Hàm Hiệu Độ Lớn Trung Bình (AMDF): Khung Hữu Thanh vs Vô Thanh",
):
    """Plot a comprehensive 2x2 comparison between a Voiced frame and an Unvoiced frame for AMDF:

    - Top Left: Voiced Frame Waveform (shows quasi-periodicity).
    - Bottom Left: Voiced Frame Normalized AMDF (shows deep local dip at pitch period T0).
    - Top Right: Unvoiced Frame Waveform (noise-like, non-periodic).
    - Bottom Right: Unvoiced Frame Normalized AMDF (stays high, no deep dip).
    """
    n_v = len(voiced_frame)
    n_uv = len(unvoiced_frame)

    t_v = np.arange(n_v) / sample_rate * 1000.0  # ms
    t_uv = np.arange(n_uv) / sample_rate * 1000.0  # ms

    lag_v_ms = np.arange(len(voiced_amdf)) / sample_rate * 1000.0  # ms
    lag_uv_ms = np.arange(len(unvoiced_amdf)) / sample_rate * 1000.0  # ms

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

    # Subplot 2: Voiced AMDF
    ax2 = axes[1, 0]
    ax2.plot(lag_v_ms, voiced_amdf, color="#1f77b4", linewidth=1.5, label="Normalized AMDF D(τ)")
    ax2.axvspan(lag_min_ms, lag_max_ms, color="green", alpha=0.15, label=f"Vùng tìm F0 [{f0_min:.0f}-{f0_max:.0f} Hz]")
    t0_ms = voiced_lag / sample_rate * 1000.0
    ax2.plot(t0_ms, voiced_dip, "ro", markersize=8, label=f"Đáy T0 = {t0_ms:.2f} ms\nF0 = {voiced_f0:.1f} Hz (Dip: {voiced_dip:.3f})")
    ax2.axvline(t0_ms, color="red", linestyle="--", alpha=0.7)
    if threshold is not None:
        ax2.axhline(threshold, color="purple", linestyle=":", linewidth=1.8, label=f"Ngưỡng T = {threshold:.3f}")
    ax2.set_title(f"AMDF Khung Hữu Thanh -> F0 = {voiced_f0:.1f} Hz (Đáy cực tiểu sâu)", fontsize=12, fontweight="bold", color="darkgreen")
    ax2.set_xlabel("Độ trễ lag τ (ms)")
    ax2.set_ylabel("AMDF chuẩn hóa D(τ)")
    ax2.set_ylim(-0.05, 1.05)
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

    # Subplot 4: Unvoiced AMDF
    ax4 = axes[1, 1]
    ax4.plot(lag_uv_ms, unvoiced_amdf, color="#d62728", linewidth=1.5, label="Normalized AMDF D(τ)")
    ax4.axvspan(lag_min_ms, lag_max_ms, color="orange", alpha=0.15, label=f"Vùng tìm F0 [{f0_min:.0f}-{f0_max:.0f} Hz]")
    unvoiced_t0_ms = unvoiced_lag / sample_rate * 1000.0
    ax4.plot(unvoiced_t0_ms, unvoiced_dip, "kx", markersize=8, markeredgewidth=2, label=f"Cực tiểu cục bộ ({unvoiced_dip:.3f} > Ngưỡng)")
    if threshold is not None:
        ax4.axhline(threshold, color="purple", linestyle=":", linewidth=1.8, label=f"Ngưỡng T = {threshold:.3f}")
    ax4.set_title("AMDF Khung Vô Thanh -> F0 không xác định (Không có đáy sâu)", fontsize=12, fontweight="bold", color="darkred")
    ax4.set_xlabel("Độ trễ lag τ (ms)")
    ax4.set_ylabel("AMDF chuẩn hóa D(τ)")
    ax4.set_ylim(-0.05, 1.05)
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

    if method.lower() == "amdf":
        ax.set_xlabel(f"Biên độ đáy cực tiểu chuẩn hóa của AMDF [D(τ*)]", fontsize=11)
    else:
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
    """Plot horizontal bar charts comparing Mean Absolute Error, Std Error, F1-Score, and Accuracy across all plugin combinations."""
    names = [rec["config_name"] for rec in summary_records]
    errs = [rec["average_error_hz"] for rec in summary_records]
    std_errs = [rec.get("average_std_error", 0.0) for rec in summary_records]
    f1s = [rec["average_voiced_f1_pct"] for rec in summary_records]
    accs = [rec["average_accuracy_pct"] for rec in summary_records]

    fig_height = max(8, len(names) * 0.42)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, fig_height))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    y_pos = np.arange(len(names))
    label_fontsize = 8.5 if len(names) > 20 else 10
    annot_fontsize = 8.0 if len(names) > 20 else 9.0

    # Panel 1: Error (Lower is better)
    min_err = min(errs)
    colors_err = ["#2ca02c" if e == min_err else "#1f77b4" for e in errs]
    bars1 = ax1.barh(y_pos, errs, color=colors_err, alpha=0.85, edgecolor="gray")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=label_fontsize)
    ax1.invert_yaxis()  # top-down
    ax1.set_xlabel("Sai số tuyệt đối trung bình |ΔF0| (Hz) - [Thấp hơn là tốt hơn]", fontsize=10, fontweight="bold")
    ax1.set_title("1. Sai số F0 TB (|ΔF0|)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f"{w:.2f} Hz", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=annot_fontsize, fontweight="bold")

    # Panel 2: Std Error (Lower is better)
    min_std = min(std_errs)
    colors_std = ["#2ca02c" if s == min_std else "#9467bd" for s in std_errs]
    bars2 = ax2.barh(y_pos, std_errs, color=colors_std, alpha=0.85, edgecolor="gray")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # hide duplicate labels
    ax2.invert_yaxis()
    ax2.set_xlabel("Sai số độ lệch chuẩn TB |Δstd| - [Thấp hơn là tốt hơn]", fontsize=10, fontweight="bold")
    ax2.set_title("2. Sai số Độ lệch chuẩn TB (|Δstd|)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w:.2f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=annot_fontsize, fontweight="bold")

    # Panel 3: F1-Score (Higher is better)
    max_f1 = max(f1s)
    colors_f1 = ["#2ca02c" if f == max_f1 else "#ff7f0e" for f in f1s]
    bars3 = ax3.barh(y_pos, f1s, color=colors_f1, alpha=0.85, edgecolor="gray")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])  # hide duplicate labels
    ax3.invert_yaxis()
    f1_min = max(0.0, float(np.floor(min(f1s) - 2.0)))
    f1_max = min(100.0, float(np.ceil(max(f1s) + 2.0)))
    ax3.set_xlim(f1_min, f1_max)
    ax3.set_xlabel("Voiced F1-Score (%) - [Cao hơn là tốt hơn]", fontsize=10, fontweight="bold")
    ax3.set_title("3. Voiced F1-Score (%)", fontsize=11, fontweight="bold")
    ax3.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars3:
        w = bar.get_width()
        ax3.annotate(f"{w:.2f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=annot_fontsize, fontweight="bold")

    # Panel 4: Accuracy (Higher is better)
    max_acc = max(accs)
    colors_acc = ["#2ca02c" if a == max_acc else "#17becf" for a in accs]
    bars4 = ax4.barh(y_pos, accs, color=colors_acc, alpha=0.85, edgecolor="gray")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([])  # hide duplicate labels
    ax4.invert_yaxis()
    acc_min = max(0.0, float(np.floor(min(accs) - 2.0)))
    acc_max = min(100.0, float(np.ceil(max(accs) + 2.0)))
    ax4.set_xlim(acc_min, acc_max)
    ax4.set_xlabel("V/UV Accuracy (%) - [Cao hơn là tốt hơn]", fontsize=10, fontweight="bold")
    ax4.set_title("4. V/UV Accuracy (%)", fontsize=11, fontweight="bold")
    ax4.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars4:
        w = bar.get_width()
        ax4.annotate(f"{w:.2f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=annot_fontsize, fontweight="bold")

    # Ensure margin for annotations on all panels
    for ax in [ax1, ax2]:
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.15)
    for ax in [ax3, ax4]:
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_noise_robustness_curves(
    results_list: list,
    save_path: Optional[str] = None,
    title: str = "Khảo sát Độ Bền Vững Kháng Nhiễu (Noise Robustness): ACF vs. AMDF",
):
    """Plot dual-panel degradation curves (MAE and F1-Score) across various SNR levels."""
    labels = [r["snr_label"] for r in results_list]
    x = np.arange(len(labels))

    err_acf_b = [r["acf_base"]["average_error_hz"] for r in results_list]
    err_amdf_b = [r["amdf_base"]["average_error_hz"] for r in results_list]
    err_acf_e = [r["acf_enh"]["average_error_hz"] for r in results_list]
    err_amdf_e = [r["amdf_enh"]["average_error_hz"] for r in results_list]

    f1_acf_b = [r["acf_base"]["average_voiced_f1_pct"] for r in results_list]
    f1_amdf_b = [r["amdf_base"]["average_voiced_f1_pct"] for r in results_list]
    f1_acf_e = [r["acf_enh"]["average_voiced_f1_pct"] for r in results_list]
    f1_amdf_e = [r["amdf_enh"]["average_voiced_f1_pct"] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Error vs SNR (Lower is better)
    ax1.plot(x, err_acf_b, "o--", color="#1f77b4", linewidth=2.0, markersize=7, label="ACF (Baseline)")
    ax1.plot(x, err_amdf_b, "s--", color="#d62728", linewidth=2.0, markersize=7, label="AMDF (Baseline)")
    ax1.plot(x, err_acf_e, "^-", color="#08519c", linewidth=2.5, markersize=8, label="ACF (Tổ hợp Nâng cao Tối ưu)")
    ax1.plot(x, err_amdf_e, "v-", color="#a50f15", linewidth=2.5, markersize=8, label="AMDF (Tổ hợp Nâng cao Tối ưu)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Mức tỷ số Tín hiệu trên Nhiễu (SNR) [Giảm dần độ sạch →]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Sai số tuyệt đối trung bình |ΔF0| (Hz)", fontsize=11, fontweight="bold")
    ax1.set_title("1. Sai số F0 (Hz) theo mức nhiễu [Thấp hơn là tốt hơn]", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    # Panel 2: F1-Score vs SNR (Higher is better)
    ax2.plot(x, f1_acf_b, "o--", color="#1f77b4", linewidth=2.0, markersize=7, label="ACF (Baseline)")
    ax2.plot(x, f1_amdf_b, "s--", color="#d62728", linewidth=2.0, markersize=7, label="AMDF (Baseline)")
    ax2.plot(x, f1_acf_e, "^-", color="#08519c", linewidth=2.5, markersize=8, label="ACF (Tổ hợp Nâng cao Tối ưu)")
    ax2.plot(x, f1_amdf_e, "v-", color="#a50f15", linewidth=2.5, markersize=8, label="AMDF (Tổ hợp Nâng cao Tối ưu)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Mức tỷ số Tín hiệu trên Nhiễu (SNR) [Giảm dần độ sạch →]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Voiced F1-Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("2. Độ chính xác phân loại V/UV theo mức nhiễu [Cao hơn là tốt hơn]", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower left", frameon=True, fontsize=9.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_noise_mechanism_demo(
    clean_frame: np.ndarray,
    noisy_frame: np.ndarray,
    sample_rate: int,
    f0_ref: float,
    snr_db: float = 0.0,
    save_path: Optional[str] = None,
):
    """Plot mechanism demonstration comparing ACF and AMDF response under 0dB SNR."""
    from src.core.acf import compute_acf
    from src.core.amdf import compute_amdf

    acf_clean = compute_acf(clean_frame, mode="normalized")
    acf_noisy = compute_acf(noisy_frame, mode="normalized")

    amdf_clean = compute_amdf(clean_frame, mode="normalized")
    amdf_noisy = compute_amdf(noisy_frame, mode="normalized")

    t_frame = np.arange(len(clean_frame)) / sample_rate * 1000.0  # ms
    tau_ms = np.arange(len(acf_clean)) / sample_rate * 1000.0  # ms
    t0_ref_ms = (1.0 / f0_ref) * 1000.0

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.22)
    fig.suptitle(f"Minh họa Cơ chế Kháng Nhiễu tại Khung Hữu Thanh (SNR = {snr_db:.0f} dB)", fontsize=14, fontweight="bold")

    # Subplot 1: Waveforms (Span both columns)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_frame, noisy_frame, color="#ff7f0e", alpha=0.65, label=f"Tín hiệu lẫn nhiễu ({snr_db:.0f} dB AWGN)")
    ax0.plot(t_frame, clean_frame, color="#1f77b4", linewidth=1.8, label="Tín hiệu gốc sạch (Voiced Frame)")
    ax0.set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    ax0.set_ylabel("Biên độ", fontsize=10, fontweight="bold")
    ax0.set_title("Dạng sóng miền thời gian: Khung sạch vs Khung nhiễu cực nặng (0 dB)", fontsize=11, fontweight="bold")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right", frameon=True)

    # Subplot 2: ACF comparison
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(tau_ms, acf_clean, color="#1f77b4", linewidth=2.0, label="ACF Khung sạch")
    ax1.plot(tau_ms, acf_noisy, color="#08519c", linewidth=2.0, linestyle="--", label=f"ACF Khung nhiễu ({snr_db:.0f} dB)")
    ax1.axvline(t0_ref_ms, color="red", linestyle=":", linewidth=2, label=f"Chu kỳ thật T0 = {t0_ref_ms:.2f} ms ({f0_ref:.1f} Hz)")
    ax1.set_xlim(0, max(tau_ms))
    ax1.set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Hàm tương quan chuẩn hóa R(τ)", fontsize=10, fontweight="bold")
    ax1.set_title("ACF: Đỉnh chu kỳ T0 vẫn nhô cao rõ rệt nhờ nhiễu tự triệt tiêu", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Subplot 3: AMDF comparison
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(tau_ms, amdf_clean, color="#d62728", linewidth=2.0, label="AMDF Khung sạch")
    ax2.plot(tau_ms, amdf_noisy, color="#800026", linewidth=2.0, linestyle="--", label=f"AMDF Khung nhiễu ({snr_db:.0f} dB)")
    ax2.axvline(t0_ref_ms, color="blue", linestyle=":", linewidth=2, label=f"Chu kỳ thật T0 = {t0_ref_ms:.2f} ms ({f0_ref:.1f} Hz)")
    ax2.set_xlim(0, max(tau_ms))
    ax2.set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Hàm hiệu chuẩn hóa D(τ)", fontsize=10, fontweight="bold")
    ax2.set_title("AMDF: Đáy cực tiểu T0 bị sàn nhiễu nâng cao, lấp phẳng", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_ptdb_benchmark_summary(
    summary_metrics: dict,
    sample_contour: Optional[dict] = None,
    save_path: Optional[str] = "outputs/figures/06_ptdb_benchmark.png",
):
    """Plot PTDB-TUG benchmark summary: VDE, GPE, FFE bar charts and representative pitch contour overlay."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)

    systems = list(summary_metrics.keys())
    vde_vals = [summary_metrics[s]["vde"] for s in systems]
    gpe_vals = [summary_metrics[s]["gpe"] for s in systems]
    ffe_vals = [summary_metrics[s]["ffe"] for s in systems]

    colors = ["#9ecae1", "#2171b5", "#fc9272", "#cb181d"]
    if len(colors) < len(systems):
        colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))

    # Ax0: VDE
    ax0 = fig.add_subplot(gs[0, 0])
    bars0 = ax0.bar(systems, vde_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax0.set_title("1. Voicing Decision Error (VDE %)\n(Càng thấp càng tốt)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("VDE (%)", fontsize=10, fontweight="bold")
    ax0.tick_params(axis="x", rotation=20)
    for b in bars0:
        h = b.get_height()
        ax0.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2.0, h),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Ax1: GPE
    ax1 = fig.add_subplot(gs[0, 1])
    bars1 = ax1.bar(systems, gpe_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_title("2. Gross Pitch Error (GPE %)\n(Sai số pitch > 20%, càng thấp càng tốt)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("GPE (%)", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", rotation=20)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2.0, h),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Ax2: FFE
    ax2 = fig.add_subplot(gs[0, 2])
    bars2 = ax2.bar(systems, ffe_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_title("3. F0 Frame Error (FFE %)\n(Tổng hợp lỗi VDE + GPE)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("FFE (%)", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=20)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2.0, h),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Row 1: Pitch contour overlay
    if sample_contour is not None:
        ax_cont = fig.add_subplot(gs[1, :])
        t = sample_contour["times"]
        gt_f0 = sample_contour["gt_f0"]
        base_f0 = sample_contour["base_f0"]
        enh_f0 = sample_contour["enh_f0"]
        title_cont = sample_contour.get("title", "Đối sánh Quỹ đạo Pitch trên câu mẫu PTDB-TUG")

        gt_mask = gt_f0 > 0
        ax_cont.scatter(t[gt_mask], gt_f0[gt_mask], color="#2ca02c", s=18, label="Ground Truth (Laryngograph EGG)", zorder=3)

        base_mask = base_f0 > 0
        ax_cont.plot(t[base_mask], base_f0[base_mask], color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8, label="Baseline (No plugins)", zorder=2)

        enh_mask = enh_f0 > 0
        ax_cont.plot(t[enh_mask], enh_f0[enh_mask], color="#1f77b4", linewidth=2.0, alpha=0.9, label="Enhanced (With plugins)", zorder=4)

        ax_cont.set_title(title_cont, fontsize=12, fontweight="bold")
        ax_cont.set_xlabel("Thời gian (s)", fontsize=10, fontweight="bold")
        ax_cont.set_ylabel("Tần số F0 (Hz)", fontsize=10, fontweight="bold")
        ax_cont.set_ylim(50, 420)
        ax_cont.legend(loc="upper right", frameon=True, fontsize=10)
        ax_cont.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("ĐÁNH GIÁ CHUẨN QUỐC TẾ TRÊN CƠ SỞ DỮ LIỆU PTDB-TUG (LARYNGOGRAPH GROUND TRUTH)", fontsize=14, fontweight="bold", y=0.98)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_ptdb_noise_robustness_curves(
    snr_results: list,
    save_path: Optional[str] = "outputs/figures/09_ptdb_noise_robustness_curves.png",
    title: str = "ĐÁNH GIÁ ĐỘ BỀN VỮNG KHÁNG NHIỄU TRÊN PTDB-TUG (LARYNGOGRAPH EGG GROUND TRUTH)",
):
    """Plot 3-panel degradation curves (GPE %, VDE %, FFE %) across SNR levels on PTDB-TUG."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    labels = [r["snr_label"] for r in snr_results]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    metrics = [
        ("gpe", "1. Gross Pitch Error (GPE %)\n(Lỗi thô > 20%, càng thấp càng tốt)", axes[0]),
        ("vde", "2. Voicing Decision Error (VDE %)\n(Lỗi phân loại V/UV, càng thấp càng tốt)", axes[1]),
        ("ffe", "3. F0 Frame Error (FFE %)\n(Tổng hợp lỗi VDE + GPE)", axes[2]),
    ]

    for key, panel_title, ax in metrics:
        y_acf_b = [r["Baseline ACF"][key] for r in snr_results]
        y_acf_e = [r["Enhanced ACF"][key] for r in snr_results]
        y_amdf_b = [r["Baseline AMDF"][key] for r in snr_results]
        y_amdf_e = [r["Enhanced AMDF"][key] for r in snr_results]

        ax.plot(x, y_acf_b, "o--", color="#1f77b4", linewidth=2.0, markersize=7, label="ACF Baseline")
        ax.plot(x, y_acf_e, "^-", color="#08519c", linewidth=2.5, markersize=8, label="ACF Enhanced (All Plugins)")
        ax.plot(x, y_amdf_b, "s--", color="#d62728", linewidth=2.0, markersize=7, label="AMDF Baseline")
        ax.plot(x, y_amdf_e, "v-", color="#800026", linewidth=2.5, markersize=8, label="AMDF Enhanced (All Plugins)")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mức tỷ số Tín hiệu trên Nhiễu (SNR)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{key.upper()} (%)", fontsize=10, fontweight="bold")
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


