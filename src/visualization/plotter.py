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

    stats_str = ""
    if f0_mean is not None:
        stats_str += f"F0mean = {f0_mean:.1f} Hz, F0std = {f0_std:.1f} Hz"
    if ref_f0_mean is not None:
        stats_str += f"\n(Tham chiếu Lab: F0mean = {ref_f0_mean:.1f} Hz, F0std = {ref_f0_std:.1f} Hz)"

    if stats_str:
        ax2.text(0.02, 0.90, stats_str, transform=ax2.transAxes,
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"))

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
