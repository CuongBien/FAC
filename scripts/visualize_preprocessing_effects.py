#!/usr/bin/env python3
"""Script: Trực quan hóa chi tiết tác động của các kỹ thuật Tiền xử lý (Pre-processing):
1. Bandpass Filter (Butterworth bậc 2, 70 - 900 Hz, zero-phase filtfilt)
2. Center Clipping (Cắt gọt trung tâm - Sondhi 1968, ratio = 0.40)
3. Chuỗi tiền xử lý tích hợp (Unified Pre-processing Pipeline)

Xuất 3 đồ thị minh họa trực quan:
- outputs/figures/preprocessing_bandpass_demo.png
- outputs/figures/preprocessing_center_clipping_demo.png
- outputs/figures/preprocessing_pipeline_comprehensive.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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
from src.core.acf import compute_acf
from src.plugins.pre_processing.bandpass_filter import BandpassFilterPlugin
from src.plugins.pre_processing.center_clipping import CenterClippingPlugin

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_bandpass_demo(
    signal: np.ndarray,
    sample_rate: int,
    t_start: float = 1.0,
    duration: float = 0.08,
    save_path: str = "outputs/figures/preprocessing_bandpass_demo.png",
):
    """Minh họa tác động của Bộ lọc thông dải Bandpass Filter (Butterworth 2 chiều Zero-Phase)."""
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0, order=2)
    filtered_signal = bp.pre_process_signal(signal, sample_rate)

    # Trích xuất đoạn ngắn để xem rõ dao động tuần hoàn
    start_idx = int(t_start * sample_rate)
    end_idx = start_idx + int(duration * sample_rate)
    time_ms = (np.arange(end_idx - start_idx) / sample_rate) * 1000.0

    raw_segment = signal[start_idx:end_idx]
    filtered_segment = filtered_signal[start_idx:end_idx]

    # Tính phổ biên độ FFT (Frequency Spectrum)
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    fft_raw = np.abs(np.fft.rfft(raw_segment * np.hamming(len(raw_segment)), n=n_fft))
    fft_filt = np.abs(np.fft.rfft(filtered_segment * np.hamming(len(filtered_segment)), n=n_fft))

    fft_raw_db = 20 * np.log10(fft_raw / (np.max(fft_raw) + 1e-12) + 1e-12)
    fft_filt_db = 20 * np.log10(fft_filt / (np.max(fft_filt) + 1e-12) + 1e-12)

    # Tính ACF của 1 khung
    frame_len = int(0.030 * sample_rate)
    acf_raw = compute_acf(raw_segment[:frame_len], mode="normalized")
    acf_filt = compute_acf(filtered_segment[:frame_len], mode="normalized")
    tau_ms = (np.arange(len(acf_raw)) / sample_rate) * 1000.0

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.22)
    fig.suptitle(
        "MINH HỌA TÁC ĐỘNG CỦA BỘ LỌC DẢI THÔNG BANDPASS FILTER (70 - 900 Hz, ZERO-PHASE)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # 1. Miền thời gian (Dạng sóng tín hiệu)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(time_ms, raw_segment, color="#999999", linewidth=1.2, alpha=0.8, label="Tín hiệu gốc (Gồm DC trôi dạt & nhiễu hài cao tần)")
    ax0.plot(time_ms, filtered_segment, color="#1f77b4", linewidth=1.8, label="Sau Bandpass Filter (70 - 900 Hz Zero-Phase)")
    ax0.set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    ax0.set_ylabel("Biên độ", fontsize=10, fontweight="bold")
    ax0.set_title("1. Dạng sóng miền thời gian: Lọc 2 chiều (filtfilt) triệt tiêu hoàn toàn độ lệch pha (ϕ = 0), bảo toàn 100% vị trí đỉnh/đáy", fontsize=11, fontweight="bold")
    ax0.set_xlim(0, max(time_ms))
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right", frameon=True, fontsize=10)

    # 2. Miền tần số (Phổ biên độ FFT)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(freqs, fft_raw_db, color="#999999", linewidth=1.2, alpha=0.7, label="Phổ gốc (Raw FFT)")
    ax1.plot(freqs, fft_filt_db, color="#2ca02c", linewidth=1.8, label="Phổ sau lọc (Filtered FFT)")
    ax1.axvspan(70, 900, color="#2ca02c", alpha=0.10, label="Dải thông cho phép (70 - 900 Hz)")
    ax1.axvline(70, color="red", linestyle=":", linewidth=1.5, label="Cắt dưới (70 Hz)")
    ax1.axvline(900, color="darkorange", linestyle=":", linewidth=1.5, label="Cắt trên (900 Hz)")
    ax1.set_xlabel("Tần số (Hz)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Biên độ chuẩn hóa (dB)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, min(3000, sample_rate // 2))
    ax1.set_ylim(-65, 5)
    ax1.set_title("2. Miền tần số: Triệt tiêu trôi DC (< 70 Hz) & dập tắt nhiễu cao tần (> 900 Hz)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    # 3. Hàm tự tương quan ACF
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(tau_ms, acf_raw, color="#999999", linewidth=1.5, linestyle="--", label="ACF Tín hiệu gốc")
    ax2.plot(tau_ms, acf_filt, color="#08519c", linewidth=2.0, label="ACF Sau lọc thông dải")
    ax2.set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Hàm tương quan R(τ)", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(tau_ms))
    ax2.set_title("3. Hàm ACF: Dạng sóng tương quan mượt mà hơn, giảm gợn sóng nhiễu bậc cao", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã xuất biểu đồ Bandpass Filter: {save_path}")


def plot_center_clipping_demo(
    signal: np.ndarray,
    sample_rate: int,
    t_start: float = 1.0,
    duration: float = 0.04,
    clipping_ratio: float = 0.40,
    save_path: str = "outputs/figures/preprocessing_center_clipping_demo.png",
):
    """Minh họa tác động của Cắt gọt trung tâm Center Clipping (Sondhi 1968, ratio = 0.40)."""
    start_idx = int(t_start * sample_rate)
    frame_len = int(duration * sample_rate)
    frame = signal[start_idx : start_idx + frame_len].copy()
    time_ms = (np.arange(frame_len) / sample_rate) * 1000.0

    # Áp dụng Center Clipping
    clip_plugin = CenterClippingPlugin(clipping_ratio=clipping_ratio, mode="standard")
    clipped_frame = clip_plugin.pre_process_frame(frame, sample_rate)

    max_val = np.max(np.abs(frame))
    c_l = clipping_ratio * max_val

    # Tính hàm ACF tương ứng
    acf_raw = compute_acf(frame, mode="normalized")
    acf_clipped = compute_acf(clipped_frame, mode="normalized")
    tau_ms = (np.arange(len(acf_raw)) / sample_rate) * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5))
    fig.suptitle(
        f"MINH HỌA TÁC ĐỘNG CỦA CẮT GỌT TRUNG TÂM CENTER CLIPPING (SONDHI 1968, RATIO = {clipping_ratio:.2f})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # 1. Khung gốc với 2 vạch ngưỡng cắt
    axes[0, 0].plot(time_ms, frame, color="#1f77b4", linewidth=1.8, label="Dạng sóng khung hữu thanh (Voiced Frame)")
    axes[0, 0].axhline(c_l, color="#d62728", linestyle="--", linewidth=1.5, label=f"Ngưỡng trên +C_L = +{clipping_ratio*100:.0f}% max")
    axes[0, 0].axhline(-c_l, color="#d62728", linestyle="--", linewidth=1.5, label=f"Ngưỡng dưới -C_L = -{clipping_ratio*100:.0f}% max")
    axes[0, 0].axhspan(-c_l, c_l, color="#d62728", alpha=0.12, label="Vùng trung tâm bị gọt bỏ về 0")
    axes[0, 0].set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Biên độ", fontsize=10, fontweight="bold")
    axes[0, 0].set_title("1. Khung gốc: Chứa nhiều dao động phụ Formant F1 ở dải giữa [-C_L, +C_L]", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlim(0, max(time_ms))
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    # 2. Sau khi Center Clipping
    axes[0, 1].plot(time_ms, clipped_frame, color="#2ca02c", linewidth=2.0, label="Sau Center Clipping (y[n])")
    axes[0, 1].axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
    axes[0, 1].set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Biên độ sau cắt", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("2. Sau Center Clipping: Gọt phẳng dải giữa về 0, chỉ còn các xung đóng thanh môn (GCI)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlim(0, max(time_ms))
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    # 3. Hàm ACF trước khi Center Clipping
    axes[1, 0].plot(tau_ms, acf_raw, color="#d62728", linewidth=1.8, label="ACF Khung gốc")
    axes[1, 0].set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Hàm tương quan R(τ)", fontsize=10, fontweight="bold")
    axes[1, 0].set_title("3. ACF gốc: Nhiều đỉnh phụ Formant nhô cao, nguy cơ bắt nhầm đỉnh (Octave Halving)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlim(0, max(tau_ms))
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right", frameon=True, fontsize=9)

    # 4. Hàm ACF sau khi Center Clipping
    axes[1, 1].plot(tau_ms, acf_clipped, color="#006837", linewidth=2.0, label="ACF Sau Center Clipping")
    axes[1, 1].set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Hàm tương quan R(τ)", fontsize=10, fontweight="bold")
    axes[1, 1].set_title("4. ACF sau Clipping: Toàn bộ đỉnh phụ formant bị đập phẳng, đỉnh chu kỳ T0 nổi bật tuyệt đối", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlim(0, max(tau_ms))
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã xuất biểu đồ Center Clipping: {save_path}")


def plot_comprehensive_pipeline(
    signal: np.ndarray,
    sample_rate: int,
    t_start: float = 1.0,
    duration: float = 0.04,
    save_path: str = "outputs/figures/preprocessing_pipeline_comprehensive.png",
):
    """Minh họa toàn diện chuỗi tích hợp 3 giai đoạn: Gốc -> Bandpass Filter -> Center Clipping."""
    start_idx = int(t_start * sample_rate)
    frame_len = int(duration * sample_rate)

    # 1. Gốc
    frame_raw = signal[start_idx : start_idx + frame_len].copy()

    # 2. Qua Bandpass Filter
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0, order=2)
    sig_bp = bp.pre_process_signal(signal, sample_rate)
    frame_bp = sig_bp[start_idx : start_idx + frame_len].copy()

    # 3. Qua Center Clipping
    clip = CenterClippingPlugin(clipping_ratio=0.40, mode="standard")
    frame_final = clip.pre_process_frame(frame_bp, sample_rate)

    time_ms = (np.arange(frame_len) / sample_rate) * 1000.0

    # Tính ACF cả 3 giai đoạn
    acf_raw = compute_acf(frame_raw, mode="normalized")
    acf_bp = compute_acf(frame_bp, mode="normalized")
    acf_final = compute_acf(frame_final, mode="normalized")
    tau_ms = (np.arange(len(acf_raw)) / sample_rate) * 1000.0

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.suptitle(
        "CHUỖI BIẾN ĐỔI TÍN HIỆU TOÀN DIỆN QUA CÁC BƯỚC TIỀN XỬ LÝ (PRE-PROCESSING PIPELINE)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Hàng 1: Giai đoạn 1 - Tín hiệu thô ban đầu
    axes[0, 0].plot(time_ms, frame_raw, color="#7f7f7f", linewidth=1.5, label="1. Tín hiệu gốc (Raw)")
    axes[0, 0].set_ylabel("Biên độ", fontsize=9.5, fontweight="bold")
    axes[0, 0].set_title("Giai đoạn 1: Khung tín hiệu thô (Dính DC, nhiễu cao tần & formant)", fontsize=10.5, fontweight="bold")
    axes[0, 0].set_xlim(0, max(time_ms))
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[0, 1].plot(tau_ms, acf_raw, color="#7f7f7f", linewidth=1.8, label="ACF Khung thô")
    axes[0, 1].set_ylabel("R(τ)", fontsize=9.5, fontweight="bold")
    axes[0, 1].set_title("ACF gốc: Đỉnh chu kỳ T0 bị cạnh tranh gay gắt bởi đỉnh phụ formant", fontsize=10.5, fontweight="bold")
    axes[0, 1].set_xlim(0, max(tau_ms))
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    # Hàng 2: Giai đoạn 2 - Sau Bandpass Filter
    axes[1, 0].plot(time_ms, frame_bp, color="#1f77b4", linewidth=1.8, label="2. Sau Bandpass (70-900Hz)")
    axes[1, 0].set_ylabel("Biên độ", fontsize=9.5, fontweight="bold")
    axes[1, 0].set_title("Giai đoạn 2: Sau lọc dải thông (Triệt tiêu DC drift, làm sạch dải thông 70-900 Hz)", fontsize=10.5, fontweight="bold")
    axes[1, 0].set_xlim(0, max(time_ms))
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[1, 1].plot(tau_ms, acf_bp, color="#1f77b4", linewidth=1.8, label="ACF Sau Bandpass")
    axes[1, 1].set_ylabel("R(τ)", fontsize=9.5, fontweight="bold")
    axes[1, 1].set_title("ACF sau lọc: Dạng tương quan trơn tru hơn, loại bỏ gai nhiễu cao tần", fontsize=10.5, fontweight="bold")
    axes[1, 1].set_xlim(0, max(tau_ms))
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    # Hàng 3: Giai đoạn 3 - Sau Center Clipping (Tối ưu hoàn chỉnh)
    axes[2, 0].plot(time_ms, frame_final, color="#2ca02c", linewidth=2.0, label="3. Sau Center Clipping (40%)")
    axes[2, 0].axhline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.5)
    axes[2, 0].set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Biên độ", fontsize=9.5, fontweight="bold")
    axes[2, 0].set_title("Giai đoạn 3: Sau Center Clipping (San phẳng phổ, giữ lại các xung nhọn GCI)", fontsize=10.5, fontweight="bold")
    axes[2, 0].set_xlim(0, max(time_ms))
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    axes[2, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[2, 1].plot(tau_ms, acf_final, color="#006837", linewidth=2.2, label="ACF Hoàn chỉnh (BP + Clip)")
    axes[2, 1].set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    axes[2, 1].set_ylabel("R(τ)", fontsize=9.5, fontweight="bold")
    axes[2, 1].set_title("ACF hoàn chỉnh: Đỉnh T0 nhô cao đơn cực tuyệt đối, đỉnh phụ bằng 0", fontsize=10.5, fontweight="bold")
    axes[2, 1].set_xlim(0, max(tau_ms))
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    axes[2, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã xuất biểu đồ Chuỗi tích hợp: {save_path}")


def main():
    audio_path = "TinHieuKiemThu/studio_F2.wav"
    if not os.path.exists(audio_path):
        audio_path = "TinHieuKiemThu/phone_M2.wav"

    sr, sig, dur = load_wav(audio_path)
    print(f"[*] Đang tải tín hiệu: {audio_path} (Fs = {sr} Hz, Thời lượng = {dur:.2f}s)")

    # Chọn vị trí khung nguyên âm hữu thanh rõ ràng (ví dụ khoảng 1.2s - 1.5s)
    t_frame = 1.35 if dur > 1.5 else dur / 2.0

    # 1. Sinh đồ thị Bandpass Filter
    plot_bandpass_demo(
        signal=sig,
        sample_rate=sr,
        t_start=t_frame,
        duration=0.08,
        save_path="outputs/figures/preprocessing_bandpass_demo.png",
    )

    # 2. Sinh đồ thị Center Clipping
    plot_center_clipping_demo(
        signal=sig,
        sample_rate=sr,
        t_start=t_frame,
        duration=0.035,
        clipping_ratio=0.40,
        save_path="outputs/figures/preprocessing_center_clipping_demo.png",
    )

    # 3. Sinh đồ thị Chuỗi tích hợp toàn diện
    plot_comprehensive_pipeline(
        signal=sig,
        sample_rate=sr,
        t_start=t_frame,
        duration=0.035,
        save_path="outputs/figures/preprocessing_pipeline_comprehensive.png",
    )


if __name__ == "__main__":
    main()
