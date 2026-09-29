#!/usr/bin/env python3
"""Script: Trực quan hóa chi tiết tác động của Tiền xử lý lên thuật toán AMDF:
1. Bandpass Filter (Butterworth bậc 2, 70 - 900 Hz, Zero-Phase filtfilt) lên AMDF
2. Center Clipping (Cắt gọt trung tâm - Sondhi 1968, ratio = 0.40) lên AMDF
3. Chuỗi biến đổi toàn diện 3 giai đoạn đối với AMDF
4. So sánh trực quan cơ chế đối ngẫu giữa ACF (bắt đỉnh Peak) và AMDF (bắt đáy Dip/Valley)

Xuất các đồ thị minh họa:
- outputs/figures/preprocessing_bandpass_amdf_demo.png
- outputs/figures/preprocessing_center_clipping_amdf_demo.png
- outputs/figures/preprocessing_pipeline_amdf_comprehensive.png
- outputs/figures/amdf_vs_acf_comparison.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
from src.core.amdf import compute_amdf
from src.plugins.pre_processing.bandpass_filter import BandpassFilterPlugin
from src.plugins.pre_processing.center_clipping import CenterClippingPlugin

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_bandpass_amdf_demo(
    signal: np.ndarray,
    sample_rate: int,
    t_start: float = 1.35,
    duration: float = 0.08,
    save_path: str = "outputs/figures/preprocessing_bandpass_amdf_demo.png",
):
    """Minh họa tác động của Bandpass Filter lên dạng sóng phổ và hàm hiệu độ lớn trung bình AMDF."""
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0, order=2)
    filtered_signal = bp.pre_process_signal(signal, sample_rate)

    start_idx = int(t_start * sample_rate)
    end_idx = start_idx + int(duration * sample_rate)
    time_ms = (np.arange(end_idx - start_idx) / sample_rate) * 1000.0

    raw_segment = signal[start_idx:end_idx]
    filtered_segment = filtered_signal[start_idx:end_idx]

    # Tính phổ FFT
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    fft_raw = np.abs(np.fft.rfft(raw_segment * np.hamming(len(raw_segment)), n=n_fft))
    fft_filt = np.abs(np.fft.rfft(filtered_segment * np.hamming(len(filtered_segment)), n=n_fft))

    fft_raw_db = 20 * np.log10(fft_raw / (np.max(fft_raw) + 1e-12) + 1e-12)
    fft_filt_db = 20 * np.log10(fft_filt / (np.max(fft_filt) + 1e-12) + 1e-12)

    # Tính AMDF của khung 35ms
    frame_len = int(0.035 * sample_rate)
    amdf_raw = compute_amdf(raw_segment[:frame_len], mode="normalized")
    amdf_filt = compute_amdf(filtered_segment[:frame_len], mode="normalized")
    tau_ms = (np.arange(len(amdf_raw)) / sample_rate) * 1000.0

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.22)
    fig.suptitle(
        "TÁC ĐỘNG CỦA BỘ LỌC DẢI THÔNG BANDPASS FILTER (70 - 900 Hz, ZERO-PHASE) LÊN THUẬT TOÁN AMDF",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # 1. Miền thời gian
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(time_ms, raw_segment, color="#999999", linewidth=1.2, alpha=0.8, label="Tín hiệu thô (Nhiễu hài cao tần & trôi DC)")
    ax0.plot(time_ms, filtered_segment, color="#1f77b4", linewidth=1.8, label="Sau lọc Bandpass (70 - 900 Hz, filtfilt ϕ = 0)")
    ax0.set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    ax0.set_ylabel("Biên độ", fontsize=10, fontweight="bold")
    ax0.set_title("1. Dạng sóng thời gian: Khử trôi DC, bảo toàn 100% pha (không lệch vị trí các đáy AMDF)", fontsize=11, fontweight="bold")
    ax0.set_xlim(0, max(time_ms))
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right", frameon=True, fontsize=10)

    # 2. Miền tần số FFT
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
    ax1.set_title("2. Miền tần số: Triệt tiêu trôi dạt DC (< 70 Hz) & dập tắt nhiễu cao tần (> 900 Hz)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    # 3. Hàm hiệu độ lớn AMDF
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(tau_ms, amdf_raw, color="#999999", linewidth=1.5, linestyle="--", label="AMDF Tín hiệu gốc")
    ax2.plot(tau_ms, amdf_filt, color="#d95f02", linewidth=2.0, label="AMDF Sau lọc thông dải")
    ax2.set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Hàm AMDF chuẩn hóa D(τ)", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(tau_ms))
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_title("3. Hàm AMDF: Đáy thung lũng mịn màng, loại bỏ các gợn răng cưa nhiễu vi mô", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã xuất biểu đồ Bandpass Filter AMDF: {save_path}")


def plot_center_clipping_amdf_demo(
    signal: np.ndarray,
    sample_rate: int,
    t_start: float = 1.35,
    duration: float = 0.035,
    clipping_ratio: float = 0.40,
    save_path: str = "outputs/figures/preprocessing_center_clipping_amdf_demo.png",
):
    """Minh họa tác động của Center Clipping lên việc xóa các đáy phụ Formant trong AMDF."""
    start_idx = int(t_start * sample_rate)
    frame_len = int(duration * sample_rate)
    frame = signal[start_idx : start_idx + frame_len].copy()
    time_ms = (np.arange(frame_len) / sample_rate) * 1000.0

    # Áp dụng lọc Bandpass trước để có tín hiệu chuẩn sạch
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0, order=2)
    sig_bp = bp.pre_process_signal(signal, sample_rate)
    frame_clean = sig_bp[start_idx : start_idx + frame_len].copy()

    clip_plugin = CenterClippingPlugin(clipping_ratio=clipping_ratio, mode="standard")
    clipped_frame = clip_plugin.pre_process_frame(frame_clean, sample_rate)

    max_val = np.max(np.abs(frame_clean))
    c_l = clipping_ratio * max_val

    # Tính hàm AMDF
    amdf_raw = compute_amdf(frame, mode="normalized")
    amdf_clipped = compute_amdf(clipped_frame, mode="normalized")
    tau_ms = (np.arange(len(amdf_raw)) / sample_rate) * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5))
    fig.suptitle(
        f"MINH HỌA TÁC ĐỘNG CỦA CẮT GỌT TRUNG TÂM CENTER CLIPPING (RATIO = {clipping_ratio:.2f}) LÊN AMDF",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # 1. Khung dạng sóng với ngưỡng gọt
    axes[0, 0].plot(time_ms, frame_clean, color="#1f77b4", linewidth=1.8, label="Dạng sóng hữu thanh (Voiced Frame)")
    axes[0, 0].axhline(c_l, color="#d62728", linestyle="--", linewidth=1.5, label=f"Ngưỡng trên +C_L = +{clipping_ratio*100:.0f}% max")
    axes[0, 0].axhline(-c_l, color="#d62728", linestyle="--", linewidth=1.5, label=f"Ngưỡng dưới -C_L = -{clipping_ratio*100:.0f}% max")
    axes[0, 0].axhspan(-c_l, c_l, color="#d62728", alpha=0.12, label="Vùng dao động formant bị gọt về 0")
    axes[0, 0].set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Biên độ", fontsize=10, fontweight="bold")
    axes[0, 0].set_title("1. Khung tín hiệu: Chứa nhiều gợn sóng dao động phụ do Formant F1 ở dải giữa", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlim(0, max(time_ms))
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    # 2. Dạng sóng sau cắt gọt
    axes[0, 1].plot(time_ms, clipped_frame, color="#2ca02c", linewidth=2.0, label="Sau Center Clipping (y[n])")
    axes[0, 1].axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
    axes[0, 1].set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Biên độ sau cắt", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("2. Sau Clipping: Gọt sạch dải giữa về 0, chỉ còn các xung cực đại đóng thanh môn (GCI)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlim(0, max(time_ms))
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    # 3. AMDF trước khi cắt gọt
    axes[1, 0].plot(tau_ms, amdf_raw, color="#d62728", linewidth=1.8, label="AMDF Tín hiệu thô")
    axes[1, 0].set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Hàm AMDF chuẩn hóa D(τ)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylim(-0.05, 1.1)
    axes[1, 0].set_title("3. AMDF thô: Xuất hiện nhiều đáy phụ sâu do Formant F1 cạnh tranh với đáy chu kỳ T0", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlim(0, max(tau_ms))
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right", frameon=True, fontsize=9)

    # 4. AMDF sau Center Clipping
    axes[1, 1].plot(tau_ms, amdf_clipped, color="#006837", linewidth=2.0, label="AMDF Sau Center Clipping")
    axes[1, 1].set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Hàm AMDF chuẩn hóa D(τ)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylim(-0.05, 1.1)
    axes[1, 1].set_title("4. AMDF sau Clipping: Các đáy phụ formant bị đẩy lên cao, đáy chu kỳ T0 sâu & cô lập tuyệt đối", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlim(0, max(tau_ms))
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã xuất biểu đồ Center Clipping AMDF: {save_path}")


def plot_comprehensive_pipeline_amdf(
    signal: np.ndarray,
    sample_rate: int,
    t_start: float = 1.35,
    duration: float = 0.035,
    save_path: str = "outputs/figures/preprocessing_pipeline_amdf_comprehensive.png",
):
    """Minh họa toàn diện chuỗi tích hợp 3 giai đoạn: Gốc -> Bandpass Filter -> Center Clipping đối với AMDF."""
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

    # Tính AMDF cả 3 giai đoạn
    amdf_raw = compute_amdf(frame_raw, mode="normalized")
    amdf_bp = compute_amdf(frame_bp, mode="normalized")
    amdf_final = compute_amdf(frame_final, mode="normalized")
    tau_ms = (np.arange(len(amdf_raw)) / sample_rate) * 1000.0

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.suptitle(
        "CHUỖI BIẾN ĐỔI TIỀN XỬ LÝ TOÀN DIỆN CHO THUẬT TOÁN AMDF (RAW -> BANDPASS -> CENTER CLIPPING)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Hàng 1: Giai đoạn 1 - Khung thô
    axes[0, 0].plot(time_ms, frame_raw, color="#7f7f7f", linewidth=1.5, label="1. Tín hiệu thô (Raw)")
    axes[0, 0].set_ylabel("Biên độ", fontsize=9.5, fontweight="bold")
    axes[0, 0].set_title("Giai đoạn 1: Khung tín hiệu thô (Nhiễu cao tần, DC offset, dao động formant F1)", fontsize=10.5, fontweight="bold")
    axes[0, 0].set_xlim(0, max(time_ms))
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[0, 1].plot(tau_ms, amdf_raw, color="#7f7f7f", linewidth=1.8, label="AMDF Khung thô")
    axes[0, 1].set_ylabel("D(τ)", fontsize=9.5, fontweight="bold")
    axes[0, 1].set_ylim(-0.05, 1.1)
    axes[0, 1].set_title("AMDF gốc: Đáy chu kỳ T0 bị che lấp & cạnh tranh bởi các thung lũng phụ Formant", fontsize=10.5, fontweight="bold")
    axes[0, 1].set_xlim(0, max(tau_ms))
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    # Hàng 2: Giai đoạn 2 - Sau Bandpass Filter
    axes[1, 0].plot(time_ms, frame_bp, color="#1f77b4", linewidth=1.8, label="2. Sau Bandpass (70-900Hz)")
    axes[1, 0].set_ylabel("Biên độ", fontsize=9.5, fontweight="bold")
    axes[1, 0].set_title("Giai đoạn 2: Sau lọc thông dải (Triệt tiêu DC trôi dạt, mượt mà hóa dải thông)", fontsize=10.5, fontweight="bold")
    axes[1, 0].set_xlim(0, max(time_ms))
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[1, 1].plot(tau_ms, amdf_bp, color="#1f77b4", linewidth=1.8, label="AMDF Sau Bandpass")
    axes[1, 1].set_ylabel("D(τ)", fontsize=9.5, fontweight="bold")
    axes[1, 1].set_ylim(-0.05, 1.1)
    axes[1, 1].set_title("AMDF sau lọc: Đáy AMDF trơn tru, loại bỏ hoàn toàn các gai nhọn nhiễu cao tần", fontsize=10.5, fontweight="bold")
    axes[1, 1].set_xlim(0, max(tau_ms))
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    # Hàng 3: Giai đoạn 3 - Sau Center Clipping (Hoàn chỉnh)
    axes[2, 0].plot(time_ms, frame_final, color="#2ca02c", linewidth=2.0, label="3. Sau Center Clipping (40%)")
    axes[2, 0].axhline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.5)
    axes[2, 0].set_xlabel("Thời gian (ms)", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Biên độ", fontsize=9.5, fontweight="bold")
    axes[2, 0].set_title("Giai đoạn 3: Sau Center Clipping (Chỉ còn các xung nhọn GCI, phổ san phẳng)", fontsize=10.5, fontweight="bold")
    axes[2, 0].set_xlim(0, max(time_ms))
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    axes[2, 0].legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[2, 1].plot(tau_ms, amdf_final, color="#006837", linewidth=2.2, label="AMDF Hoàn chỉnh (BP + Clip)")
    axes[2, 1].set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    axes[2, 1].set_ylabel("D(τ)", fontsize=9.5, fontweight="bold")
    axes[2, 1].set_ylim(-0.05, 1.1)
    axes[2, 1].set_title("AMDF hoàn chỉnh: Đáy chu kỳ T0 rơi cực sâu và đơn độc, thuật toán bắt cực tiểu 100% chuẩn xác", fontsize=10.5, fontweight="bold")
    axes[2, 1].set_xlim(0, max(tau_ms))
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    axes[2, 1].legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã xuất biểu đồ Chuỗi tích hợp AMDF: {save_path}")


def plot_amdf_vs_acf_comparison(
    signal: np.ndarray,
    sample_rate: int,
    t_start: float = 1.35,
    duration: float = 0.035,
    save_path: str = "outputs/figures/amdf_vs_acf_comparison.png",
):
    """So sánh trực tiếp cơ chế đối ngẫu giữa ACF (Tự tương quan) và AMDF (Hiệu độ lớn) trên cùng tín hiệu."""
    start_idx = int(t_start * sample_rate)
    frame_len = int(duration * sample_rate)

    # Xử lý qua Bandpass + Center Clipping
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0, order=2)
    sig_bp = bp.pre_process_signal(signal, sample_rate)
    clip = CenterClippingPlugin(clipping_ratio=0.40, mode="standard")
    frame_clean = clip.pre_process_frame(sig_bp[start_idx : start_idx + frame_len], sample_rate)

    time_ms = (np.arange(frame_len) / sample_rate) * 1000.0
    acf_val = compute_acf(frame_clean, mode="normalized")
    amdf_val = compute_amdf(frame_clean, mode="normalized")
    tau_ms = (np.arange(len(acf_val)) / sample_rate) * 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "SO SÁNH CƠ CHẾ ĐỐI NGẪU: ACF (BẮT CỰC ĐẠI / PEAK) VS AMDF (BẮT CỰC TIỂU / VALLEY)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # 1. Dạng sóng đã tiền xử lý
    axes[0].plot(time_ms, frame_clean, color="#2ca02c", linewidth=1.8, label="Khung tín hiệu sau Tiền xử lý (Bandpass + Center Clipping)")
    axes[0].axhline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.5)
    axes[0].set_ylabel("Biên độ", fontsize=10, fontweight="bold")
    axes[0].set_title("1. Dạng sóng sau tiền xử lý: Chuỗi xung nhọn GCI phân tách rõ ràng", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0, max(tau_ms))
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9.5)

    # 2. ACF
    axes[1].plot(tau_ms, acf_val, color="#08519c", linewidth=2.0, label="Hàm tự tương quan ACF R(τ)")
    axes[1].set_ylabel("R(τ)", fontsize=10, fontweight="bold")
    axes[1].set_title("2. Cơ chế ACF: Tìm CỰC ĐẠI (PEAK) tại lag τ = T0 (Điểm có độ tương đồng tích chập lớn nhất)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9.5)

    # 3. AMDF
    axes[2].plot(tau_ms, amdf_val, color="#d95f02", linewidth=2.0, label="Hàm hiệu độ lớn AMDF D(τ)")
    axes[2].set_xlabel("Độ trễ lag τ (ms)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("D(τ)", fontsize=10, fontweight="bold")
    axes[2].set_title("3. Cơ chế AMDF: Tìm CỰC TIỂU (VALLEY / DIP) tại lag τ = T0 (Điểm có sai khác biên độ nhỏ nhất ≈ 0)", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã xuất biểu đồ so sánh đối ngẫu ACF vs AMDF: {save_path}")


def main():
    audio_path = "TinHieuKiemThu/studio_F2.wav"
    if not os.path.exists(audio_path):
        audio_path = "TinHieuKiemThu/phone_M2.wav"

    sr, sig, dur = load_wav(audio_path)
    print(f"[*] Đang tải tín hiệu: {audio_path} (Fs = {sr} Hz, Thời lượng = {dur:.2f}s)")

    t_frame = 1.35 if dur > 1.5 else dur / 2.0

    # 1. Sinh đồ thị Bandpass Filter cho AMDF
    plot_bandpass_amdf_demo(
        signal=sig,
        sample_rate=sr,
        t_start=t_frame,
        duration=0.08,
        save_path="outputs/figures/preprocessing_bandpass_amdf_demo.png",
    )

    # 2. Sinh đồ thị Center Clipping cho AMDF
    plot_center_clipping_amdf_demo(
        signal=sig,
        sample_rate=sr,
        t_start=t_frame,
        duration=0.035,
        clipping_ratio=0.40,
        save_path="outputs/figures/preprocessing_center_clipping_amdf_demo.png",
    )

    # 3. Sinh đồ thị Chuỗi tích hợp toàn diện cho AMDF
    plot_comprehensive_pipeline_amdf(
        signal=sig,
        sample_rate=sr,
        t_start=t_frame,
        duration=0.035,
        save_path="outputs/figures/preprocessing_pipeline_amdf_comprehensive.png",
    )

    # 4. Sinh đồ thị so sánh đối ngẫu trực tiếp ACF vs AMDF
    plot_amdf_vs_acf_comparison(
        signal=sig,
        sample_rate=sr,
        t_start=t_frame,
        duration=0.035,
        save_path="outputs/figures/amdf_vs_acf_comparison.png",
    )


if __name__ == "__main__":
    main()
