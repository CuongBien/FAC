"""Script 01: Minh họa và phân tích trên 2 khung tín hiệu (Voiced vs Unvoiced):
1. Khung Hữu Thanh (Voiced) -> Xác định F0 (Hz)
2. Khung Vô Thanh (Unvoiced) -> F0 không xác định
Hỗ trợ cả 3 thuật toán: ACF (miền thời gian), AMDF (miền thời gian), Cepstrum (miền quefrency).
"""
import argparse
import os
import sys
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

from src import config
from src.core.audio import load_wav, frame_signal, compute_ste
from src.core.lab_parser import parse_lab_file, get_frame_labels
from src.core.acf import compute_acf, find_f0_acf
from src.core.amdf import compute_amdf, find_f0_amdf
from src.core.cepstrum import compute_real_cepstrum, find_f0_cepstrum
from src.visualization.plotter import (
    plot_frame_acf_comparison,
    plot_frame_amdf_comparison,
    plot_frame_cepstrum_comparison,
)


def run_demo(
    wav_path: str = "TinHieuHuanLuyen/phone_F1.wav",
    lab_path: str = "TinHieuHuanLuyen/phone_F1.lab",
    method: str = "acf",
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    output_fig: str = None,
):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")

    method = method.lower()
    if output_fig is None:
        output_fig = f"outputs/figures/01_demo_{method}_frames.png"

    # 1. Đọc tín hiệu và nhãn
    sr, signal, duration = load_wav(wav_path)
    lab_data = parse_lab_file(lab_path) if os.path.exists(lab_path) else {"segments": [], "f0_mean": None, "f0_std": None}

    # 2. Phân khung
    frames, frame_times = frame_signal(
        signal, sr, frame_duration_ms=frame_duration_ms, hop_duration_ms=hop_duration_ms, window="rectangular"
    )

    # Tìm khung Voiced và Unvoiced đại diện
    v_idx = None
    uv_idx = None

    if lab_data["segments"]:
        frame_labels = get_frame_labels(frame_times, lab_data["segments"])
        voiced_indices = [i for i, lbl in enumerate(frame_labels) if lbl == "v"]
        unvoiced_indices = [i for i, lbl in enumerate(frame_labels) if lbl == "uv"]
        if voiced_indices:
            v_idx = voiced_indices[len(voiced_indices) // 3]
        if unvoiced_indices:
            uv_idx = unvoiced_indices[len(unvoiced_indices) // 2]

    # Nếu không có nhãn segments hoặc không đủ khung, tự động chọn dựa vào năng lượng STE & ZCR
    if v_idx is None or uv_idx is None:
        ste, _ = compute_ste(signal, sr, frame_duration_ms=frame_duration_ms, hop_duration_ms=hop_duration_ms)
        max_ste = np.max(ste) if len(ste) > 0 else 1.0
        # Voiced frame: khung năng lượng cao gần đỉnh âm vực
        v_idx = int(np.argmax(ste))

        # Unvoiced frame: khung trước khi bắt đầu nguyên âm hoặc phụ âm ma sát
        zcr = np.array([np.sum(np.abs(np.diff(np.sign(f)))) / (2.0 * len(f)) for f in frames])
        uv_candidates = [i for i in range(len(frames)) if 0.002 * max_ste < ste[i] < 0.15 * max_ste and zcr[i] > 0.10]
        if uv_candidates:
            # Chọn khung có zcr cao nhất trong các khung ứng viên
            uv_idx = int(uv_candidates[np.argmax([zcr[i] for i in uv_candidates])])
        else:
            uv_idx = 20

    frame_v = frames[v_idx]
    time_v = frame_times[v_idx]
    frame_uv = frames[uv_idx]
    time_uv = frame_times[uv_idx]

    print(f"File: {wav_path} (Fs={sr}Hz, Duration={duration:.3f}s)")
    if lab_data.get("f0_mean") is not None:
        print(f"Ground Truth: F0mean={lab_data['f0_mean']}Hz, F0std={lab_data['f0_std']}Hz")
    print(f"Params: Method={method.upper()}, FrameLen={frame_duration_ms}ms, HopLen={hop_duration_ms}ms, F0Range=[{f0_min:.0f}, {f0_max:.0f}]Hz")

    # 3. Tính toán theo phương pháp đã chọn
    if method == "amdf":
        amdf_v = compute_amdf(frame_v, mode="normalized")
        amdf_uv = compute_amdf(frame_uv, mode="normalized")

        f0_v, dip_v, lag_v = find_f0_amdf(frame_v, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
        f0_uv, dip_uv, lag_uv = find_f0_amdf(frame_uv, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")

        t0_v_ms = (lag_v / sr) * 1000.0
        t0_uv_ms = (lag_uv / sr) * 1000.0

        print(f"Voiced   [Frame #{v_idx:3d}, t={time_v:.3f}s]: lag={lag_v:3d}, T0={t0_v_ms:5.2f}ms, F0={f0_v:6.2f}Hz, Dip={dip_v:.3f}")
        print(f"Unvoiced [Frame #{uv_idx:3d}, t={time_uv:.3f}s]: lag={lag_uv:3d}, T0={t0_uv_ms:5.2f}ms, F0=Undefined, Dip={dip_uv:.3f}")

        plot_frame_amdf_comparison(
            voiced_frame=frame_v,
            voiced_amdf=amdf_v,
            voiced_f0=f0_v,
            voiced_lag=lag_v,
            voiced_dip=dip_v,
            unvoiced_frame=frame_uv,
            unvoiced_amdf=amdf_uv,
            unvoiced_lag=lag_uv,
            unvoiced_dip=dip_uv,
            sample_rate=sr,
            f0_min=f0_min,
            f0_max=f0_max,
            threshold=config.baseline_threshold(method),
            save_path=output_fig,
            title=f"Phân tích AMDF Khung Hữu Thanh vs Vô Thanh ({os.path.basename(wav_path)})",
        )
    elif method == "cepstrum":
        cep_v, _ = compute_real_cepstrum(frame_v, sr, window="hamming", n_fft=2048)
        cep_uv, _ = compute_real_cepstrum(frame_uv, sr, window="hamming", n_fft=2048)

        f0_v, peak_v, lag_v = find_f0_cepstrum(frame_v, sr, f0_min=f0_min, f0_max=f0_max, window="hamming", n_fft=2048)
        f0_uv, peak_uv, lag_uv = find_f0_cepstrum(frame_uv, sr, f0_min=f0_min, f0_max=f0_max, window="hamming", n_fft=2048)

        q0_v_ms = (lag_v / sr) * 1000.0
        q0_uv_ms = (lag_uv / sr) * 1000.0

        print(f"Voiced   [Frame #{v_idx:3d}, t={time_v:.3f}s]: quefrency={lag_v:3d}, q0={q0_v_ms:5.2f}ms, F0={f0_v:6.2f}Hz, Peak={peak_v:.3f}")
        print(f"Unvoiced [Frame #{uv_idx:3d}, t={time_uv:.3f}s]: quefrency={lag_uv:3d}, q0={q0_uv_ms:5.2f}ms, F0=Undefined, Peak={peak_uv:.3f}")

        plot_frame_cepstrum_comparison(
            voiced_frame=frame_v,
            voiced_cep=cep_v,
            voiced_f0=f0_v,
            voiced_lag=lag_v,
            voiced_peak=peak_v,
            unvoiced_frame=frame_uv,
            unvoiced_cep=cep_uv,
            unvoiced_lag=lag_uv,
            unvoiced_peak=peak_uv,
            sample_rate=sr,
            f0_min=f0_min,
            f0_max=f0_max,
            threshold=config.baseline_threshold(method),
            save_path=output_fig,
            title=f"Phân tích Cepstrum Khung Hữu Thanh vs Vô Thanh ({os.path.basename(wav_path)})",
        )
    else:  # acf
        acf_v = compute_acf(frame_v, mode="normalized")
        acf_uv = compute_acf(frame_uv, mode="normalized")

        f0_v, peak_v, lag_v = find_f0_acf(frame_v, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
        f0_uv, peak_uv, lag_uv = find_f0_acf(frame_uv, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")

        t0_v_ms = (lag_v / sr) * 1000.0
        t0_uv_ms = (lag_uv / sr) * 1000.0

        print(f"Voiced   [Frame #{v_idx:3d}, t={time_v:.3f}s]: lag={lag_v:3d}, T0={t0_v_ms:5.2f}ms, F0={f0_v:6.2f}Hz, Peak={peak_v:.3f}")
        print(f"Unvoiced [Frame #{uv_idx:3d}, t={time_uv:.3f}s]: lag={lag_uv:3d}, T0={t0_uv_ms:5.2f}ms, F0=Undefined, Peak={peak_uv:.3f}")

        plot_frame_acf_comparison(
            voiced_frame=frame_v,
            voiced_acf=acf_v,
            voiced_f0=f0_v,
            voiced_lag=lag_v,
            voiced_peak=peak_v,
            unvoiced_frame=frame_uv,
            unvoiced_acf=acf_uv,
            unvoiced_lag=lag_uv,
            unvoiced_peak=peak_uv,
            sample_rate=sr,
            f0_min=f0_min,
            f0_max=f0_max,
            save_path=output_fig,
            title=f"Phân tích ACF Khung Hữu Thanh vs Vô Thanh ({os.path.basename(wav_path)})",
        )

    print(f"Saved: {output_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo ACF/AMDF/Cepstrum on Voiced and Unvoiced frames.")
    parser.add_argument("--wav", type=str, default="TinHieuHuanLuyen/phone_F1.wav", help="Đường dẫn file .wav")
    parser.add_argument("--lab", type=str, default="TinHieuHuanLuyen/phone_F1.lab", help="Đường dẫn file .lab")
    parser.add_argument("--method", type=str, default="acf", choices=["acf", "amdf", "cepstrum"], help="Thuật toán (acf, amdf, hoặc cepstrum)")
    parser.add_argument("--frame_len", type=float, default=25.0, help="Độ dài khung (ms)")
    parser.add_argument("--hop_len", type=float, default=10.0, help="Độ dịch khung (ms)")
    parser.add_argument("--f0_min", type=float, default=70.0, help="F0 tối thiểu (Hz)")
    parser.add_argument("--f0_max", type=float, default=400.0, help="F0 tối đa (Hz)")
    parser.add_argument("--output", type=str, default=None, help="File ảnh xuất ra (mặc định theo method)")

    args = parser.parse_args()
    run_demo(
        wav_path=args.wav,
        lab_path=args.lab,
        method=args.method,
        frame_duration_ms=args.frame_len,
        hop_duration_ms=args.hop_len,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        output_fig=args.output,
    )
