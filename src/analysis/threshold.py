"""Statistical distribution extraction and optimal threshold calculation (Gaussian threshold / histogram)."""
import glob
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.core.audio import load_wav, frame_signal
from src.core.lab_parser import parse_lab_file, get_frame_labels
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.core.processing import center_frame, prepare_analysis_signal
from src.analysis.advanced_unsupervised import _gaussian_intersection


def extract_training_distributions(
    training_dir: str = "TinHieuHuanLuyen",
    method: str = "acf",
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    plugins: Optional[List[Any]] = None,
) -> Dict:
    """Analyze all voiced and unvoiced frames in the training dataset using *.lab timestamps.

    Args:
        training_dir: Directory containing training WAV and LAB files.
        method: Pitch estimation method ('acf').
        frame_duration_ms: Frame length in milliseconds.
        hop_duration_ms: Frame shift in milliseconds.
        f0_min: Minimum pitch search frequency in Hz.
        f0_max: Maximum pitch search frequency in Hz.
        plugins: Optional list of plugins (e.g. BandpassFilterPlugin) applied before framing.

    Returns:
        dict: {
            'mean_v': float,
            'std_v': float,
            'mean_u': float,
            'std_u': float,
            'values_v': np.ndarray,
            'values_u': np.ndarray,
            'num_voiced_frames': int,
            'num_unvoiced_frames': int,
            'files_processed': List[str]
        }
    """
    method = method.lower()
    if method not in ("acf", "amdf", "cepstrum"):
        raise ValueError(f"Unknown pitch method: {method}")
    wav_files = sorted(glob.glob(os.path.join(training_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in directory: {training_dir}")

    all_values_v = []
    all_values_u = []
    processed_files = []

    for wav_path in wav_files:
        lab_path = wav_path.rsplit(".", 1)[0] + ".lab"
        if not os.path.exists(lab_path):
            continue

        lab_data = parse_lab_file(lab_path)
        if not lab_data["segments"]:
            continue
        sr, signal, _ = load_wav(wav_path)
        signal = prepare_analysis_signal(signal, sr, frame_duration_ms)

        if plugins:
            for p in plugins:
                if hasattr(p, "reset"):
                    p.reset()
                if hasattr(p, "pre_process_signal"):
                    signal = p.pre_process_signal(signal, sr)

        frames, frame_times = frame_signal(
            signal, sr, frame_duration_ms=frame_duration_ms, hop_duration_ms=hop_duration_ms, window="rectangular"
        )
        labels = get_frame_labels(frame_times, lab_data["segments"])

        for frame, label in zip(frames, labels):
            frame = center_frame(frame)
            if plugins:
                for p in plugins:
                    if hasattr(p, "pre_process_frame"):
                        frame = p.pre_process_frame(frame, sr)

            if label == "v":
                if method.lower() == "acf":
                    _, peak_val, _ = find_f0_acf(frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
                    all_values_v.append(peak_val)
                elif method.lower() == "amdf":
                    _, dip_val, _ = find_f0_amdf(frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
                    all_values_v.append(dip_val)
                elif method.lower() == "cepstrum":
                    _, peak_val, _ = find_f0_cepstrum(frame, sr, f0_min=f0_min, f0_max=f0_max, window="hamming", n_fft=2048, mode="peak")
                    all_values_v.append(peak_val)
            elif label == "uv":
                if method.lower() == "acf":
                    _, peak_val, _ = find_f0_acf(frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
                    all_values_u.append(peak_val)
                elif method.lower() == "amdf":
                    _, dip_val, _ = find_f0_amdf(frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
                    all_values_u.append(dip_val)
                elif method.lower() == "cepstrum":
                    _, peak_val, _ = find_f0_cepstrum(frame, sr, f0_min=f0_min, f0_max=f0_max, window="hamming", n_fft=2048, mode="peak")
                    all_values_u.append(peak_val)

        processed_files.append(os.path.basename(wav_path))

    values_v = np.array(all_values_v, dtype=np.float64)
    values_u = np.array(all_values_u, dtype=np.float64)

    if len(values_v) < 2 or len(values_u) < 2:
        raise ValueError(
            "Gaussian training needs at least 2 labeled frames in each class "
            f"(V={len(values_v)}, UV={len(values_u)}). LAB must contain timed v/uv "
            "segments; F0mean/F0std/F0num summaries cannot train the distributions."
        )

    mean_v = float(np.mean(values_v)) if len(values_v) > 0 else 0.0
    std_v = float(np.std(values_v)) if len(values_v) > 0 else 0.0
    mean_u = float(np.mean(values_u)) if len(values_u) > 0 else 0.0
    std_u = float(np.std(values_u)) if len(values_u) > 0 else 0.0

    return {
        "mean_v": mean_v,
        "std_v": std_v,
        "mean_u": mean_u,
        "std_u": std_u,
        "values_v": values_v,
        "values_u": values_u,
        "num_voiced_frames": len(values_v),
        "num_unvoiced_frames": len(values_u),
        "files_processed": processed_files,
    }


def find_gaussian_threshold(mean_v: float, std_v: float, mean_u: float, std_u: float) -> float:
    """Find the optimal separation threshold T assuming normal distributions N(mean_v, std_v^2) and N(mean_u, std_u^2).

    Solves for the intersection point where P(x|V) = P(x|U).
    Quadratic equation: A*T^2 + B*T + C = 0.
    """
    parameters = np.array([mean_v, std_v, mean_u, std_u], dtype=float)
    if not np.all(np.isfinite(parameters)) or std_v <= 0 or std_u <= 0:
        raise ValueError("Gaussian fitting requires finite means and positive standard deviations in both labeled classes")
    threshold, status = _gaussian_intersection(mean_v, std_v, mean_u, std_u)
    if status not in ("between_means", "outside_means"):
        raise ValueError(f"No identifiable Gaussian separation threshold: {status}")
    return threshold
