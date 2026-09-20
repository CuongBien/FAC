"""Statistical distribution extraction and optimal threshold calculation (Gaussian threshold / histogram)."""
import glob
import os
from typing import Dict, List, Tuple
import numpy as np

from src.core.audio import load_wav, frame_signal
from src.core.lab_parser import parse_lab_file, get_frame_labels
from src.core.acf import find_f0_acf


def extract_training_distributions(
    training_dir: str = "TinHieuHuanLuyen",
    method: str = "acf",
    frame_duration_ms: float = 30.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
) -> Dict:
    """Analyze all voiced and unvoiced frames in the training dataset using *.lab timestamps.

    Args:
        training_dir: Directory containing training WAV and LAB files.
        method: Pitch estimation method ('acf').
        frame_duration_ms: Frame length in milliseconds.
        hop_duration_ms: Frame shift in milliseconds.
        f0_min: Minimum pitch search frequency in Hz.
        f0_max: Maximum pitch search frequency in Hz.

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

        sr, signal, _ = load_wav(wav_path)
        lab_data = parse_lab_file(lab_path)

        frames, frame_times = frame_signal(
            signal, sr, frame_duration_ms=frame_duration_ms, hop_duration_ms=hop_duration_ms, window="rectangular"
        )
        labels = get_frame_labels(frame_times, lab_data["segments"])

        for frame, label in zip(frames, labels):
            if label == "v":
                if method == "acf":
                    _, peak_val, _ = find_f0_acf(frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
                    all_values_v.append(peak_val)
            elif label == "uv":
                if method == "acf":
                    _, peak_val, _ = find_f0_acf(frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized")
                    all_values_u.append(peak_val)

        processed_files.append(os.path.basename(wav_path))

    values_v = np.array(all_values_v, dtype=np.float64)
    values_u = np.array(all_values_u, dtype=np.float64)

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
    var_v = std_v ** 2
    var_u = std_u ** 2

    if abs(var_v - var_u) < 1e-6:
        # Equal variances: midpoint
        return float((mean_v + mean_u) / 2.0)

    # Coefficients of quadratic equation:
    # (T - mean_v)^2 / var_v - (T - mean_u)^2 / var_u + 2 * ln(std_v / std_u) = 0
    a = 1.0 / var_v - 1.0 / var_u
    b = -2.0 * (mean_v / var_v - mean_u / var_u)
    c = (mean_v ** 2) / var_v - (mean_u ** 2) / var_u + 2.0 * np.log(std_v / std_u)

    discriminant = b ** 2 - 4.0 * a * c
    if discriminant < 0:
        # Fallback to weighted mean
        return float((mean_v * std_u + mean_u * std_v) / (std_v + std_u))

    root1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
    root2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

    min_mean = min(mean_v, mean_u)
    max_mean = max(mean_v, mean_u)

    # Pick root that lies between mean_u and mean_v
    in_range_roots = [r for r in (root1, root2) if min_mean <= r <= max_mean]
    if in_range_roots:
        return float(in_range_roots[0])

    # If neither or both outside, pick the one closest to midpoint
    midpoint = (mean_v + mean_u) / 2.0
    return float(min((root1, root2), key=lambda r: abs(r - midpoint)))
