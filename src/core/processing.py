"""Shared input, pitch-bound and segment-processing rules for all detectors."""
import numpy as np
from scipy.ndimage import median_filter


ENERGY_FLOOR = 1e-12


def compute_zcr(frame: np.ndarray) -> float:
    """Compute normalized Zero Crossing Rate for a 1D audio frame."""
    if len(frame) < 2:
        return 0.0
    signs = np.sign(frame)
    # Replace zeros with 1 to avoid zero-product artifacts
    signs[signs == 0] = 1.0
    crossings = np.sum(np.abs(signs[1:] - signs[:-1])) / 2.0
    return float(crossings / (len(frame) - 1))


def center_frame(frame: np.ndarray) -> np.ndarray:
    """Return a floating-point, DC-free frame without mutating the input."""
    values = np.asarray(frame, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Audio must be a finite one-dimensional signal")
    return values - np.mean(values) if values.size else values.copy()


def prepare_analysis_signal(signal, sample_rate, frame_duration_ms):
    """Validate detector input and remove the file's constant DC component."""
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(frame_duration_ms) or frame_duration_ms <= 0:
        raise ValueError("Frame duration must be positive")
    values = center_frame(signal)
    frame_len = max(1, int(round(frame_duration_ms * 1e-3 * sample_rate)))
    if len(values) < frame_len:
        raise ValueError(
            f"Audio too short for analysis: need at least {frame_len} samples "
            f"({frame_duration_ms:g} ms); got {len(values)}"
        )
    return values


def silence_energy_threshold(ste, ratio):
    """A positive floor also gates all-zero signals when the relative maximum is 0."""
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Silence energy ratio must be non-negative")
    maximum = float(np.max(ste)) if len(ste) else 0.0
    return max(ENERGY_FLOOR, ratio * maximum)


def validate_pitch_range(sample_rate, f0_min, f0_max):
    """Reject invalid configuration even when audio is empty or silent."""
    if not (np.isfinite(sample_rate) and np.isfinite(f0_min) and np.isfinite(f0_max)):
        raise ValueError("Pitch range and sample rate must be finite")
    if not (0 < f0_min < f0_max <= sample_rate / 2):
        raise ValueError("Pitch range must satisfy 0 < f0_min < f0_max <= Nyquist")


def pitch_lag_bounds(sample_rate, f0_min, f0_max, last_lag):
    """Integer search anchors inside the requested physical pitch range."""
    validate_pitch_range(sample_rate, f0_min, f0_max)
    return max(1, int(np.ceil(sample_rate / f0_max))), min(last_lag, int(np.floor(sample_rate / f0_min)))


def refined_f0(curve, lag, sample_rate, f0_min, f0_max):
    """Parabolic refinement constrained to the continuous physical lag interval."""
    delta = 0.0
    if 0 < lag < len(curve) - 1:
        left, middle, right = map(float, curve[lag - 1:lag + 2])
        denominator = left - 2.0 * middle + right
        if abs(denominator) > 1e-12:
            delta = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    refined = np.clip(lag + delta, sample_rate / f0_max, sample_rate / f0_min)
    return float(np.clip(sample_rate / refined, f0_min, f0_max))


def voiced_segments(mask):
    """Yield half-open intervals of contiguous True values."""
    transitions = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))


def smooth_voiced_segments(f0, labels=None, size=3):
    """Median-filter each voiced segment independently, preserving every gap."""
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Median filter size must be a positive integer")
    result = np.array(f0, dtype=np.float64, copy=True)
    mask = np.isfinite(result) & (result > 0)
    if labels is not None:
        mask &= np.asarray(labels) == "v"
    for start, end in voiced_segments(mask):
        if end - start >= size:
            result[start:end] = median_filter(result[start:end], size=size, mode="nearest")
    return result
