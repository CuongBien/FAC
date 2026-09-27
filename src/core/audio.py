from typing import Optional
import warnings
import numpy as np
from scipy.io import wavfile

# Suppress metadata chunk warnings in WAV headers
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


def load_wav(filepath: str):
    """Load a WAV file and normalize amplitude to [-1, 1].

    Args:
        filepath: Path to .wav file.

    Returns:
        tuple: (sample_rate: int, signal: np.ndarray (float64 in [-1, 1]), duration: float)
    """
    sr, data = wavfile.read(filepath)

    # Convert stereo to mono if necessary
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Convert integer types to float [-1.0, 1.0]
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        signal = data.astype(np.float64) / max_val
    else:
        signal = data.astype(np.float64)
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal = signal / max_abs

    duration = len(signal) / sr
    return sr, signal, duration


def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    window: str = "rectangular",
):
    """Split continuous signal into overlapping frames.

    Args:
        signal: 1D audio samples.
        sample_rate: Audio sampling frequency in Hz.
        frame_duration_ms: Frame length in milliseconds (e.g., 25.0 ms).
        hop_duration_ms: Frame shift / hop size in milliseconds (e.g., 10.0 ms).
        window: Window type: 'rectangular', 'hamming', or 'hanning'.

    Returns:
        tuple: (frames: np.ndarray of shape (num_frames, frame_len),
                frame_times: np.ndarray representing the center time of each frame in seconds)
    """
    frame_len = int(round(frame_duration_ms * 1e-3 * sample_rate))
    hop_len = int(round(hop_duration_ms * 1e-3 * sample_rate))

    total_samples = len(signal)
    if total_samples < frame_len:
        num_frames = 1
        pad_len = frame_len - total_samples
        padded_signal = np.pad(signal, (0, pad_len), mode="constant")
    else:
        num_frames = 1 + int(np.floor((total_samples - frame_len) / hop_len))
        padded_signal = signal

    # Build window
    if window.lower() == "hamming":
        win = np.hamming(frame_len)
    elif window.lower() == "hanning":
        win = np.hanning(frame_len)
    else:
        win = np.ones(frame_len)

    frames = np.zeros((num_frames, frame_len), dtype=np.float64)
    frame_times = np.zeros(num_frames, dtype=np.float64)

    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frames[i] = padded_signal[start:end] * win
        frame_times[i] = (start + frame_len / 2.0) / sample_rate

    return frames, frame_times


def compute_ste(
    signal: np.ndarray,
    sample_rate: int,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
):
    """Compute Short-Time Energy (STE) for each frame.

    STE = sum(x[n]^2) / N

    Returns:
        tuple: (ste_values: np.ndarray, frame_times: np.ndarray)
    """
    frames, frame_times = frame_signal(
        signal, sample_rate, frame_duration_ms, hop_duration_ms, window="rectangular"
    )
    ste = np.mean(frames ** 2, axis=1)
    return ste, frame_times


def add_awgn_noise(
    signal: np.ndarray,
    snr_db: Optional[float],
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Add Additive White Gaussian Noise (AWGN) to achieve a target Signal-to-Noise Ratio (SNR in dB).

    Args:
        signal: 1D input audio array.
        snr_db: Target SNR in dB. If None or inf, returns a copy of the clean signal.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Noisy audio signal.
    """
    if snr_db is None or np.isinf(snr_db):
        return np.copy(signal)

    sig_power = float(np.mean(signal ** 2))
    if sig_power <= 0:
        return np.copy(signal)

    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(signal))
    return signal + noise
