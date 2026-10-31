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

    if data.size == 0:
        raise ValueError("Audio signal is empty")

    # Normalize PCM before averaging channels; uint8 PCM silence is 128.
    if np.issubdtype(data.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(data.dtype).max + 1) / 2.0
        signal = (data.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(data.dtype, np.signedinteger):
        info = np.iinfo(data.dtype)
        signal = data.astype(np.float64) / max(abs(int(info.min)), int(info.max))
    else:
        signal = data.astype(np.float64)
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal = signal / max_abs

    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

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
    if frame_len < 1 or hop_len < 1 or sample_rate <= 0:
        raise ValueError("Frame length, hop length and sample rate must be positive")
    if np.ndim(signal) != 1 or len(signal) == 0:
        raise ValueError("Audio must be a non-empty one-dimensional signal")

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

    STE = sum((x[n] - mean(x))^2) / N; constant DC is not speech energy.

    Returns:
        tuple: (ste_values: np.ndarray, frame_times: np.ndarray)
    """
    frames, frame_times = frame_signal(
        signal, sample_rate, frame_duration_ms, hop_duration_ms, window="rectangular"
    )
    centered = frames - np.mean(frames, axis=1, keepdims=True)
    ste = np.mean(centered ** 2, axis=1)
    return ste, frame_times
