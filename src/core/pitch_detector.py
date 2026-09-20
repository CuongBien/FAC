"""Pitch Detector engine: combines framing, ACF/AMDF, silence detection, and V/UV thresholding."""
from typing import Dict, Literal, Optional
import numpy as np
from scipy.ndimage import median_filter

from src.core.audio import load_wav, frame_signal, compute_ste
from src.core.acf import find_f0_acf


class PitchDetector:
    """Estimates F0 contour and V/UV/Silence classification for an entire audio signal."""

    def __init__(
        self,
        method: Literal["acf", "amdf"] = "acf",
        frame_duration_ms: float = 30.0,
        hop_duration_ms: float = 10.0,
        f0_min: float = 70.0,
        f0_max: float = 400.0,
        threshold: float = 0.462,
        ste_silence_ratio: float = 0.008,
        use_median_filter: bool = True,
        median_size: int = 3,
    ):
        self.method = method
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = threshold
        self.ste_silence_ratio = ste_silence_ratio
        self.use_median_filter = use_median_filter
        self.median_size = median_size

    def process_file(self, wav_path: str) -> Dict:
        """Process a WAV file and return F0 contour, frame times, and labels."""
        sr, signal, duration = load_wav(wav_path)
        return self.process_signal(signal, sr, wav_path=wav_path, duration=duration)

    def process_signal(
        self,
        signal: np.ndarray,
        sample_rate: int,
        wav_path: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> Dict:
        """Process a 1D audio signal array."""
        if duration is None:
            duration = len(signal) / sample_rate

        frames, frame_times = frame_signal(
            signal,
            sample_rate,
            frame_duration_ms=self.frame_duration_ms,
            hop_duration_ms=self.hop_duration_ms,
            window="rectangular",
        )
        ste, _ = compute_ste(
            signal,
            sample_rate,
            frame_duration_ms=self.frame_duration_ms,
            hop_duration_ms=self.hop_duration_ms,
        )

        # Silence threshold: ratio of maximum short-time energy
        max_ste = np.max(ste) if len(ste) > 0 else 1.0
        ste_thresh = self.ste_silence_ratio * max_ste

        num_frames = len(frames)
        f0_raw = np.zeros(num_frames, dtype=np.float64)
        labels = np.full(num_frames, "sil", dtype=object)
        peak_values = np.zeros(num_frames, dtype=np.float64)
        lags = np.zeros(num_frames, dtype=int)

        for i in range(num_frames):
            if ste[i] < ste_thresh:
                labels[i] = "sil"
                f0_raw[i] = 0.0
                continue

            if self.method == "acf":
                f0_val, peak_val, lag = find_f0_acf(
                    frames[i],
                    sample_rate,
                    f0_min=self.f0_min,
                    f0_max=self.f0_max,
                    mode="normalized",
                )
                peak_values[i] = peak_val
                lags[i] = lag

                if peak_val >= self.threshold:
                    labels[i] = "v"
                    f0_raw[i] = f0_val
                else:
                    labels[i] = "uv"
                    f0_raw[i] = 0.0

        # Post-processing: Median filter on continuous voiced segments to remove isolated octave jumps
        f0_contour = np.copy(f0_raw)
        if self.use_median_filter:
            voiced_idx = np.where(f0_contour > 0)[0]
            if len(voiced_idx) >= self.median_size:
                f0_contour[voiced_idx] = median_filter(f0_contour[voiced_idx], size=self.median_size)

        voiced_f0 = f0_contour[f0_contour > 0]
        f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        return {
            "wav_path": wav_path,
            "sample_rate": sample_rate,
            "signal": signal,
            "duration": duration,
            "frame_times": frame_times,
            "f0_contour": f0_contour,
            "f0_raw": f0_raw,
            "labels": labels,
            "peak_values": peak_values,
            "lags": lags,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "num_frames": num_frames,
            "num_voiced": int(np.sum(labels == "v")),
            "num_unvoiced": int(np.sum(labels == "uv")),
            "num_silence": int(np.sum(labels == "sil")),
        }
