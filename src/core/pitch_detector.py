"""Pitch Detector engine: combines framing, ACF/AMDF, silence detection, and V/UV thresholding."""
import numpy as np
from typing import Dict, Literal


class PitchDetector:
    """Estimates F0 contour and V/UV/Silence classification for an entire audio signal."""

    def __init__(
        self,
        method: Literal["acf", "amdf"] = "acf",
        frame_duration_ms: float = 30.0,
        hop_duration_ms: float = 10.0,
        f0_min: float = 70.0,
        f0_max: float = 400.0,
        threshold: float = 0.5,
        ste_silence_threshold: float = 0.005,
    ):
        self.method = method
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = threshold
        self.ste_silence_threshold = ste_silence_threshold

    def process_file(self, wav_path: str) -> Dict:
        """Process a WAV file and return F0 contour, frame times, and labels.

        Returns:
            dict: {
                'frame_times': np.ndarray,
                'f0_contour': np.ndarray, (0 for unvoiced/silence)
                'labels': np.ndarray, ('v', 'uv', 'sil')
                'f0_mean': float,
                'f0_std': float
            }
        """
        pass
