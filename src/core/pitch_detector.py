"""Pitch Detector engine: combines framing, ACF/AMDF/Cepstrum, silence detection, and V/UV thresholding."""
from typing import Dict, Literal, Optional
import numpy as np

from src.core.audio import load_wav
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.pipeline.contracts import FeatureSpec, PredictionResult
from src.pipeline.engine import extract_signal
from src import config


class PitchDetector:
    """Estimates F0 contour and V/UV/Silence classification for an entire audio signal."""

    def __init__(
        self,
        method: Literal["acf", "amdf", "cepstrum"] = "acf",
        frame_duration_ms: float = config.FRAME_DURATION_MS,
        hop_duration_ms: float = config.HOP_DURATION_MS,
        f0_min: float = config.F0_MIN,
        f0_max: float = config.F0_MAX,
        threshold: Optional[float] = None,
        ste_silence_ratio: float = config.STE_SILENCE_RATIO,
        use_median_filter: bool = True,
        median_size: int = config.MEDIAN_SIZE,
        cache=None,
    ):
        self.method = method.lower()
        default_threshold = config.baseline_threshold(self.method)
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = default_threshold if threshold is None else threshold
        self.ste_silence_ratio = ste_silence_ratio
        self.use_median_filter = use_median_filter
        self.median_size = median_size
        self.cache = cache

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
        spec = FeatureSpec.from_detector(self)
        result = extract_signal(
            signal, sample_rate, spec, wav_path=wav_path, duration=duration,
            threshold=self.threshold, use_median_filter=self.use_median_filter,
            median_size=self.median_size, cache=self.cache,
            extractors={'acf': find_f0_acf, 'amdf': find_f0_amdf, 'cepstrum': find_f0_cepstrum},
        )
        # Keep the baseline dictionary API while attaching a versioned contract.
        keys = ('wav_path', 'sample_rate', 'signal', 'duration', 'frame_times',
                'f0_contour', 'f0_raw', 'labels', 'peak_values', 'lags', 'f0_mean',
                'f0_std', 'num_frames', 'num_voiced', 'num_unvoiced', 'num_silence',
                'feature_contract')
        return PredictionResult({key: result[key] for key in keys}, self.f0_min, self.f0_max).to_dict()
