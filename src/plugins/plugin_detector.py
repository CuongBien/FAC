"""Plugin-enabled Pitch Detector wrapper coordinating the three pipeline stages:
1. Pre-processing: Signal filtering & frame spectral flattening
2. Detection & In-Loop Decision: Metric evaluation & adaptive hysteresis thresholding
3. Post-processing: Boundary energy extension & Viterbi tracking
"""
from typing import Dict, List, Optional
import numpy as np

from src.core.audio import load_wav, frame_signal, compute_ste
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.core.pitch_detector import PitchDetector
from src.core.processing import (
    center_frame,
    prepare_analysis_signal,
    silence_energy_threshold,
    smooth_voiced_segments,
)
from .base import (
    BasePlugin,
    PluginStage,
    PreProcessingPlugin,
    DecisionPlugin,
    PostProcessingPlugin,
)


class PluginPitchDetector:
    """Wrapper that executes a pipeline of plugins around baseline PitchDetector logic,
    strictly organized by pipeline stages:
    - pre_processors: Stage 1 plugins (Bandpass filter, Center clipping)
    - decision_modifiers: Stage 2 plugins (Hysteresis thresholding)
    - post_processors: Stage 3 plugins (Energy edge extension, Viterbi tracking)
    """

    def __init__(
        self,
        base_detector: Optional[PitchDetector] = None,
        plugins: Optional[List[BasePlugin]] = None,
        pre_processing: Optional[List[BasePlugin]] = None,
        decision: Optional[List[BasePlugin]] = None,
        post_processing: Optional[List[BasePlugin]] = None,
    ):
        self.base_detector = base_detector if base_detector is not None else PitchDetector()

        self.pre_processors: List[BasePlugin] = list(pre_processing) if pre_processing is not None else []
        self.decision_modifiers: List[BasePlugin] = list(decision) if decision is not None else []
        self.post_processors: List[BasePlugin] = list(post_processing) if post_processing is not None else []

        if plugins is not None:
            for p in plugins:
                self.add_plugin(p)

    @property
    def plugins(self) -> List[BasePlugin]:
        """All registered plugins in pipeline execution order."""
        return self.pre_processors + self.decision_modifiers + self.post_processors

    def add_plugin(self, plugin: BasePlugin) -> "PluginPitchDetector":
        """Add a plugin to its corresponding stage in the pipeline."""
        stage = getattr(plugin, "stage", PluginStage.PRE_PROCESSING)
        if stage == PluginStage.PRE_PROCESSING or isinstance(plugin, PreProcessingPlugin):
            self.pre_processors.append(plugin)
        elif stage == PluginStage.DECISION or isinstance(plugin, DecisionPlugin):
            self.decision_modifiers.append(plugin)
        elif stage == PluginStage.POST_PROCESSING or isinstance(plugin, PostProcessingPlugin):
            self.post_processors.append(plugin)
        else:
            self.pre_processors.append(plugin)
        return self

    def clear_plugins(self) -> "PluginPitchDetector":
        """Clear all registered plugins across all pipeline stages."""
        self.pre_processors.clear()
        self.decision_modifiers.clear()
        self.post_processors.clear()
        return self

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
        """Process an in-memory 1D audio array through the 3-stage plugin pipeline."""
        analysis_signal = prepare_analysis_signal(
            signal, sample_rate, self.base_detector.frame_duration_ms
        )
        if duration is None:
            duration = len(signal) / sample_rate

        # 1. Reset all plugins in all stages
        for p in self.plugins:
            p.reset()

        # =========================================================================
        # STAGE 1: PRE-PROCESSING (Signal-level filtering)
        # =========================================================================
        processed_signal = np.copy(analysis_signal)
        for p in self.pre_processors:
            processed_signal = p.pre_process_signal(processed_signal, sample_rate)

        # 2. Framing
        frames, frame_times = frame_signal(
            processed_signal,
            sample_rate,
            frame_duration_ms=self.base_detector.frame_duration_ms,
            hop_duration_ms=self.base_detector.hop_duration_ms,
            window="rectangular",
        )
        # Gate on DC-free speech energy before signal-level plugins.
        ste, _ = compute_ste(
            analysis_signal,
            sample_rate,
            frame_duration_ms=self.base_detector.frame_duration_ms,
            hop_duration_ms=self.base_detector.hop_duration_ms,
        )

        ste_thresh = silence_energy_threshold(ste, self.base_detector.ste_silence_ratio)

        num_frames = len(frames)
        f0_raw = np.zeros(num_frames, dtype=np.float64)
        labels = np.full(num_frames, "sil", dtype=object)
        peak_values = np.zeros(num_frames, dtype=np.float64)
        lags = np.zeros(num_frames, dtype=int)
        candidate_f0 = np.zeros(num_frames, dtype=np.float64)
        pitch_frames = np.copy(frames)

        method = getattr(self.base_detector, "method", "acf").lower()

        # =========================================================================
        # STAGE 2: PITCH DETECTION & IN-LOOP DECISION (Frame-level)
        # =========================================================================
        for i in range(num_frames):
            is_silence = ste[i] < ste_thresh
            context = {
                "frame_idx": i,
                "is_silence": is_silence,
                "ste": ste[i],
                "sample_rate": sample_rate,
                "frame": frames[i],
                "method": method,
            }

            if is_silence:
                labels[i] = "sil"
                f0_raw[i] = 0.0
                candidate_f0[i] = 0.0
                # Notify decision modifiers of silence frame
                for p in self.decision_modifiers:
                    p.adjust_frame_decision(i, 0.0, 0.0, False, context)
                continue

            # Stage 1 (cont.): Frame-level pre-processing (Center Clipping)
            pitch_frame = center_frame(frames[i])
            for p in self.pre_processors:
                pitch_frame = p.pre_process_frame(pitch_frame, sample_rate)
            pitch_frames[i] = pitch_frame

            # Core pitch extraction
            if method == "amdf":
                f0_val, dip_val, lag = find_f0_amdf(
                    pitch_frame,
                    sample_rate,
                    f0_min=self.base_detector.f0_min,
                    f0_max=self.base_detector.f0_max,
                    mode="normalized",
                )
                peak_values[i] = dip_val
                lags[i] = lag
                candidate_f0[i] = f0_val
                is_voiced = dip_val <= self.base_detector.threshold
                current_metric = dip_val
            elif method == "cepstrum":
                f0_val, peak_val, lag = find_f0_cepstrum(
                    pitch_frame,
                    sample_rate,
                    f0_min=self.base_detector.f0_min,
                    f0_max=self.base_detector.f0_max,
                    window="hamming",
                    n_fft=2048,
                    mode="peak",
                )
                peak_values[i] = peak_val
                lags[i] = lag
                candidate_f0[i] = f0_val
                is_voiced = peak_val >= self.base_detector.threshold
                current_metric = peak_val
            else:
                f0_val, peak_val, lag = find_f0_acf(
                    pitch_frame,
                    sample_rate,
                    f0_min=self.base_detector.f0_min,
                    f0_max=self.base_detector.f0_max,
                    mode="normalized",
                )
                peak_values[i] = peak_val
                lags[i] = lag
                candidate_f0[i] = f0_val
                is_voiced = peak_val >= self.base_detector.threshold
                current_metric = peak_val

            # Stage 2: Decision adjustment hooks (Hysteresis thresholding)
            for p in self.decision_modifiers:
                is_voiced, f0_val = p.adjust_frame_decision(
                    i, current_metric, f0_val, is_voiced, context
                )

            if is_voiced and np.isfinite(f0_val) and self.base_detector.f0_min <= f0_val <= self.base_detector.f0_max:
                labels[i] = "v"
                f0_raw[i] = f0_val
            else:
                labels[i] = "uv"
                f0_raw[i] = 0.0

        # Baseline Median filter smoothing
        f0_contour = np.copy(f0_raw)
        if self.base_detector.use_median_filter:
            f0_contour = smooth_voiced_segments(
                f0_contour, labels, self.base_detector.median_size
            )

        initial_result = {
            "wav_path": wav_path,
            "sample_rate": sample_rate,
            "signal": signal,
            "processed_signal": processed_signal,
            "pitch_frames": pitch_frames,
            "frame_duration_ms": self.base_detector.frame_duration_ms,
            "hop_duration_ms": self.base_detector.hop_duration_ms,
            "f0_min": self.base_detector.f0_min,
            "f0_max": self.base_detector.f0_max,
            "use_median_filter": self.base_detector.use_median_filter,
            "median_size": self.base_detector.median_size,
            "duration": duration,
            "frame_times": frame_times,
            "f0_contour": f0_contour,
            "f0_raw": f0_raw,
            "candidate_f0": candidate_f0,
            "labels": labels,
            "peak_values": peak_values,
            "lags": lags,
            "ste": ste,
            "ste_thresh": ste_thresh,
            "threshold": self.base_detector.threshold,
            "num_frames": num_frames,
            "method": method,
            "applied_plugins": [p.name for p in self.plugins],
        }

        # =========================================================================
        # STAGE 3: POST-PROCESSING (Boundary extension & Viterbi tracking)
        # =========================================================================
        result = initial_result
        for p in self.post_processors:
            result = p.post_process_results(result)

        # Keep final labels/counts consistent even if a post-processor rejects pitch.
        valid_pitch = (
            np.isfinite(result["f0_contour"])
            & (result["f0_contour"] >= self.base_detector.f0_min)
            & (result["f0_contour"] <= self.base_detector.f0_max)
        )
        result["labels"][(result["labels"] == "v") & ~valid_pitch] = "uv"
        result["f0_contour"][result["labels"] != "v"] = 0.0

        # Final statistics recalculation
        v_f0 = result["f0_contour"][result["f0_contour"] > 0]
        result["f0_mean"] = float(np.mean(v_f0)) if len(v_f0) > 0 else 0.0
        result["f0_std"] = float(np.std(v_f0)) if len(v_f0) > 0 else 0.0
        result["num_voiced"] = int(np.sum(result["labels"] == "v"))
        result["num_unvoiced"] = int(np.sum(result["labels"] == "uv"))
        result["num_silence"] = int(np.sum(result["labels"] == "sil"))

        return result
