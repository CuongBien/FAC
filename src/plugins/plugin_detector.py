"""Plugin-enabled Pitch Detector wrapper coordinating the three pipeline stages:
1. Pre-processing: Signal filtering & frame spectral flattening
2. Detection & In-Loop Decision: Metric evaluation & optional frame decisions
3. Post-processing: Boundary energy extension & Viterbi tracking
"""
from typing import Dict, List, Optional
import numpy as np

from src.core.audio import load_wav
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.core.pitch_detector import PitchDetector
from src.core.processing import smooth_voiced_segments
from src.pipeline.contracts import FeatureSpec, PredictionResult, plugin_configuration
from src.pipeline.engine import extract_signal
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
    - decision_modifiers: Stage 2 plugins (optional frame decisions)
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

    def pre_processing_configuration(self):
        return plugin_configuration(self.pre_processors)

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
        pre = self.pre_processing_configuration()
        base = self.base_detector
        spec = FeatureSpec.from_detector(base, pre, plugin_wrapper=True)
        result = extract_signal(
            signal, sample_rate, spec, wav_path=wav_path, duration=duration,
            pre_processors=self.pre_processors, decision_modifiers=self.decision_modifiers,
            threshold=base.threshold, use_median_filter=base.use_median_filter,
            median_size=base.median_size, cache=base.cache,
            reset_plugins=self.plugins,
            extractors={'acf': find_f0_acf, 'amdf': find_f0_amdf, 'cepstrum': find_f0_cepstrum},
        )
        result.update(applied_plugins=[p.name for p in self.plugins], pre_processing_configuration=pre)
        return self._finalize_results(result)

    def process_feature_labels(self, features: Dict, labels: np.ndarray) -> Dict:
        """Apply native median and post-processing to aligned external decisions.

        Features must come from the same method, framing and pre-processing.
        The detector's threshold remains the fixed Energy periodicity guard;
        external scalar/vector decision thresholds do not silently alter it.
        """
        base = self.base_detector
        expected_pre = [p.name for p in self.pre_processors]
        feature_pre = features.get("feature_pre_processors")
        if (features.get("method") != base.method
                or features.get("frame_duration_ms") != base.frame_duration_ms
                or features.get("hop_duration_ms") != base.hop_duration_ms
                or features.get("f0_min") != base.f0_min
                or features.get("f0_max") != base.f0_max
                or feature_pre != expected_pre
                or features.get("pre_processing_configuration") != self.pre_processing_configuration()):
            raise ValueError("Features do not match detector method/framing/pre-processing")
        expected_contract = FeatureSpec.from_detector(base, self.pre_processing_configuration(), plugin_wrapper=True)
        if features.get('feature_contract') != expected_contract.to_dict():
            raise ValueError('Features do not match detector feature contract/gate/pre-processing')
        count = len(features["frame_times"])
        labels = np.array(labels, dtype=object, copy=True)
        if labels.shape != (count,) or not np.all(np.isin(labels, ["v", "uv", "sil"])):
            raise ValueError("External labels must align with frames and use v/uv/sil")
        active = np.asarray(features["ste"]) >= features["ste_thresh"]
        labels[~active] = "sil"
        candidate = np.asarray(features["candidate_f0"])
        valid = np.isfinite(candidate) & (candidate >= base.f0_min) & (candidate <= base.f0_max)
        labels[(labels == "v") & ~valid] = "uv"
        raw = np.where(labels == "v", candidate, 0.)
        contour = smooth_voiced_segments(raw, labels, base.median_size) if base.use_median_filter else raw.copy()
        result = {**features, "labels": labels, "f0_raw": raw, "f0_contour": contour,
                  "threshold": base.threshold, "use_median_filter": base.use_median_filter,
                  "median_size": base.median_size, "applied_plugins": [p.name for p in self.plugins]}
        for plugin in self.post_processors:
            plugin.reset()
        return self._finalize_results(result)

    def _finalize_results(self, initial_result: Dict) -> Dict:
        """Shared native post-processing and statistics for every decision rule."""
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

        return PredictionResult(result, self.base_detector.f0_min, self.base_detector.f0_max).to_dict()
