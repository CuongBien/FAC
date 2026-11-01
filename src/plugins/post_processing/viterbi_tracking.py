"""Viterbi pitch tracking on the detector's actual preprocessed frames."""
from typing import Any, Dict, List
import numpy as np

from src.plugins.base import PostProcessingPlugin
from src.pipeline.cache import cached_call
from src.core.acf import compute_acf
from src.core.amdf import compute_amdf
from src.core.cepstrum import compute_real_cepstrum
from src.core.processing import center_frame, pitch_lag_bounds, refined_f0, voiced_segments


class ViterbiTrackingPlugin(PostProcessingPlugin):
    """Optimize candidate pitch paths independently within voiced segments.

    Pipeline results supply pitch_frames and detector configuration. No signal is
    reframed here, so center clipping, frame duration and frame alignment survive.
    """

    def __init__(self, w_freq=3.0, w_octave=4.0, max_candidates=5,
                 f0_min=70.0, f0_max=400.0, cache=None):
        super().__init__(name="ViterbiTracking")
        self.w_freq = w_freq
        self.w_octave = w_octave
        self.max_candidates = max_candidates
        self.f0_min = f0_min
        self.f0_max = f0_max
        self._cache = cache

    def _curve_candidates(self, curve, sample_rate, method, f0_min, f0_max):
        last_lag = len(curve) // 2 if method == "cepstrum" else len(curve) - 2
        low, high = pitch_lag_bounds(sample_rate, f0_min, f0_max, last_lag)
        if low > high:
            return []
        minimize = method == "amdf"
        extrema = []
        for lag in range(low, high + 1):
            if minimize:
                is_extremum = curve[lag] < curve[lag - 1] and curve[lag] <= curve[lag + 1]
            else:
                is_extremum = curve[lag] > curve[lag - 1] and curve[lag] >= curve[lag + 1]
            if is_extremum:
                extrema.append(lag)
        if not extrema:
            select = np.argmin if minimize else np.argmax
            extrema = [low + int(select(curve[low:high + 1]))]
        extrema.sort(key=lambda lag: curve[lag], reverse=not minimize)
        extrema = extrema[:self.max_candidates]
        peak_scale = max((float(curve[lag]) for lag in extrema), default=1.0)
        if peak_scale <= 0:
            peak_scale = 1.0
        candidates = []
        for lag in extrema:
            value = float(curve[lag])
            if minimize:
                cost = float(np.clip(value, 0.0, 1.0))
            elif method == "cepstrum":
                cost = max(0.0, 1.0 - value / peak_scale)
            else:
                cost = max(0.0, 1.0 - value)
            candidates.append({
                "f0": refined_f0(curve, lag, sample_rate, f0_min, f0_max),
                "cost": cost,
            })
        return candidates

    def _extract_acf_candidates(self, frame, sample_rate, f0_min=None, f0_max=None):
        return cached_call(self._cache, self._compute_acf_candidates, frame, sample_rate,
                           f0_min=f0_min, f0_max=f0_max,
                           namespace=(self.f0_min, self.f0_max, self.max_candidates))

    def _compute_acf_candidates(self, frame, sample_rate, f0_min=None, f0_max=None):
        frame = center_frame(frame)
        if not len(frame) or np.mean(frame ** 2) <= 1e-12:
            return []
        return self._curve_candidates(
            compute_acf(frame), sample_rate, "acf",
            self.f0_min if f0_min is None else f0_min,
            self.f0_max if f0_max is None else f0_max,
        )

    def _extract_amdf_candidates(self, frame, sample_rate, f0_min=None, f0_max=None):
        return cached_call(self._cache, self._compute_amdf_candidates, frame, sample_rate,
                           f0_min=f0_min, f0_max=f0_max,
                           namespace=(self.f0_min, self.f0_max, self.max_candidates))

    def _compute_amdf_candidates(self, frame, sample_rate, f0_min=None, f0_max=None):
        frame = center_frame(frame)
        if not len(frame) or np.mean(frame ** 2) <= 1e-12:
            return []
        return self._curve_candidates(
            compute_amdf(frame), sample_rate, "amdf",
            self.f0_min if f0_min is None else f0_min,
            self.f0_max if f0_max is None else f0_max,
        )

    def _extract_cepstrum_candidates(self, frame, sample_rate, f0_min=None, f0_max=None):
        return cached_call(self._cache, self._compute_cepstrum_candidates, frame, sample_rate,
                           f0_min=f0_min, f0_max=f0_max,
                           namespace=(self.f0_min, self.f0_max, self.max_candidates))

    def _compute_cepstrum_candidates(self, frame, sample_rate, f0_min=None, f0_max=None):
        frame = center_frame(frame)
        if not len(frame) or np.mean(frame ** 2) <= 1e-12:
            return []
        cepstrum, _ = compute_real_cepstrum(frame, sample_rate, window="hamming", n_fft=2048)
        return self._curve_candidates(
            cepstrum, sample_rate, "cepstrum",
            self.f0_min if f0_min is None else f0_min,
            self.f0_max if f0_max is None else f0_max,
        )

    def _transition_cost(self, f0_prev, f0_curr):
        if f0_prev <= 0 or f0_curr <= 0:
            return 1.0
        octave_diff = abs(np.log2(f0_curr / f0_prev))
        cost = self.w_freq * octave_diff ** 2
        if 0.8 <= octave_diff <= 1.2 or 1.8 <= octave_diff <= 2.2:
            cost += self.w_octave
        return cost

    def _viterbi_segment(self, segment_candidates: List[List[Dict[str, float]]]):
        if not segment_candidates:
            return []
        costs = [[c["cost"] for c in segment_candidates[0]]]
        pointers = [[-1] * len(segment_candidates[0])]
        for t in range(1, len(segment_candidates)):
            current_costs, current_pointers = [], []
            for current in segment_candidates[t]:
                totals = [
                    costs[-1][i] + self._transition_cost(previous["f0"], current["f0"])
                    + current["cost"]
                    for i, previous in enumerate(segment_candidates[t - 1])
                ]
                best = int(np.argmin(totals))
                current_costs.append(totals[best])
                current_pointers.append(best)
            costs.append(current_costs)
            pointers.append(current_pointers)
        path = [int(np.argmin(costs[-1]))]
        for t in range(len(segment_candidates) - 1, 0, -1):
            path.append(pointers[t][path[-1]])
        path.reverse()
        return [segment_candidates[t][i]["f0"] for t, i in enumerate(path)]

    def post_process_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        labels = result["labels"]
        if not len(labels) or not np.any(labels == "v"):
            return result
        if "pitch_frames" not in result:
            raise ValueError("Viterbi requires the detector's preprocessed pitch_frames")
        frames = result["pitch_frames"]
        if len(frames) != len(labels):
            raise ValueError("Viterbi pitch_frames and labels must have the same length")
        sample_rate = result["sample_rate"]
        f0_min = result.get("f0_min", self.f0_min)
        f0_max = result.get("f0_max", self.f0_max)
        method = result.get("method", "acf").lower()
        extract = {
            "acf": self._extract_acf_candidates,
            "amdf": self._extract_amdf_candidates,
            "cepstrum": self._extract_cepstrum_candidates,
        }[method]
        optimized = np.copy(result["f0_contour"])
        for start, end in voiced_segments(labels == "v"):
            if end - start < 2:
                continue
            segment_candidates = []
            for index in range(start, end):
                candidates = extract(frames[index], sample_rate, f0_min=f0_min, f0_max=f0_max)
                candidates = [
                    c for c in candidates if np.isfinite(c["f0"]) and f0_min <= c["f0"] <= f0_max
                ]
                original = float(optimized[index])
                if np.isfinite(original) and f0_min <= original <= f0_max:
                    if not any(abs(c["f0"] - original) < 1.0 for c in candidates):
                        candidates.append({"f0": original, "cost": 0.1})
                if not candidates:
                    candidates = [{"f0": 0.0, "cost": 1.0}]
                segment_candidates.append(candidates)
            optimized[start:end] = self._viterbi_segment(segment_candidates)
        result["f0_contour"] = optimized
        return result
