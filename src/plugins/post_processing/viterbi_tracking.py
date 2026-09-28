"""Stage 3 Post-processing Plugin: Quy hoạch động Viterbi làm mượt cao độ (Viterbi Tracking).

Giải quyết triệt để lỗi nhảy quãng tám (Pitch Doubling / Pitch Halving) và formant ripple:
- Hoạt động trên các đoạn âm hữu thanh (Voiced segments) đã được phân loại chuẩn xác.
- Tại mỗi khung, trích xuất Top-K cực trị cục bộ (Local Peaks đối với ACF, Local Dips đối với AMDF).
- Thuật toán Viterbi tìm đường đi mượt mà nhất tối ưu toàn cục dựa trên:
  * Chi phí cục bộ (Local cost): Độ cao tương quan 1 - R(tau) (ACF) hoặc độ sâu D(tau) (AMDF).
  * Chi phí chuyển tiếp (Transition cost): Phạt bước nhảy tần số log2 (bán âm) và phạt nặng bước nhảy quãng tám (0.8 - 1.2 octave).
"""
from typing import Any, Dict, List, Optional
import numpy as np

from src.plugins.base import PostProcessingPlugin
from src.core.audio import frame_signal
from src.core.acf import compute_acf
from src.core.amdf import compute_amdf
from src.core.yin import (
    compute_difference_function,
    cumulative_mean_normalized_difference,
    parabolic_interpolation,
)


class ViterbiTrackingPlugin(PostProcessingPlugin):
    """Dynamic programming (Viterbi) pitch tracker for voiced segments.

    Eliminates octave jumps and sudden pitch contours anomalies by solving the shortest
    path problem across candidate peaks/dips in continuous voiced segments.
    """

    def __init__(
        self,
        w_freq: float = 3.0,
        w_octave: float = 2.0,
        max_candidates: int = 5,
        f0_min: float = 70.0,
        f0_max: float = 400.0,
    ):
        super().__init__(name="ViterbiTracking")
        self.w_freq = w_freq
        self.w_octave = w_octave
        self.max_candidates = max_candidates
        self.f0_min = f0_min
        self.f0_max = f0_max

    def _extract_acf_candidates(
        self, frame: np.ndarray, sample_rate: int
    ) -> List[Dict[str, float]]:
        r = compute_acf(frame, mode="normalized")
        lag_min = max(1, int(round(sample_rate / self.f0_max)))
        lag_max = min(len(r) - 2, int(round(sample_rate / self.f0_min)))

        if lag_min >= lag_max or lag_max >= len(r) - 1:
            return [{"f0": 0.0, "cost": 1.0}]

        search_slice = r[lag_min : lag_max + 1]
        peaks = []
        for i in range(1, len(search_slice) - 1):
            if search_slice[i] > search_slice[i - 1] and search_slice[i] >= search_slice[i + 1]:
                lag = lag_min + i
                val = float(search_slice[i])
                f0 = float(sample_rate / lag)
                peaks.append((val, f0))

        if not peaks:
            best_idx = int(np.argmax(search_slice))
            best_lag = lag_min + best_idx
            best_val = float(search_slice[best_idx])
            peaks.append((best_val, float(sample_rate / best_lag)))

        peaks.sort(key=lambda x: x[0], reverse=True)
        peaks = peaks[: self.max_candidates]

        candidates = []
        for val, f0 in peaks:
            cost = max(0.0, 1.0 - val)
            candidates.append({"f0": f0, "cost": cost})

        return candidates

    def _extract_amdf_candidates(
        self, frame: np.ndarray, sample_rate: int
    ) -> List[Dict[str, float]]:
        d = compute_amdf(frame, mode="normalized")
        lag_min = max(1, int(round(sample_rate / self.f0_max)))
        lag_max = min(len(d) - 2, int(round(sample_rate / self.f0_min)))

        if lag_min >= lag_max or lag_max >= len(d) - 1:
            return [{"f0": 0.0, "cost": 1.0}]

        search_slice = d[lag_min : lag_max + 1]
        dips = []
        for i in range(1, len(search_slice) - 1):
            if search_slice[i] < search_slice[i - 1] and search_slice[i] <= search_slice[i + 1]:
                lag = lag_min + i
                val = float(search_slice[i])
                f0 = float(sample_rate / lag)
                dips.append((val, f0))

        if not dips:
            best_idx = int(np.argmin(search_slice))
            best_lag = lag_min + best_idx
            best_val = float(search_slice[best_idx])
            dips.append((best_val, float(sample_rate / best_lag)))

        dips.sort(key=lambda x: x[0])
        dips = dips[: self.max_candidates]

        candidates = []
        for val, f0 in dips:
            cost = max(0.0, min(1.0, val))
            candidates.append({"f0": f0, "cost": cost})

        return candidates

    def _extract_yin_candidates(
        self, frame: np.ndarray, sample_rate: int
    ) -> List[Dict[str, float]]:
        lag_min = max(1, int(round(sample_rate / self.f0_max)))
        lag_max = min(len(frame) - 1, int(round(sample_rate / self.f0_min)))
        if lag_min >= lag_max:
            return [{"f0": 0.0, "cost": 1.0}]

        d = compute_difference_function(frame, max_lag=lag_max)
        d_prime = cumulative_mean_normalized_difference(d)

        dips = []
        for lag in range(lag_min, min(lag_max, len(d_prime) - 1)):
            if d_prime[lag] < d_prime[lag - 1] and d_prime[lag] <= d_prime[lag + 1]:
                tau_fine, val_fine = parabolic_interpolation(d_prime, lag)
                f0 = float(sample_rate / tau_fine) if tau_fine > 0 else 0.0
                if self.f0_min * 0.9 <= f0 <= self.f0_max * 1.1:
                    dips.append((val_fine, f0))

        if not dips:
            search_slice = d_prime[lag_min : lag_max + 1]
            if len(search_slice) > 0:
                rel_idx = int(np.argmin(search_slice))
                best_lag = lag_min + rel_idx
                tau_fine, val_fine = parabolic_interpolation(d_prime, best_lag)
                f0 = float(sample_rate / tau_fine) if tau_fine > 0 else 0.0
                dips.append((val_fine, f0))

        dips.sort(key=lambda x: x[0])
        dips = dips[: self.max_candidates]

        candidates = []
        for val, f0 in dips:
            cost = max(0.0, min(1.0, val))
            candidates.append({"f0": f0, "cost": cost})

        if not candidates:
            return [{"f0": 0.0, "cost": 1.0}]
        return candidates

    def _transition_cost(self, f0_prev: float, f0_curr: float) -> float:
        if f0_prev <= 0 or f0_curr <= 0:
            return 1.0

        octave_diff = abs(np.log2(f0_curr / f0_prev))
        cost = self.w_freq * (octave_diff ** 2)

        if 0.8 <= octave_diff <= 1.2 or 1.8 <= octave_diff <= 2.2:
            cost += self.w_octave

        return cost

    def _viterbi_segment(
        self, segment_candidates: List[List[Dict[str, float]]]
    ) -> List[float]:
        t_len = len(segment_candidates)
        if t_len == 0:
            return []
        if t_len == 1:
            return [min(segment_candidates[0], key=lambda c: c["cost"])["f0"]]

        trellis = []
        backpointers = []

        first_cands = segment_candidates[0]
        trellis.append([c["cost"] for c in first_cands])
        backpointers.append([-1] * len(first_cands))

        for t in range(1, t_len):
            curr_cands = segment_candidates[t]
            prev_cands = segment_candidates[t - 1]
            prev_trellis = trellis[t - 1]

            curr_trellis = []
            curr_bp = []

            for j, c_curr in enumerate(curr_cands):
                min_total_cost = float("inf")
                best_prev_idx = 0

                for i, c_prev in enumerate(prev_cands):
                    trans = self._transition_cost(c_prev["f0"], c_curr["f0"])
                    total_cost = prev_trellis[i] + trans + c_curr["cost"]

                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        best_prev_idx = i

                curr_trellis.append(min_total_cost)
                curr_bp.append(best_prev_idx)

            trellis.append(curr_trellis)
            backpointers.append(curr_bp)

        best_last_idx = int(np.argmin(trellis[-1]))
        best_path = [best_last_idx]

        for t in range(t_len - 1, 0, -1):
            prev_idx = backpointers[t][best_path[-1]]
            best_path.append(prev_idx)

        best_path.reverse()

        smooth_f0 = [
            segment_candidates[t][best_path[t]]["f0"] for t in range(t_len)
        ]
        return smooth_f0

    def post_process_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        labels = result["labels"]
        num_frames = len(labels)
        if num_frames == 0 or np.sum(labels == "v") == 0:
            return result

        method = result.get("method", "acf").lower()
        signal = result.get("processed_signal", result.get("signal"))
        sample_rate = result.get("sample_rate", 16000)

        frames, _ = frame_signal(
            signal,
            sample_rate,
            frame_duration_ms=25.0,
            hop_duration_ms=10.0,
            window="rectangular",
        )

        in_voiced = False
        seg_start = 0
        segments = []

        for i in range(num_frames):
            if labels[i] == "v":
                if not in_voiced:
                    in_voiced = True
                    seg_start = i
            else:
                if in_voiced:
                    in_voiced = False
                    segments.append((seg_start, i - 1))
        if in_voiced:
            segments.append((seg_start, num_frames - 1))

        f0_optimized = np.copy(result["f0_contour"])

        for start_idx, end_idx in segments:
            seg_len = end_idx - start_idx + 1
            if seg_len < 2:
                continue

            seg_cands = []
            for idx in range(start_idx, end_idx + 1):
                if idx < len(frames):
                    frame_data = frames[idx]
                    if method == "amdf":
                        cands = self._extract_amdf_candidates(frame_data, sample_rate)
                    elif method == "yin":
                        cands = self._extract_yin_candidates(frame_data, sample_rate)
                    else:
                        cands = self._extract_acf_candidates(frame_data, sample_rate)
                else:
                    cands = [{"f0": float(f0_optimized[idx]), "cost": 0.0}]

                orig_f0 = float(f0_optimized[idx])
                if orig_f0 > 0 and not any(abs(c["f0"] - orig_f0) < 1.0 for c in cands):
                    cands.append({"f0": orig_f0, "cost": 0.1})

                seg_cands.append(cands)

            smooth_seg = self._viterbi_segment(seg_cands)

            for offset, f0_val in enumerate(smooth_seg):
                f0_optimized[start_idx + offset] = f0_val

        result["f0_contour"] = f0_optimized
        return result
