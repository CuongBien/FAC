"""Stage 3 Post-processing Plugin: Bù trễ biên dựa trên Năng lượng (Edge Extension by Energy)."""
from typing import Any, Dict
import numpy as np
from scipy.ndimage import median_filter

from src.plugins.base import PostProcessingPlugin


class EnergyExtensionPlugin(PostProcessingPlugin):
    """Extends voiced segment boundaries when unvoiced boundary frames carry significant energy and near-pitch periodicity.

    - Target: Frames immediately before or after a stable voiced segment (>= min_voiced_len frames).
    - Energy Condition: STE >= ste_multiplier * ste_thresh (sound has not completely died down).
    - Correlation Condition: ACF peak >= threshold * (1 - threshold_discount) (e.g. 15-20% discount).
    - Pitch Continuity: Candidate pitch within max_pitch_diff Hz of the adjacent voiced frame.
    """

    def __init__(
        self,
        threshold_discount: float = 0.20,
        min_voiced_len: int = 3,
        max_pitch_diff: float = 40.0,
        ste_multiplier: float = 1.5,
    ):
        super().__init__(name="EnergyEdgeExtension")
        self.threshold_discount = threshold_discount
        self.min_voiced_len = min_voiced_len
        self.max_pitch_diff = max_pitch_diff
        self.ste_multiplier = ste_multiplier

    def post_process_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        labels = np.copy(result["labels"])
        f0_contour = np.copy(result["f0_contour"])
        candidate_f0 = result.get("candidate_f0")
        peak_values = result.get("peak_values")
        ste = result.get("ste")
        ste_thresh = result.get("ste_thresh", 0.0)
        base_thresh = result.get("threshold", 0.462)

        if candidate_f0 is None or peak_values is None or ste is None:
            return result

        num_frames = len(labels)
        method = result.get("method", "acf").lower()
        if method == "amdf":
            # For AMDF: smaller dip is better, discount increases tolerable threshold
            effective_thresh = base_thresh * (1.0 + self.threshold_discount)
            def check_periodicity(val):
                return val <= effective_thresh
        else:
            effective_thresh = base_thresh * (1.0 - self.threshold_discount)
            def check_periodicity(val):
                return val >= effective_thresh

        min_energy = self.ste_multiplier * ste_thresh

        # Find continuous voiced segments
        in_segment = False
        start_idx = 0
        segments = []

        for i in range(num_frames):
            if labels[i] == "v":
                if not in_segment:
                    in_segment = True
                    start_idx = i
            else:
                if in_segment:
                    in_segment = False
                    segments.append((start_idx, i - 1))
        if in_segment:
            segments.append((start_idx, num_frames - 1))

        # Check and expand each segment
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start + 1
            if seg_len < self.min_voiced_len:
                continue

            # 1. Expand left (onset)
            curr = seg_start - 1
            while curr >= 0 and labels[curr] != "v":
                if labels[curr] == "sil":
                    break
                if ste[curr] < min_energy:
                    break
                if not check_periodicity(peak_values[curr]):
                    break

                cand = candidate_f0[curr]
                adj_f0 = f0_contour[curr + 1]
                if abs(cand - adj_f0) > self.max_pitch_diff:
                    break

                labels[curr] = "v"
                f0_contour[curr] = cand
                curr -= 1

            # 2. Expand right (offset)
            curr = seg_end + 1
            while curr < num_frames and labels[curr] != "v":
                if labels[curr] == "sil":
                    break
                if ste[curr] < min_energy:
                    break
                if not check_periodicity(peak_values[curr]):
                    break

                cand = candidate_f0[curr]
                adj_f0 = f0_contour[curr - 1]
                if abs(cand - adj_f0) > self.max_pitch_diff:
                    break

                labels[curr] = "v"
                f0_contour[curr] = cand
                curr += 1

        # Smooth newly added pitch frames with a local median filter
        v_idx = np.where(labels == "v")[0]
        if len(v_idx) >= 3:
            f0_contour[v_idx] = median_filter(f0_contour[v_idx], size=3)

        result["labels"] = labels
        result["f0_contour"] = f0_contour
        return result
