"""Plugin 2: Bù trễ biên dựa trên Năng lượng (Edge Extension by Energy)."""
from typing import Any, Dict
import numpy as np
from scipy.ndimage import median_filter

from .base import BasePlugin


class EnergyExtensionPlugin(BasePlugin):
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
            # For AMDF: relaxing threshold means allowing slightly shallower dips (higher threshold)
            relaxed_thresh = base_thresh * (1.0 + self.threshold_discount)
            metric_check = lambda val: val <= relaxed_thresh
        else:
            # For ACF: relaxing threshold means allowing slightly lower peaks (lower threshold)
            reduced_thresh = base_thresh * (1.0 - self.threshold_discount)
            metric_check = lambda val: val >= reduced_thresh

        min_ste_required = self.ste_multiplier * ste_thresh

        # 1. Identify continuous voiced runs
        voiced_segments = []
        start = None
        for i in range(num_frames):
            if labels[i] == "v":
                if start is None:
                    start = i
            else:
                if start is not None:
                    voiced_segments.append((start, i - 1))
                    start = None
        if start is not None:
            voiced_segments.append((start, num_frames - 1))

        # 2. Inspect boundaries of stable voiced segments
        modified = False
        for start_idx, end_idx in voiced_segments:
            seg_len = end_idx - start_idx + 1
            if seg_len < self.min_voiced_len:
                continue

            # Check preceding frame
            pre_idx = start_idx - 1
            if pre_idx >= 0 and labels[pre_idx] == "uv":
                if ste[pre_idx] >= min_ste_required and metric_check(peak_values[pre_idx]):
                    cand_f0 = candidate_f0[pre_idx]
                    adj_f0 = f0_contour[start_idx]
                    if cand_f0 > 0 and adj_f0 > 0 and abs(cand_f0 - adj_f0) <= self.max_pitch_diff:
                        labels[pre_idx] = "v"
                        f0_contour[pre_idx] = cand_f0
                        modified = True

            # Check succeeding frame
            post_idx = end_idx + 1
            if post_idx < num_frames and labels[post_idx] == "uv":
                if ste[post_idx] >= min_ste_required and metric_check(peak_values[post_idx]):
                    cand_f0 = candidate_f0[post_idx]
                    adj_f0 = f0_contour[end_idx]
                    if cand_f0 > 0 and adj_f0 > 0 and abs(cand_f0 - adj_f0) <= self.max_pitch_diff:
                        labels[post_idx] = "v"
                        f0_contour[post_idx] = cand_f0
                        modified = True

        # 3. Optional median smoothing on updated voiced contour
        if modified:
            v_idx = np.where(f0_contour > 0)[0]
            if len(v_idx) >= 3:
                f0_contour[v_idx] = median_filter(f0_contour[v_idx], size=3)

        result["labels"] = labels
        result["f0_contour"] = f0_contour
        return result
