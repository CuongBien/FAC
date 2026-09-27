"""Plugin 5: Quy hoạch động làm mượt cao độ (Viterbi / Dynamic Programming Pitch Tracking).

Giải quyết triệt để lỗi nhảy quãng tám (Pitch Doubling / Pitch Halving) và formant ripple:
- Hoạt động trên các đoạn âm hữu thanh (Voiced segments) đã được phân loại chuẩn xác.
- Tại mỗi khung, trích xuất Top-K cực trị cục bộ (Local Peaks đối với ACF, Local Dips đối với AMDF).
- Thuật toán Viterbi tìm đường đi mượt mà nhất tối ưu toàn cục dựa trên:
  * Chi phí cục bộ (Local cost): Độ cao tương quan 1 - R(tau) (ACF) hoặc độ sâu 1 - (1 - D(tau)) = D(tau) (AMDF).
  * Chi phí chuyển tiếp (Transition cost): Phạt bước nhảy tần số log2 (bán âm) và phạt nặng bước nhảy quãng tám (0.8 - 1.2 octave).
"""
from typing import Any, Dict, List, Optional
import numpy as np

from .base import BasePlugin
from src.core.audio import frame_signal
from src.core.acf import compute_acf
from src.core.amdf import compute_amdf


class ViterbiTrackingPlugin(BasePlugin):
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
        lag_max = min(len(frame) - 1, int(round(sample_rate / self.f0_min)))

        if lag_min >= lag_max or lag_max >= len(r):
            return []

        peaks = []
        for lag in range(lag_min, lag_max + 1):
            left = r[lag - 1] if lag > 0 else -np.inf
            right = r[lag + 1] if lag + 1 < len(r) else -np.inf
            if r[lag] >= left and r[lag] >= right:
                peaks.append((lag, float(r[lag])))

        peaks.sort(key=lambda x: x[1], reverse=True)
        candidates = []
        for lag, val in peaks[: self.max_candidates]:
            f0 = float(sample_rate / lag) if lag > 0 else 0.0
            candidates.append({"f0": f0, "lag": lag, "metric": val})
        return candidates

    def _extract_amdf_candidates(
        self, frame: np.ndarray, sample_rate: int
    ) -> List[Dict[str, float]]:
        d = compute_amdf(frame, mode="normalized")
        lag_min = max(1, int(round(sample_rate / self.f0_max)))
        lag_max = min(len(frame) - 1, int(round(sample_rate / self.f0_min)))

        if lag_min >= lag_max or lag_max >= len(d):
            return []

        dips = []
        for lag in range(lag_min, lag_max + 1):
            left = d[lag - 1] if lag > 0 else np.inf
            right = d[lag + 1] if lag + 1 < len(d) else np.inf
            if d[lag] <= left and d[lag] <= right:
                dips.append((lag, float(d[lag])))

        dips.sort(key=lambda x: x[1])
        candidates = []
        for lag, val in dips[: self.max_candidates]:
            f0 = float(sample_rate / lag) if lag > 0 else 0.0
            candidates.append({"f0": f0, "lag": lag, "metric": val})
        return candidates

    def _viterbi_segment(
        self, cands_list: List[List[Dict[str, float]]], method: str
    ) -> List[float]:
        m = len(cands_list)
        if m == 0:
            return []
        if m == 1:
            return [cands_list[0][0]["f0"]]

        V = []
        bp = []

        # Khởi tạo t = 0
        if method == "acf":
            v0 = [1.0 - c["metric"] for c in cands_list[0]]
        else:
            v0 = [c["metric"] for c in cands_list[0]]
        V.append(v0)
        bp.append([-1] * len(v0))

        # Lan truyền tiến qua lưới Trellis
        for t in range(1, m):
            vt = []
            bpt = []
            curr_cands = cands_list[t]
            prev_cands = cands_list[t - 1]

            for j, c_j in enumerate(curr_cands):
                local_cost = (1.0 - c_j["metric"]) if method == "acf" else c_j["metric"]
                best_cost = np.inf
                best_i = 0

                for i, c_i in enumerate(prev_cands):
                    # Chênh lệch cao độ theo thang Octave (Logarithm cơ số 2)
                    if c_i["f0"] > 0 and c_j["f0"] > 0:
                        oct_diff = abs(np.log2(c_j["f0"]) - np.log2(c_i["f0"]))
                    else:
                        oct_diff = 1.0

                    trans_cost = self.w_freq * (oct_diff ** 2)
                    # Phạt đặc biệt nếu rơi vào vùng nhảy quãng tám (Pitch Doubling/Halving)
                    if 0.8 <= oct_diff <= 1.2:
                        trans_cost += self.w_octave

                    total = V[t - 1][i] + trans_cost + local_cost
                    if total < best_cost:
                        best_cost = total
                        best_i = i

                vt.append(best_cost)
                bpt.append(best_i)

            V.append(vt)
            bp.append(bpt)

        # Truy vết ngược tìm đường đi tối ưu (Backtracking)
        best_last = int(np.argmin(V[-1]))
        path = [best_last]
        for t in range(m - 1, 0, -1):
            prev = bp[t][path[-1]]
            path.append(prev)
        path.reverse()

        return [cands_list[t][path[t]]["f0"] for t in range(m)]

    def post_process_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Viterbi dynamic programming to all continuous voiced segments."""
        labels = result.get("labels", [])
        num_frames = len(labels)
        if num_frames == 0:
            return result

        sample_rate = result["sample_rate"]
        method = result.get("method", "acf").lower()
        processed_signal = result.get("processed_signal", result.get("signal"))

        # Phân khung tín hiệu đã qua tiền xử lý
        frames, _ = frame_signal(
            processed_signal,
            sample_rate,
            frame_duration_ms=25.0,
            hop_duration_ms=10.0,
            window="rectangular",
        )

        # Cắt gọt center clipping nếu có plugin áp dụng (kiểm tra từ applied_plugins)
        applied = result.get("applied_plugins", [])
        has_clipping = any("CenterClipping" in p or "Clipping" in p for p in applied)
        if has_clipping:
            from .center_clipping import CenterClippingPlugin
            clip_p = CenterClippingPlugin(clipping_ratio=0.40)
            eval_frames = [clip_p.pre_process_frame(fr, sample_rate) for fr in frames]
        else:
            eval_frames = frames

        f0_vit = np.copy(result["f0_contour"])

        # Xác định tất cả các phân đoạn Voiced liên tục
        in_v = False
        v_start = 0
        segments = []
        for i, lab in enumerate(labels):
            if lab == "v" and not in_v:
                in_v = True
                v_start = i
            elif lab != "v" and in_v:
                in_v = False
                segments.append((v_start, i))
        if in_v:
            segments.append((v_start, num_frames))

        # Tối ưu hóa Viterbi trên từng phân đoạn Voiced
        for start, end in segments:
            seg_cands = []
            for i in range(start, min(end, len(eval_frames))):
                if method == "amdf":
                    cands = self._extract_amdf_candidates(eval_frames[i], sample_rate)
                else:
                    cands = self._extract_acf_candidates(eval_frames[i], sample_rate)

                if not cands:
                    fallback_f0 = result["f0_raw"][i] if i < len(result["f0_raw"]) else 0.0
                    cands = [{"f0": fallback_f0, "lag": 0, "metric": 0.5}]
                seg_cands.append(cands)

            smoothed_f0 = self._viterbi_segment(seg_cands, method=method)
            for idx, f_val in enumerate(smoothed_f0):
                if start + idx < num_frames:
                    f0_vit[start + idx] = f_val

        # Cập nhật kết quả đầu ra
        result["f0_contour"] = f0_vit
        v_f0 = f0_vit[f0_vit > 0]
        result["f0_mean"] = float(np.mean(v_f0)) if len(v_f0) > 0 else 0.0
        result["f0_std"] = float(np.std(v_f0)) if len(v_f0) > 0 else 0.0

        return result
