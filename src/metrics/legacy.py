"""The existing per-file legacy-v1 metric formula, including all aliases."""
from typing import Dict, Optional
import numpy as np
from .numeric import round_metric, mean_metric


def compute_error_metrics(
    pred_mean: float,
    pred_std: float,
    ref_mean: float,
    ref_std: float,
    pred_num: Optional[float] = None,
    ref_num: Optional[float] = None,
) -> Dict:
    """Preserve absolute error at zero references; undefined MAPE is None.

    Final Score always averages THREE valid components. Missing/zero references
    never produce zero error or a two-component score.
    """
    def errors(prediction, reference):
        if prediction is None or reference is None:
            return None, None
        if not np.isfinite(prediction) or not np.isfinite(reference):
            return None, None
        absolute = abs(prediction - reference)
        return round_metric(absolute), (absolute / reference * 100.0 if reference > 0 else None)

    abs_mean, mape_mean = errors(pred_mean, ref_mean)
    abs_std, mape_std = errors(pred_std, ref_std)
    abs_num, mape_num = errors(pred_num, ref_num)
    score = mean_metric([mape_mean, mape_std, mape_num])
    return {
        "pred_f0_mean": round_metric(pred_mean), "ref_f0_mean": round_metric(ref_mean),
        "abs_error_mean": abs_mean, "rel_error_mean_pct": round_metric(mape_mean),
        "mape_f0": round_metric(mape_mean), "mape_f0_mean": round_metric(mape_mean),
        "mape_mean": round_metric(mape_mean),
        "pred_f0_std": round_metric(pred_std), "ref_f0_std": round_metric(ref_std),
        "abs_error_std": abs_std, "rel_error_std_pct": round_metric(mape_std),
        "mape_f0_std": round_metric(mape_std), "mape_std": round_metric(mape_std),
        "pred_f0_num": int(round(pred_num)) if round_metric(pred_num) is not None else None,
        "ref_f0_num": int(round(ref_num)) if round_metric(ref_num) is not None else None,
        "abs_error_num": int(round(abs_num)) if abs_num is not None else None,
        "mape_f0_num": round_metric(mape_num), "mape_num": round_metric(mape_num),
        "composite_score": score, "final_score": score,
    }
