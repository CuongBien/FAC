"""Evaluation module to compare algorithm predictions with ground truth from *.lab files."""
from typing import Dict


def compute_error_metrics(pred_mean: float, pred_std: float, ref_mean: float, ref_std: float) -> Dict:
    """Compute absolute and relative errors between predicted F0 statistics and reference *.lab statistics."""
    pass


def evaluate_against_ground_truth(test_results: Dict, lab_path: str) -> Dict:
    """Compare full test predictions with reference *.lab file."""
    pass
