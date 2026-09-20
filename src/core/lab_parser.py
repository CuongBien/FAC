"""Module for parsing ground-truth *.lab files."""
from typing import Dict, List, Tuple
import numpy as np


def parse_lab_file(filepath: str) -> Dict:
    """Parse a .lab file containing segmentation intervals and ground truth F0 stats.

    Returns:
        dict: {
            'segments': [(start_sec, end_sec, label), ...],
            'f0_mean': float,
            'f0_std': float
        }
    """
    pass


def get_frame_labels(frame_times: np.ndarray, segments: List[Tuple[float, float, str]]) -> np.ndarray:
    """Assign ground-truth label ('sil', 'v', 'uv') to each frame center time.

    Returns:
        np.ndarray: Array of strings ('sil', 'v', 'uv') for each frame.
    """
    pass
