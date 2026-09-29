"""Module for parsing ground-truth *.lab files."""
import re
from typing import Dict, List, Tuple
import numpy as np


def parse_lab_file(filepath: str) -> Dict:
    """Parse a .lab file containing segmentation intervals and ground truth F0 stats.

    Args:
        filepath: Path to .lab file.

    Returns:
        dict: {
            'segments': [(start_sec, end_sec, label), ...],
            'f0_mean': float or None,
            'f0_std': float or None
        }
    """
    segments: List[Tuple[float, float, str]] = []
    f0_mean = None
    f0_std = None
    f0_num = None

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Check for F0mean
            match_mean = re.match(r"^F0mean\s+([0-9.]+)", line, re.IGNORECASE)
            if match_mean:
                f0_mean = float(match_mean.group(1))
                continue

            # Check for F0std
            match_std = re.match(r"^F0std\s+([0-9.]+)", line, re.IGNORECASE)
            if match_std:
                f0_std = float(match_std.group(1))
                continue

            # Check for F0num
            match_num = re.match(r"^F0num\s+([0-9.]+)", line, re.IGNORECASE)
            if match_num:
                f0_num = float(match_num.group(1))
                continue

            # Check for segment line: start_sec end_sec label
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start_sec = float(parts[0])
                    end_sec = float(parts[1])
                    label = parts[2].lower()
                    segments.append((start_sec, end_sec, label))
                except ValueError:
                    pass

    return {
        "segments": segments,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_num": f0_num,
    }


def get_frame_labels(
    frame_times: np.ndarray, segments: List[Tuple[float, float, str]]
) -> np.ndarray:
    """Assign ground-truth label ('sil', 'v', 'uv') to each frame based on its center time.

    Args:
        frame_times: 1D array of frame center times (seconds).
        segments: List of tuples (start_sec, end_sec, label).

    Returns:
        np.ndarray: 1D array of string labels for each frame ('sil', 'v', 'uv').
    """
    labels = np.full(len(frame_times), "sil", dtype=object)

    for i, t in enumerate(frame_times):
        for start, end, lbl in segments:
            if start <= t < end:
                labels[i] = lbl
                break

    return labels
