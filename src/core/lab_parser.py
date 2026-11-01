"""Module for parsing ground-truth *.lab files."""
import re
from typing import Dict, List, Tuple
import numpy as np


_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_STATISTIC = re.compile(r"^(F0mean|F0std|F0num)\b(.*)$", re.IGNORECASE)


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
    statistics = {"f0mean": None, "f0std": None, "f0num": None}

    # utf-8-sig accepts both ordinary UTF-8 and a leading Windows UTF-8 BOM.
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            statistic = _STATISTIC.fullmatch(line)
            if statistic:
                name, remainder = statistic.groups()
                value_text = remainder.partition("#")[0].strip()
                if not remainder or not remainder[0].isspace() or not _NUMBER.fullmatch(value_text):
                    raise ValueError(f"{filepath}:{line_number}: Invalid {name} numeric value")
                value = float(value_text)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{filepath}:{line_number}: {name} must be finite and non-negative")
                if name.lower() == "f0num" and not value.is_integer():
                    raise ValueError(f"{filepath}:{line_number}: F0num must be an integer count")
                statistics[name.lower()] = value
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
        "f0_mean": statistics["f0mean"],
        "f0_std": statistics["f0std"],
        "f0_num": statistics["f0num"],
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
