"""PTDB-TUG Speech and Laryngograph Dataset Loader.

Provides utilities to parse PTDB-TUG audio (.wav) and reference pitch (.f0) files,
discover available dataset pairs, and align predicted pitch contours with ground truth.

Reference:
    Pirker et al., 'A Pitch Tracking Corpus with Evaluation on Multipitch Tracking Scenario',
    Interspeech 2011.
"""

from dataclasses import dataclass
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.audio import load_wav


@dataclass
class PTDBUtterance:
    """Represents a single utterance in the PTDB-TUG corpus."""
    speaker: str
    gender: str
    sentence_id: str
    wav_path: str
    f0_path: str
    sample_rate: int
    signal: np.ndarray
    duration: float
    gt_times: np.ndarray
    gt_f0: np.ndarray
    gt_voicing: np.ndarray
    raw_f0: np.ndarray
    confidence: np.ndarray


def load_ptdb_f0(f0_path: str, hop_sec: float = 0.010) -> Dict[str, np.ndarray]:
    """Parse a PTDB-TUG .f0 reference file.

    The file contains 4 columns:
        Col 0: Reference F0 in Hz (0.0 for unvoiced frames).
        Col 1: Voicing flag (1.0 for voiced, 0.0 for unvoiced).
        Col 2: Raw candidate pitch from RAPT/EGG.
        Col 3: Normalized confidence / cross-correlation score.

    Args:
        f0_path: Path to .f0 text file.
        hop_sec: Time step between consecutive frames in seconds (default 10ms = 0.010s).

    Returns:
        Dict with keys: 'times', 'f0', 'voicing', 'raw_f0', 'confidence'.
    """
    if not os.path.exists(f0_path):
        raise FileNotFoundError(f"Reference F0 file not found: {f0_path}")

    data = np.loadtxt(f0_path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    f0 = data[:, 0]
    voicing = data[:, 1] > 0.5
    raw_f0 = data[:, 2]
    confidence = data[:, 3]

    num_frames = len(f0)
    # Frame centers: 0.5 * hop_sec, 1.5 * hop_sec, ...
    times = (np.arange(num_frames, dtype=np.float64) + 0.5) * hop_sec

    return {
        "times": times,
        "f0": f0,
        "voicing": voicing,
        "raw_f0": raw_f0,
        "confidence": confidence,
    }


def load_ptdb_utterance(wav_path: str, f0_path: Optional[str] = None) -> PTDBUtterance:
    """Load both audio and reference pitch for a single PTDB utterance.

    Args:
        wav_path: Path to mic_*.wav audio file.
        f0_path: Optional path to ref_*.f0 file. If None, inferred automatically.

    Returns:
        PTDBUtterance dataclass instance.
    """
    if f0_path is None:
        # Infer ref file path: replace 'MIC' with 'REF' and 'mic_' with 'ref_' and '.wav' with '.f0'
        dirname = os.path.dirname(wav_path)
        basename = os.path.basename(wav_path)
        ref_basename = basename.replace("mic_", "ref_").replace(".wav", ".f0")
        ref_dirname = dirname.replace("MIC", "REF").replace("mic", "ref")
        f0_path = os.path.join(ref_dirname, ref_basename)

    # Parse metadata from filename e.g. mic_F01_sa1.wav
    basename = os.path.basename(wav_path)
    match = re.match(r"(?:mic|ref)_([FM]\d{2})_([a-zA-Z0-9]+)\.(?:wav|f0)", basename)
    if match:
        speaker = match.group(1).upper()
        sentence_id = match.group(2).lower()
        gender = "female" if speaker.startswith("F") else "male"
    else:
        speaker = "UNKNOWN"
        sentence_id = "UNKNOWN"
        gender = "unknown"

    sr, signal, duration = load_wav(wav_path)
    gt = load_ptdb_f0(f0_path)

    return PTDBUtterance(
        speaker=speaker,
        gender=gender,
        sentence_id=sentence_id,
        wav_path=wav_path,
        f0_path=f0_path,
        sample_rate=sr,
        signal=signal,
        duration=duration,
        gt_times=gt["times"],
        gt_f0=gt["f0"],
        gt_voicing=gt["voicing"],
        raw_f0=gt["raw_f0"],
        confidence=gt["confidence"],
    )


def find_ptdb_dataset(root_dir: str = "data/ptdb_tug") -> List[Dict[str, str]]:
    """Scan directory recursively and return all available (wav, f0) pairs.

    Returns:
        List of dicts: [{'speaker': ..., 'gender': ..., 'sentence_id': ..., 'wav_path': ..., 'f0_path': ...}]
    """
    pairs = []
    if not os.path.isdir(root_dir):
        return pairs

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.startswith("mic_") and f.endswith(".wav"):
                wav_path = os.path.join(root, f)
                # Corresponding ref file
                ref_name = f.replace("mic_", "ref_").replace(".wav", ".f0")
                ref_dir = root.replace("MIC", "REF").replace("mic", "ref")
                f0_path = os.path.join(ref_dir, ref_name)

                if os.path.exists(f0_path):
                    m = re.match(r"mic_([FM]\d{2})_([a-zA-Z0-9]+)\.wav", f)
                    if m:
                        spk = m.group(1).upper()
                        sent = m.group(2).lower()
                        gender = "female" if spk.startswith("F") else "male"
                    else:
                        spk, sent, gender = "UNKNOWN", "UNKNOWN", "unknown"

                    pairs.append({
                        "speaker": spk,
                        "gender": gender,
                        "sentence_id": sent,
                        "wav_path": wav_path,
                        "f0_path": f0_path,
                    })

    # Sort stably by gender, speaker, sentence_id
    pairs.sort(key=lambda x: (x["gender"], x["speaker"], x["sentence_id"]))
    return pairs


def align_predictions_to_ground_truth(
    pred_times: np.ndarray,
    pred_f0: np.ndarray,
    pred_voicing: np.ndarray,
    gt_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align predicted pitch and voicing contours onto ground-truth time points.

    Uses nearest-neighbor interpolation in time to map predicted decisions to GT frames.

    Args:
        pred_times: 1D array of center times for predicted frames.
        pred_f0: 1D array of predicted F0 values.
        pred_voicing: 1D boolean array of predicted voicing flags.
        gt_times: 1D array of ground-truth frame times.

    Returns:
        tuple: (aligned_pred_f0, aligned_pred_voicing) both having length len(gt_times).
    """
    if len(pred_times) == 0:
        return np.zeros_like(gt_times), np.zeros_like(gt_times, dtype=bool)

    # Nearest-neighbor indices
    idx = np.searchsorted(pred_times, gt_times)
    idx = np.clip(idx, 0, len(pred_times) - 1)

    # For points between i-1 and i, choose the closer timestamp
    left_idx = np.clip(idx - 1, 0, len(pred_times) - 1)
    d_right = np.abs(pred_times[idx] - gt_times)
    d_left = np.abs(pred_times[left_idx] - gt_times)
    closer_idx = np.where(d_left < d_right, left_idx, idx)

    aligned_voicing = pred_voicing[closer_idx]
    aligned_f0 = pred_f0[closer_idx]
    aligned_f0[~aligned_voicing] = 0.0

    return aligned_f0, aligned_voicing
