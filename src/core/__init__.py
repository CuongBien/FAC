"""Core speech processing modules: audio I/O, framing, lab parser, ACF, AMDF, and pitch detection."""
from .audio import load_wav, frame_signal, compute_ste, add_awgn_noise
from .lab_parser import parse_lab_file, get_frame_labels
from .acf import compute_acf, find_f0_acf
from .amdf import compute_amdf, find_f0_amdf
from .pitch_detector import PitchDetector

__all__ = [
    "load_wav",
    "frame_signal",
    "compute_ste",
    "add_awgn_noise",
    "parse_lab_file",
    "get_frame_labels",
    "compute_acf",
    "find_f0_acf",
    "compute_amdf",
    "find_f0_amdf",
    "PitchDetector",
]
