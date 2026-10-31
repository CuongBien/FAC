"""Historical baselines and explicit train-selected Champion presets."""
from src.calibrated_presets import CHAMPIONS as CALIBRATED_CHAMPIONS

FRAME_DURATION_MS = 25.0
HOP_DURATION_MS = 10.0
F0_MIN = 70.0
F0_MAX = 400.0
STE_SILENCE_RATIO = 0.008
MEDIAN_SIZE = 3
BASELINE_THRESHOLDS = {"acf": 0.4408, "amdf": 0.4380, "cepstrum": 0.0600}
PLUGIN_THRESHOLDS = {
    "acf": {(False, False): 0.4408, (True, False): 0.4892,
            (False, True): 0.3352, (True, True): 0.3953},
    "amdf": {(False, False): 0.4380, (True, False): 0.3734,
             (False, True): 0.6488, (True, True): 0.5628},
    "cepstrum": {(False, False): 0.0600, (True, False): 0.0500,
                 (False, True): 0.0600, (True, True): 0.0500},
}
HISTORICAL_CHAMPION_PLUGINS = {
    "acf": ("BP", "Energy", "Viterbi"),
    "amdf": ("Energy", "Viterbi"),
    "cepstrum": ("BP", "Clip", "Energy", "Viterbi"),
}
CHAMPION_PLUGINS = {method: tuple(spec['plugins']) for method, spec in CALIBRATED_CHAMPIONS.items()}
BANDPASS_LOW = 70.0
BANDPASS_HIGH = 900.0
CLIPPING_RATIO = 0.40
ENERGY_THRESHOLD_DISCOUNT = 0.20
VITERBI_W_FREQ = 3.0
VITERBI_W_OCTAVE = 4.0
THRESHOLD_SOURCE = "historical_saved_parameters; not refit from current summary-only LAB"


def baseline_threshold(method):
    method = method.lower()
    if method not in BASELINE_THRESHOLDS:
        raise ValueError(f"Unknown pitch method: {method}")
    return BASELINE_THRESHOLDS[method]


def threshold_for_plugins(method, plugins=()):
    method = method.lower()
    baseline_threshold(method)
    names = {name.lower() for name in plugins}
    return PLUGIN_THRESHOLDS[method][("bp" in names, "clip" in names)]
