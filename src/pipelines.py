"""Fresh detector factories shared by CLI, benchmarks, and external callers."""
from src import config
from src.core.pitch_detector import PitchDetector
from src.plugins import (
    PluginPitchDetector, BandpassFilterPlugin, CenterClippingPlugin,
    EnergyExtensionPlugin, ViterbiTrackingPlugin,
)


def build_baseline(method, *, cache=None):
    return PitchDetector(method=method, cache=cache)


def build_detector(method, plugins=(), threshold=None, frame_duration_ms=None,
                   hop_duration_ms=None, clipping_ratio=None, *, cache=None):
    """Build an explicit experiment configuration without shared plugin state."""
    method = method.lower()
    config.baseline_threshold(method)
    names = tuple(plugins)
    unknown = set(names) - {"BP", "Clip", "Energy", "Viterbi"}
    if unknown:
        raise ValueError(f"Unknown plugins: {sorted(unknown)}")
    factories = {
        "BP": lambda: BandpassFilterPlugin(config.BANDPASS_LOW, config.BANDPASS_HIGH),
        "Clip": lambda: CenterClippingPlugin(clipping_ratio=config.CLIPPING_RATIO if clipping_ratio is None else clipping_ratio),
        "Energy": lambda: EnergyExtensionPlugin(threshold_discount=config.ENERGY_THRESHOLD_DISCOUNT),
        "Viterbi": lambda: ViterbiTrackingPlugin(w_freq=config.VITERBI_W_FREQ, w_octave=config.VITERBI_W_OCTAVE, cache=cache),
    }
    base = PitchDetector(
        method=method, cache=cache, threshold=config.threshold_for_plugins(method, names) if threshold is None else threshold,
        frame_duration_ms=config.FRAME_DURATION_MS if frame_duration_ms is None else frame_duration_ms,
        hop_duration_ms=config.HOP_DURATION_MS if hop_duration_ms is None else hop_duration_ms,
    )
    return PluginPitchDetector(base_detector=base, plugins=[factories[name]() for name in names]) if names else base


def build_champion(method, *, cache=None):
    method = method.lower()
    config.baseline_threshold(method)
    return build_detector(method, cache=cache, **config.CALIBRATED_CHAMPIONS[method])
