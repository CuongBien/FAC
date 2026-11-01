"""Feature APIs used by benchmark entrypoints instead of importing scripts."""
from src.core.audio import load_wav
from .contracts import FeatureSpec
from .engine import extract_signal


def extract_file_features(wav_path, method, frame_duration_ms=25., hop_duration_ms=10.,
                          f0_min=70., f0_max=400., ste_silence_ratio=.008, *,
                          profile='centered_all', cache=None):
    rate, signal, duration = load_wav(str(wav_path))
    return extract_raw_features(signal, rate, method, frame_duration_ms, hop_duration_ms,
                                f0_min, f0_max, ste_silence_ratio, profile=profile,
                                wav_path=str(wav_path), duration=duration, cache=cache)


def extract_raw_features(signal, sample_rate, method, frame_duration_ms=25., hop_duration_ms=10.,
                         f0_min=70., f0_max=400., ste_silence_ratio=.008, *,
                         profile='centered_all', wav_path=None, duration=None, cache=None):
    if profile not in ('centered_all', 'raw_all'):
        raise ValueError('Historical feature extraction requires an all-frame profile')
    spec = FeatureSpec(method.lower(), frame_duration_ms, hop_duration_ms,
                       f0_min, f0_max, ste_silence_ratio, profile)
    return extract_signal(signal, sample_rate, spec, wav_path=wav_path, duration=duration,
                           use_median_filter=False, cache=cache)
