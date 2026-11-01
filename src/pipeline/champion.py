"""Champion feature extraction, independent of model fitting and reporting.

The small legacy feature_spec mapping is retained for archived models; the
versioned FeatureSpec contract additionally records the actual extraction policy.
"""
from src.pipelines import build_detector
from src.core.audio import load_wav
from src.pipeline.contracts import FeatureSpec

def feature_spec(method, settings):
    champion = settings['champions'][method]
    return {'method': method, 'plugins': [p for p in champion['plugins'] if p in ('BP', 'Clip')],
            **{key: champion[key] for key in ('frame_duration_ms', 'hop_duration_ms', 'clipping_ratio')},
            **{key: settings[key] for key in ('f0_min', 'f0_max', 'ste_silence_ratio', 'bandpass_low', 'bandpass_high')}}


def feature_detector(method, settings, *, cache=None):
    """Build the declared pre-processing space without decision/post plugins."""
    spec = feature_spec(method, settings)
    detector = build_detector(method, plugins=spec['plugins'], threshold=settings['champions'][method]['threshold'],
                              frame_duration_ms=spec['frame_duration_ms'], hop_duration_ms=spec['hop_duration_ms'],
                              clipping_ratio=spec['clipping_ratio'], cache=cache)
    # AMDF has no pre-processing plugin, but the wrapper exposes the same native
    # frames/candidates for it. No post-processing occurs during feature fitting.
    from src.plugins import PluginPitchDetector
    if not isinstance(detector, PluginPitchDetector):
        detector = PluginPitchDetector(detector)
    detector.base_detector.f0_min = spec['f0_min']
    detector.base_detector.f0_max = spec['f0_max']
    detector.base_detector.ste_silence_ratio = spec['ste_silence_ratio']
    detector.base_detector.median_size = settings['median_size']
    for plugin in detector.pre_processors:
        if hasattr(plugin, 'low_cutoff'):
            plugin.low_cutoff = spec['bandpass_low']
            plugin.high_cutoff = spec['bandpass_high']
    return detector


def feature_contract(method, settings):
    detector = feature_detector(method, settings)
    return FeatureSpec.from_detector(detector.base_detector,
                                     detector.pre_processing_configuration(), plugin_wrapper=True)


def extract_signal_features(signal, sample_rate, method, settings, wav_path=None, *, cache=None):
    detector = feature_detector(method, settings, cache=cache)
    fd = detector.process_signal(signal, sample_rate, wav_path=wav_path)
    fd['feature_pre_processors'] = [p.name for p in detector.pre_processors]
    fd['feature_spec'] = feature_spec(method, settings)
    return fd


def extract_file_features(path, method, settings, *, cache=None):
    rate, signal, _ = load_wav(str(path))
    return extract_signal_features(signal, rate, method, settings, wav_path=str(path), cache=cache)
