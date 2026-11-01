"""Eight train-fitted decisions on the Champion's actual active pitch frames."""
from copy import deepcopy
import numpy as np

from src import config
from src.pipelines import build_detector, build_champion
from src.core.audio import load_wav
from src.core.processing import voiced_segments
from src.analysis.unsupervised_threshold import compute_all_unsupervised_thresholds
from src.analysis.advanced_unsupervised import (
    compute_gmm_em_1d, cluster_multifeature_2d, train_and_decode_2state_hmm,
    compute_zcr, compute_sliding_window_adaptive_threshold,
)
from src.analysis.protocols import threshold_labels, predict_frozen_cluster, predict_frozen_hmm

PRIMARY = ('median', 'otsu', 'valley', 'kmeans', 'gmm', 'kmeans2d', 'hmm', 'dynamic')
ADAPTIVE = ('kmeans2d_input_active', 'hmm_input_active')


def experiment_configuration():
    native = {}
    for method in config.CALIBRATED_CHAMPIONS:
        detector = build_champion(method)
        native[method] = {'pre': detector.pre_processing_configuration(), 'post': [
            {'name': plugin.name, 'parameters': {key: value for key, value in vars(plugin).items()
             if key not in ('name', 'stage') and not key.startswith('_') and isinstance(value, (int, float, str, bool))}}
            for plugin in detector.post_processors]}
    return {
        'champions': deepcopy(config.CALIBRATED_CHAMPIONS),
        'native_plugins': native,
        'fit_population': 'train_active_after_champion_preprocessing',
        'zcr_source': 'DC-centered pitch_frames after bandpass/center clipping',
        'f0_min': config.F0_MIN, 'f0_max': config.F0_MAX,
        'ste_silence_ratio': config.STE_SILENCE_RATIO,
        'median_size': config.MEDIAN_SIZE,
        'bandpass_low': config.BANDPASS_LOW, 'bandpass_high': config.BANDPASS_HIGH,
        'energy_threshold_discount': config.ENERGY_THRESHOLD_DISCOUNT,
        'energy_periodicity_guard': 'fixed_champion_threshold_for_every_strategy',
        'viterbi_w_freq': config.VITERBI_W_FREQ, 'viterbi_w_octave': config.VITERBI_W_OCTAVE,
        'dynamic_window_frames': 50, 'dynamic_alpha': .30,
        'dynamic_context': 'within active segments; STE normalized within each segment; offline window',
        'primary_strategies': list(PRIMARY), 'input_adaptive_controls': list(ADAPTIVE),
    }


def feature_spec(method, settings):
    champion = settings['champions'][method]
    return {'method': method, 'plugins': [p for p in champion['plugins'] if p in ('BP', 'Clip')],
            **{key: champion[key] for key in ('frame_duration_ms', 'hop_duration_ms', 'clipping_ratio')},
            **{key: settings[key] for key in ('f0_min', 'f0_max', 'ste_silence_ratio', 'bandpass_low', 'bandpass_high')}}


def extract_signal_features(signal, sample_rate, method, settings, wav_path=None):
    spec = feature_spec(method, settings)
    detector = build_detector(method, plugins=spec['plugins'], threshold=settings['champions'][method]['threshold'],
                              frame_duration_ms=spec['frame_duration_ms'], hop_duration_ms=spec['hop_duration_ms'],
                              clipping_ratio=spec['clipping_ratio'])
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
    fd = detector.process_signal(signal, sample_rate, wav_path=wav_path)
    fd['feature_pre_processors'] = [p.name for p in detector.pre_processors]
    fd['feature_spec'] = spec
    fd['is_active'] = fd['ste'] >= fd['ste_thresh']
    fd['periodicity_vals'] = fd['peak_values']
    fd['zcr_vals'] = np.zeros(fd['num_frames'])
    for index in np.flatnonzero(fd['is_active']):
        # The pitch extractor itself centers the clipped frame before analysis.
        frame = fd['pitch_frames'][index]
        fd['zcr_vals'][index] = compute_zcr(frame - np.mean(frame))
    return fd


def extract_file_features(path, method, settings):
    rate, signal, _ = load_wav(str(path))
    return extract_signal_features(signal, rate, method, settings, wav_path=str(path))


def fit_champion_models(features, method, settings):
    """No LAB statistics or test inputs are accepted by this fitting API."""
    expected = feature_spec(method, settings)
    if not features or any(fd['feature_spec'] != expected for fd in features):
        raise ValueError('Training features do not match the Champion feature space')
    p = np.concatenate([fd['periodicity_vals'][fd['is_active']] for fd in features])
    z = np.concatenate([fd['zcr_vals'][fd['is_active']] for fd in features])
    if len(p) < 4:
        raise ValueError('At least four active train frames are required')
    thresholds = compute_all_unsupervised_thresholds(p)
    thresholds['gmm'] = compute_gmm_em_1d(p)
    _, cluster = cluster_multifeature_2d(p, z, method)
    # An inactive spacer prevents learning transitions between recordings.
    timeline = np.concatenate([np.r_[fd['periodicity_vals'], 0.] for fd in features])
    active = np.concatenate([np.r_[fd['is_active'], False] for fd in features])
    _, hmm = train_and_decode_2state_hmm(timeline, method, active_mask=active)
    return {'method': method, 'feature_spec': expected, 'thresholds': thresholds,
            'cluster_2d': cluster, 'hmm': hmm, 'num_train_active': len(p),
            'num_train_frames': sum(len(fd['is_active']) for fd in features)}


def predict_decisions(fd, model, settings, include_input_adaptation=False):
    if fd['feature_spec'] != model['feature_spec']:
        raise ValueError('Prediction features differ from frozen training feature space')
    method, fits = model['method'], model['thresholds']
    decisions, thresholds = {}, {}
    for name in ('median', 'otsu', 'valley', 'kmeans', 'gmm'):
        threshold = fits['gmm']['threshold'] if name == 'gmm' else fits[name]
        decisions[name] = threshold_labels(fd, method, threshold)
        thresholds[name] = np.full(len(fd['is_active']), threshold)
    decisions['kmeans2d'] = predict_frozen_cluster(fd, model['cluster_2d'])
    decisions['hmm'] = predict_frozen_hmm(fd, model['hmm'])
    dynamic = np.full(len(fd['is_active']), fits['gmm']['threshold'])
    for start, end in voiced_segments(fd['is_active']):
        dynamic[start:end] = compute_sliding_window_adaptive_threshold(
            fd['periodicity_vals'][start:end], fd['ste'][start:end], fits['gmm']['threshold'],
            window_frames=settings['dynamic_window_frames'], alpha=settings['dynamic_alpha'])
    decisions['dynamic'] = threshold_labels(fd, method, dynamic)
    thresholds['dynamic'] = dynamic
    input_models = {}
    if include_input_adaptation:
        active = fd['is_active']
        labels = np.where(active, 'uv', 'sil').astype(object)
        cluster = {'status': 'empty'}
        if np.any(active):
            labels[active], cluster = cluster_multifeature_2d(
                fd['periodicity_vals'][active], fd['zcr_vals'][active], method)
        decisions['kmeans2d_input_active'] = labels
        decisions['hmm_input_active'], hmm = train_and_decode_2state_hmm(
            fd['periodicity_vals'], method, active_mask=active)
        input_models = {'kmeans2d': cluster, 'hmm': hmm}
    candidate = fd['candidate_f0']
    valid = np.isfinite(candidate) & (candidate >= settings['f0_min']) & (candidate <= settings['f0_max'])
    for labels in decisions.values():
        labels[~fd['is_active']] = 'sil'
        labels[(labels == 'v') & ~valid] = 'uv'
    return decisions, thresholds, input_models
