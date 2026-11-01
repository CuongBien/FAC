"""Eight train-fitted decisions on the Champion's actual active pitch frames."""
from copy import deepcopy
import numpy as np

from src import config
from src.pipelines import build_champion
from src.core.processing import voiced_segments
from src.pipeline.contracts import training_contract
# Compatibility exports; extraction is owned by the pipeline package.
from src.pipeline.champion import (
    feature_spec, feature_contract, extract_signal_features, extract_file_features,
)
from src.analysis.unsupervised_threshold import compute_all_unsupervised_thresholds
from src.analysis.advanced_unsupervised import (
    compute_gmm_em_1d, cluster_multifeature_2d, train_and_decode_2state_hmm,
    compute_sliding_window_adaptive_threshold,
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



def fit_champion_models(features, method, settings):
    """No LAB statistics or test inputs are accepted by this fitting API."""
    expected = feature_spec(method, settings)
    if not features or any(fd['feature_spec'] != expected for fd in features):
        raise ValueError('Training features do not match the Champion feature space')
    contract = training_contract(features, method)
    if contract != feature_contract(method, settings):
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
    return {'method': method, 'feature_spec': expected, 'feature_contract': contract.to_dict(),
            'fit_population': 'train_active', 'thresholds': thresholds,
            'cluster_2d': cluster, 'hmm': hmm, 'num_train_active': len(p),
            'num_train_frames': sum(len(fd['is_active']) for fd in features)}


def predict_decisions(fd, model, settings, include_input_adaptation=False):
    if (fd['feature_spec'] != model['feature_spec']
            or feature_spec(model['method'], settings) != model['feature_spec']):
        raise ValueError('Prediction features differ from frozen training feature space')
    # Old Champion models carry the narrower legacy spec; expand it using the
    # frozen experiment settings, never today's configuration defaults.
    contract = feature_contract(model['method'], settings).to_dict()
    if (fd.get('feature_contract') != contract
            or model.get('feature_contract', contract) != contract):
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
