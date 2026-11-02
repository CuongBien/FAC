"""Four train-fitted decisions on aligned active pitch frames."""
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
    cluster_multifeature_2d,
    compute_sliding_window_adaptive_threshold,
)
from src.analysis.protocols import threshold_labels, predict_frozen_cluster

PRIMARY = ('otsu', 'kmeans', 'kmeans2d', 'dynamic')
ADAPTIVE = ('kmeans2d_input_active',)


def experiment_configuration():
    return {
        'champions': deepcopy(config.CALIBRATED_CHAMPIONS),
        'fit_population': 'train_active_after_champion_preprocessing',
        'zcr_source': 'DC-centered pitch frames after Champion preprocessing',
        'bandpass_low': config.BANDPASS_LOW, 'bandpass_high': config.BANDPASS_HIGH,
        'f0_min': config.F0_MIN, 'f0_max': config.F0_MAX,
        'ste_silence_ratio': config.STE_SILENCE_RATIO, 'median_size': config.MEDIAN_SIZE,
        'dynamic_base': 'otsu', 'dynamic_window_frames': 50, 'dynamic_alpha': .30,
        'dynamic_context': 'within active segments; offline centered window',
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
    _, cluster = cluster_multifeature_2d(p, z, method)
    return {'method': method, 'feature_spec': expected, 'feature_contract': contract.to_dict(),
            'fit_population': 'train_active', 'thresholds': thresholds,
            'cluster_2d': cluster, 'num_train_active': len(p),
            'num_train_frames': sum(len(fd['is_active']) for fd in features)}


def predict_decisions(fd, model, settings, include_input_adaptation=False):
    if (fd['feature_spec'] != model['feature_spec']
            or feature_spec(model['method'], settings) != model['feature_spec']):
        raise ValueError('Prediction features differ from frozen training feature space')
    contract = feature_contract(model['method'], settings).to_dict()
    if (fd.get('feature_contract') != contract
            or model.get('feature_contract', contract) != contract):
        raise ValueError('Prediction features differ from frozen training feature space')
    method, fits = model['method'], model['thresholds']
    decisions, thresholds = {}, {}
    for name in ('otsu', 'kmeans'):
        threshold = fits[name]
        decisions[name] = threshold_labels(fd, method, threshold)
        thresholds[name] = np.full(len(fd['is_active']), threshold)
    decisions['kmeans2d'] = predict_frozen_cluster(fd, model['cluster_2d'])
    dynamic = np.full(len(fd['is_active']), fits['otsu'])
    for start, end in voiced_segments(fd['is_active']):
        dynamic[start:end] = compute_sliding_window_adaptive_threshold(
            fd['periodicity_vals'][start:end], fd['ste'][start:end], fits['otsu'],
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
        input_models = {'kmeans2d': cluster}
    candidate = fd['candidate_f0']
    valid = np.isfinite(candidate) & (candidate >= settings['f0_min']) & (candidate <= settings['f0_max'])
    for labels in decisions.values():
        labels[~fd['is_active']] = 'sil'
        labels[(labels == 'v') & ~valid] = 'uv'
    return decisions, thresholds, input_models
