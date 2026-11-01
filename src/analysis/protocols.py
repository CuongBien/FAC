"""Explicit train-fitted and per-input adaptive unsupervised protocols."""
import numpy as np
from scipy.stats import norm

from src import config
from src.core.processing import smooth_voiced_segments, voiced_segments
from src.pipeline.contracts import FeatureSpec, training_contract, PredictionResult
from src.analysis.unsupervised_threshold import compute_all_unsupervised_thresholds
from src.analysis.advanced_unsupervised import (
    compute_gmm_em_1d, cluster_multifeature_2d, train_and_decode_2state_hmm,
    compute_sliding_window_adaptive_threshold, _hmm_viterbi,
)


def fit_models(features, method, *, median_size=3, dynamic_window_frames=50, dynamic_alpha=.30):
    """Fit from train features only; never consume LAB statistics or test data."""
    spec = training_contract(features, method)
    all_p = np.concatenate([fd['periodicity_vals'] for fd in features])
    active_p = np.concatenate([fd['periodicity_vals'][fd['is_active']] for fd in features])
    active_z = np.concatenate([fd['zcr_vals'][fd['is_active']] for fd in features])
    if len(active_p) < 4:
        raise ValueError('At least four active training frames are required')
    thresholds = {}
    for population, values in [('all', all_p), ('active', active_p)]:
        fit = compute_all_unsupervised_thresholds(values)
        fit['gmm'] = compute_gmm_em_1d(values)
        thresholds[population] = fit
    _, cluster = cluster_multifeature_2d(active_p, active_z, method)
    # An inactive spacer prevents transitions across different recording files.
    timeline = np.concatenate([np.r_[fd['periodicity_vals'], 0.] for fd in features])
    active = np.concatenate([np.r_[fd['is_active'], False] for fd in features])
    _, hmm = train_and_decode_2state_hmm(timeline, method, active_mask=active)
    return {'method': method, 'thresholds': thresholds, 'cluster_2d': cluster, 'hmm': hmm,
            'num_train_frames': len(all_p), 'num_train_active': len(active_p),
            'feature_contract': spec.to_dict(), 'fit_populations': ['train_all', 'train_active'],
            'frame_duration_ms': spec.frame_duration_ms, 'hop_duration_ms': spec.hop_duration_ms,
            'f0_min': spec.f0_min, 'f0_max': spec.f0_max, 'median_size': median_size,
            'dynamic_window_frames': dynamic_window_frames, 'dynamic_alpha': dynamic_alpha}


def threshold_labels(fd, method, threshold):
    values = fd['periodicity_vals']
    active = fd['is_active']
    voiced = values <= threshold if method == 'amdf' else values >= threshold
    return np.where(active, np.where(voiced, 'v', 'uv'), 'sil').astype(object)


def predict_frozen_cluster(fd, model):
    labels = np.where(fd['is_active'], 'uv', 'sil').astype(object)
    if model['status'] != 'ok':
        return labels
    active = fd['is_active']
    features = np.column_stack([
        (fd['periodicity_vals'][active]-model['p_mean'])/model['p_std'],
        (fd['zcr_vals'][active]-model['z_mean'])/model['z_std'],
    ])
    dv = np.sum((features-np.asarray(model['centroid_v_normalized']))**2, axis=1)
    du = np.sum((features-np.asarray(model['centroid_uv_normalized']))**2, axis=1)
    labels[active] = np.where((dv < du) & ~np.isclose(dv, du, rtol=1e-12, atol=1e-12), 'v', 'uv')
    return labels


def predict_frozen_hmm(fd, model):
    active = fd['is_active']
    labels = np.where(active, 'uv', 'sil').astype(object)
    if model['status'] != 'ok' or not np.any(active):
        return labels
    log_emissions = norm.logpdf(fd['periodicity_vals'][:, None],
                                loc=np.asarray(model['means']), scale=np.sqrt(model['variances']))
    path = _hmm_viterbi(log_emissions, np.asarray(model['A']), np.asarray(model['pi']),
                        list(voiced_segments(active)))
    labels[active] = np.where(path[active] == model['v_state'], 'v', 'uv')
    return labels


def predict_strategies(fd, models):
    """All strategies use the same frames, silence gate, candidates and median."""
    method = models['method']
    # Legacy Stage 4 records its historical 25/10 framing but has no v1 contract.
    # The adapter is explicit; new models always carry the full physical space.
    spec = (FeatureSpec.from_dict(models['feature_contract']) if 'feature_contract' in models
            else FeatureSpec(method, models['frame_duration_ms'], models['hop_duration_ms'],
                             70., 400., .008, 'centered_all'))
    if spec.method != method or fd.get('feature_contract') != spec.to_dict():
        raise ValueError('Prediction features differ from frozen training feature space')
    labels = {'historical_static': threshold_labels(fd, method, config.baseline_threshold(method))}
    for population, values in models['thresholds'].items():
        for name in ('median', 'otsu', 'valley', 'kmeans', 'gmm'):
            threshold = values['gmm']['threshold'] if name == 'gmm' else values[name]
            labels[f'{name}_train_{population}'] = threshold_labels(fd, method, threshold)
        dynamic = compute_sliding_window_adaptive_threshold(
            fd['periodicity_vals'], fd['ste'], values['gmm']['threshold'],
            window_frames=models['dynamic_window_frames'], alpha=models['dynamic_alpha'])
        labels[f'dynamic_train_{population}'] = threshold_labels(fd, method, dynamic)
    labels['kmeans2d_train_active'] = predict_frozen_cluster(fd, models['cluster_2d'])
    labels['hmm_train_active'] = predict_frozen_hmm(fd, models['hmm'])
    active = fd['is_active']
    adaptive = np.where(active, 'uv', 'sil').astype(object)
    if np.any(active):
        adaptive[active], _ = cluster_multifeature_2d(
            fd['periodicity_vals'][active], fd['zcr_vals'][active], method)
    labels['kmeans2d_input_active'] = adaptive
    labels['hmm_input_active'], _ = train_and_decode_2state_hmm(
        fd['periodicity_vals'], method, active_mask=active)
    valid = (np.isfinite(fd['candidate_f0']) & (fd['candidate_f0'] >= spec.f0_min)
             & (fd['candidate_f0'] <= spec.f0_max))
    for prediction in labels.values():
        prediction[(prediction == 'v') & ~valid] = 'uv'
    return labels


def result_from_labels(fd, labels, median_size=3):
    labels = np.asarray(labels, dtype=object).copy()
    contour = np.where(labels == 'v', fd['candidate_f0'], 0.)
    contour = smooth_voiced_segments(contour, labels, median_size)
    voiced = contour[labels == 'v']
    result = {**fd, 'labels': labels, 'f0_contour': contour,
            'f0_mean': float(np.mean(voiced)) if len(voiced) else 0.,
            'f0_std': float(np.std(voiced)) if len(voiced) else 0.,
            'num_voiced': len(voiced), 'num_frames': len(labels)}
    spec = FeatureSpec.from_dict(fd['feature_contract'])
    return PredictionResult(result, spec.f0_min, spec.f0_max).to_dict()
