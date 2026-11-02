"""Pitch-Shift consistency using each algorithm's own F0 and periodicity.

The historical ACF-only implementation remains byte-for-byte unchanged.
No model, threshold fitting, or statistics from the test population are used.
"""
import numpy as np
from scipy.signal import resample

from src.analysis.ssl_pitch_shift import analyze_signal_ssl_pitch_shift
from src.core.amdf import find_f0_amdf
from src.core.acf import find_f0_acf
from src.core.cepstrum import find_f0_cepstrum
from src.core.audio import frame_signal
from src.core.processing import center_frame, prepare_analysis_signal, silence_energy_threshold, validate_pitch_range
from src.calibrated_presets import CHAMPIONS


def method_pitch_strength(frame, sample_rate, method, f0_min=70., f0_max=400., cepstrum_scale=None):
    """Return F0, own metric, and a bounded strength; no cross-method teacher."""
    if method == 'acf':
        f0, metric, _ = find_f0_acf(frame, sample_rate, f0_min, f0_max)
        strength = np.clip(metric, 0., 1.)
    elif method == 'amdf':
        f0, metric, _ = find_f0_amdf(frame, sample_rate, f0_min, f0_max)
        strength = np.clip(1-metric, 0., 1.)
    elif method == 'cepstrum':
        scale = CHAMPIONS['cepstrum']['threshold'] if cepstrum_scale is None else cepstrum_scale
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Cepstrum reference scale must be positive and finite')
        f0, metric, _ = find_f0_cepstrum(frame, sample_rate, f0_min, f0_max, mode='peak')
        positive = max(0., metric)
        # Raw cepstral peaks are not probabilities. This fixed monotone mapping
        # sends the frozen Champion reference peak to 0.5, without test fitting.
        strength = positive/(positive+scale)
    else:
        raise ValueError('Unknown pitch estimator')
    return f0, float(metric), float(strength)


def analyze_signal_pitch_shift_by_method(signal, sample_rate, method='acf', *,
                                         frame_duration_ms=30., hop_duration_ms=10.,
                                         alpha=1.25, f0_min=70., f0_max=400.,
                                         sigma=.12, ste_ratio=.01, cepstrum_scale=None):
    """Use the same estimator on original/resampled frames and decide at 0.5."""
    method = method.lower()
    if method not in ('acf', 'amdf', 'cepstrum'):
        raise ValueError('Unknown Pitch-Shift method')
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Pitch scale and consistency width must be positive and finite')
    if method == 'acf':
        if sigma != .12:
            raise ValueError('The preserved ACF implementation uses sigma=0.12')
        return analyze_signal_ssl_pitch_shift(signal, sample_rate, frame_duration_ms,
            hop_duration_ms, alpha, f0_min, f0_max, ste_ratio)
    validate_pitch_range(sample_rate, f0_min, f0_max)
    values = prepare_analysis_signal(signal, sample_rate, frame_duration_ms)
    frames, times = frame_signal(values, sample_rate, frame_duration_ms, hop_duration_ms)
    energy = np.mean(frames**2, axis=1)
    active = energy >= silence_energy_threshold(energy, ste_ratio)
    pitch = np.zeros(len(frames)); scores = np.zeros(len(frames))
    errors = np.ones(len(frames)); metrics = np.zeros(len(frames)); strengths = np.zeros(len(frames))
    shifted_min = max(20., f0_min*alpha*.7)
    shifted_max = min(sample_rate/2-50., f0_max*alpha*1.3)
    if shifted_min >= shifted_max:
        raise ValueError('Transformed pitch range is empty')
    for index, raw in enumerate(frames):
        frame = center_frame(raw)
        if np.mean(frame**2) <= 1e-12:
            continue
        f0, metric, strength = method_pitch_strength(frame, sample_rate, method, f0_min, f0_max, cepstrum_scale)
        metrics[index], strengths[index] = metric, strength
        if f0 <= 0 or strength <= .10:
            continue
        shifted = resample(frame, max(16, int(round(len(frame)/alpha))))
        f0_shifted, _, _ = method_pitch_strength(shifted, sample_rate, method, shifted_min, shifted_max, cepstrum_scale)
        pitch[index] = f0
        if f0_shifted <= 0:
            continue
        errors[index] = abs(f0_shifted-alpha*f0)/(alpha*f0)
        scores[index] = strength*np.exp(-errors[index]**2/(2*sigma**2))
    scores[~active] = 0.
    labels = np.where((scores >= .5) & active, 'v', 'uv')
    return {'f0_contour':np.where(labels=='v',pitch,0.), 'labels':labels,
            'probabilities':scores,'active_mask':active,'rel_errors':errors,
            'candidate_f0':pitch,'periodicity_metric':metrics,'periodicity_strength':strengths,
            'frame_times':times,'method':method}
