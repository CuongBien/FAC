"""One framing/metric loop with explicit native and historical policies."""
import numpy as np

from src.core.audio import frame_signal, compute_ste
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.core.processing import (
    center_frame, prepare_analysis_signal, silence_energy_threshold,
    validate_pitch_range, smooth_voiced_segments, compute_zcr,
)
from .contracts import FeatureSpec, FrameFeatures, plugin_configuration
from .cache import cached_call


def pitch_measurement(frame, sample_rate, spec, *, extractors=None, cache=None):
    functions = extractors if extractors is not None else {
        'acf': find_f0_acf, 'amdf': find_f0_amdf, 'cepstrum': find_f0_cepstrum}
    options = {'f0_min': spec.f0_min, 'f0_max': spec.f0_max,
               'mode': 'peak' if spec.method == 'cepstrum' else 'normalized'}
    if spec.method == 'cepstrum':
        options.update(window='hamming', n_fft=2048)
    return cached_call(cache, functions[spec.method], frame, sample_rate, **options)


def extract_signal(signal, sample_rate, spec: FeatureSpec, *, wav_path=None, duration=None,
                   pre_processors=(), decision_modifiers=(), threshold=None,
                   use_median_filter=True, median_size=3, extractors=None, cache=None,
                   reset_plugins=()):
    """Extract actual frames; keep decision hooks in frame order when supplied.

    Native input removes file DC, rejects short signals and skips pitch/frame
    plugins on silence. Historical profiles retain padding and measure every
    raw frame. Gate energy always precedes signal plugins.
    The detector supplies its reset list; reset follows input validation.
    """
    validate_pitch_range(sample_rate, spec.f0_min, spec.f0_max)
    if plugin_configuration(pre_processors) != spec.to_dict()['pre_processing']:
        raise ValueError('Actual pre-processing differs from the feature contract')
    analysis_signal = (prepare_analysis_signal(signal, sample_rate, spec.frame_duration_ms)
                       if spec.native else signal)
    for plugin in reset_plugins:
        plugin.reset()
    processed = np.copy(analysis_signal)
    for plugin in pre_processors:
        processed = plugin.pre_process_signal(processed, sample_rate)
    frames, times = frame_signal(processed, sample_rate, spec.frame_duration_ms,
                                 spec.hop_duration_ms, window='rectangular')
    ste, _ = compute_ste(analysis_signal, sample_rate, spec.frame_duration_ms, spec.hop_duration_ms)
    energy_threshold = silence_energy_threshold(ste, spec.ste_silence_ratio)
    active = ste >= energy_threshold
    count = len(frames)
    candidates, metrics, zcr, raw = (np.zeros(count, dtype=np.float64) for _ in range(4))
    lags = np.zeros(count, dtype=int)
    labels = np.full(count, 'sil', dtype=object)
    pitch_frames = np.copy(frames)
    for index, frame in enumerate(frames):
        context = {'frame_idx': index, 'is_silence': not active[index], 'ste': ste[index],
                   'sample_rate': sample_rate, 'frame': frame, 'method': spec.method}
        if not active[index] and spec.native:
            for plugin in decision_modifiers:
                plugin.adjust_frame_decision(index, 0., 0., False, context)
            continue
        pitch_frame = center_frame(frame) if spec.center_pitch_frame else frame
        for plugin in pre_processors:
            pitch_frame = plugin.pre_process_frame(pitch_frame, sample_rate)
        pitch_frames[index] = pitch_frame
        f0, metric, lag = pitch_measurement(pitch_frame, sample_rate, spec,
                                            extractors=extractors, cache=cache)
        candidates[index], metrics[index], lags[index] = f0, metric, lag
        # Script 16 centers before metric extraction; its ZCR uses that frame
        # directly. Champion centers the final pitch frame again for ZCR.
        zcr_frame = pitch_frame if spec.profile == 'centered_all' else center_frame(pitch_frame)
        zcr[index] = compute_zcr(zcr_frame)
        if threshold is None or not active[index]:
            continue
        voiced = metric <= threshold if spec.method == 'amdf' else metric >= threshold
        for plugin in decision_modifiers:
            voiced, f0 = plugin.adjust_frame_decision(index, metric, f0, voiced, context)
        valid = (f0 > 0 if spec.profile == 'baseline'
                 else np.isfinite(f0) and spec.f0_min <= f0 <= spec.f0_max)
        labels[index] = 'v' if voiced and valid else 'uv'
        raw[index] = f0 if labels[index] == 'v' else 0.
    contour = smooth_voiced_segments(raw, labels, median_size) if use_median_filter else raw.copy()
    voiced = contour[contour > 0]
    data = {
        'wav_path': wav_path, 'sample_rate': sample_rate, 'signal': signal,
        'processed_signal': processed, 'frames': frames, 'pitch_frames': pitch_frames,
        'frame_duration_ms': spec.frame_duration_ms, 'hop_duration_ms': spec.hop_duration_ms,
        'f0_min': spec.f0_min, 'f0_max': spec.f0_max, 'ste_silence_ratio': spec.ste_silence_ratio,
        'use_median_filter': use_median_filter, 'median_size': median_size,
        'duration': len(signal)/sample_rate if duration is None else duration,
        'frame_times': times, 'f0_contour': contour, 'f0_raw': raw,
        'candidate_f0': candidates, 'labels': labels, 'peak_values': metrics,
        'periodicity_vals': metrics, 'zcr_vals': zcr, 'lags': lags,
        'ste': ste, 'ste_thresh': energy_threshold, 'is_active': active,
        'threshold': threshold, 'num_frames': count, 'method': spec.method,
        'f0_mean': float(np.mean(voiced)) if len(voiced) else 0.,
        'f0_std': float(np.std(voiced)) if len(voiced) else 0.,
        'num_voiced': int(np.sum(labels == 'v')), 'num_unvoiced': int(np.sum(labels == 'uv')),
        'num_silence': int(np.sum(labels == 'sil')),
    }
    result = FrameFeatures(spec, data).to_dict()
    if threshold is None:
        # Feature-only extraction must not advertise an all-silence prediction.
        for key in ('labels', 'f0_contour', 'f0_raw', 'threshold', 'f0_mean', 'f0_std',
                    'num_voiced', 'num_unvoiced', 'num_silence', 'use_median_filter', 'median_size'):
            result.pop(key)
    return result
