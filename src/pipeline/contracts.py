"""Versioned physical feature space, independent of thresholds and fit data."""
from dataclasses import dataclass
import json
from typing import Mapping

import numpy as np


def plugin_configuration(plugins):
    """Record the same public scalar parameters used by native plugin results."""
    return [{'name': plugin.name, 'parameters': {
        key: value for key, value in vars(plugin).items()
        if key not in ('name', 'stage') and not key.startswith('_')
        and isinstance(value, (int, float, str, bool))
    }} for plugin in plugins]


@dataclass(frozen=True)
class FeatureSpec:
    method: str
    frame_duration_ms: float
    hop_duration_ms: float
    f0_min: float
    f0_max: float
    ste_silence_ratio: float
    profile: str
    pre_processing_json: str = '[]'

    def __post_init__(self):
        if self.method not in ('acf', 'amdf', 'cepstrum'):
            raise ValueError('Unknown pitch method')
        if self.profile not in ('baseline', 'plugins', 'raw_all', 'centered_all'):
            raise ValueError('Unknown feature profile')
        values = (self.frame_duration_ms, self.hop_duration_ms, self.f0_min,
                  self.f0_max, self.ste_silence_ratio)
        if not all(isinstance(v, (int, float, np.integer, np.floating))
                   and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) for v in values):
            raise ValueError('Feature parameters must be finite numbers')
        if not 0 < self.f0_min < self.f0_max:
            raise ValueError('Pitch range must satisfy 0 < f0_min < f0_max <= Nyquist')
        if min(values[:2]) <= 0 or self.ste_silence_ratio < 0:
            raise ValueError('Invalid feature framing or silence ratio')
        plugins = json.loads(self.pre_processing_json)
        if not isinstance(plugins, list) or (plugins and self.profile != 'plugins'):
            raise ValueError('Only the plugins profile accepts pre-processing')
        for plugin in plugins:
            if (not isinstance(plugin, dict) or set(plugin) != {'name', 'parameters'}
                    or not isinstance(plugin['name'], str) or not isinstance(plugin['parameters'], dict)):
                raise ValueError('Invalid pre-processing configuration')
            if any(not isinstance(v, (str, int, float, bool))
                   or isinstance(v, float) and not np.isfinite(v) for v in plugin['parameters'].values()):
                raise ValueError('Pre-processing parameters must be finite scalar values')
        object.__setattr__(self, 'pre_processing_json', json.dumps(plugins, sort_keys=True, allow_nan=False))

    @property
    def native(self):
        return self.profile in ('baseline', 'plugins')

    @property
    def center_pitch_frame(self):
        return self.profile in ('plugins', 'centered_all')

    def to_dict(self):
        return {
            'schema_version': 1, 'method': self.method,
            'frame_duration_ms': self.frame_duration_ms, 'hop_duration_ms': self.hop_duration_ms,
            'f0_min': self.f0_min, 'f0_max': self.f0_max, 'ste_silence_ratio': self.ste_silence_ratio,
            'profile': self.profile, 'pre_processing': json.loads(self.pre_processing_json),
            'signal_source': 'file_dc_removed' if self.native else 'loaded_wav',
            'energy_source': 'before_signal_plugins_dc_centered_frames',
            'pitch_population': 'active' if self.native else 'all',
            'center_before_pitch': self.center_pitch_frame,
            'zcr_source': 'centered_pitch_active' if self.native else 'centered_frame_all',
            'window': 'rectangular', 'cepstrum_window': 'hamming', 'cepstrum_n_fft': 2048,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
            raise ValueError('Unsupported feature contract')
        try:
            result = cls(**{key: value[key] for key in (
                'method', 'frame_duration_ms', 'hop_duration_ms', 'f0_min', 'f0_max', 'ste_silence_ratio', 'profile')},
                pre_processing_json=json.dumps(value['pre_processing']))
        except (KeyError, TypeError) as exc:
            raise ValueError('Incomplete feature contract') from exc
        if result.to_dict() != dict(value):
            raise ValueError('Feature contract contains inconsistent extraction policies')
        return result

    @classmethod
    def from_detector(cls, detector, pre_processing=None, plugin_wrapper=False):
        return cls(detector.method, detector.frame_duration_ms, detector.hop_duration_ms,
                   detector.f0_min, detector.f0_max, detector.ste_silence_ratio,
                   'plugins' if plugin_wrapper else 'baseline', json.dumps(pre_processing or []))


@dataclass
class FrameFeatures:
    spec: FeatureSpec
    data: dict

    def __post_init__(self):
        times = np.asarray(self.data['frame_times'])
        if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError('Frame times must be finite and strictly increasing')
        for key in ('ste', 'is_active', 'candidate_f0', 'periodicity_vals', 'zcr_vals', 'lags'):
            value = np.asarray(self.data[key])
            if value.shape != times.shape or not np.all(np.isfinite(value)):
                raise ValueError('Feature arrays must be finite and align with frame times: '+key)
        if np.asarray(self.data['pitch_frames']).shape[0] != len(times):
            raise ValueError('Pitch frames must align with frame times')
        if (self.data.get('num_frames') != len(times)
                or np.asarray(self.data['is_active']).dtype != np.dtype(bool)
                or not np.array_equal(self.data['is_active'], self.data['ste'] >= self.data['ste_thresh'])):
            raise ValueError('Feature frame count/active mask must match the energy gate')

    def to_dict(self):
        return {**self.data, 'feature_contract': self.spec.to_dict()}


def training_contract(features, method):
    """Require one declared feature space; never infer framing from defaults."""
    if not features:
        raise ValueError('Training features must include a feature contract')
    expected = FeatureSpec.from_dict(features[0].get('feature_contract'))
    if expected.method != method:
        raise ValueError('Training method does not match the feature contract')
    for item in features:
        if FeatureSpec.from_dict(item.get('feature_contract')) != expected:
            raise ValueError('Training features use different feature spaces')
    return expected


@dataclass
class PredictionResult:
    data: dict
    f0_min: float
    f0_max: float

    def __post_init__(self):
        labels = np.asarray(self.data['labels'])
        contour = np.asarray(self.data['f0_contour'])
        if labels.ndim != 1 or labels.shape != contour.shape or len(labels) != len(self.data['frame_times']):
            raise ValueError('Prediction arrays must align with frame times')
        if not np.all(np.isin(labels, ('v', 'uv', 'sil'))):
            raise ValueError('Prediction labels must use v/uv/sil')
        active = labels == 'v'
        if (not np.all(np.isfinite(contour)) or np.any(contour[~active] != 0)
                or np.any((contour[active] < self.f0_min) | (contour[active] > self.f0_max))):
            raise ValueError('Prediction F0 must match voiced labels and pitch range')
        if 'num_voiced' in self.data and self.data['num_voiced'] != int(active.sum()):
            raise ValueError('Prediction voiced count must match labels')

    def to_dict(self):
        return dict(self.data)
