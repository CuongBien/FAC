"""Train-selected presets frozen before stage 4 test evaluation; see docs/STAGE4_RESULTS.md."""

FROZEN_CONFIG_SHA256 = '34c1c6ba11a1c76ed82747fa11977fe5f8f1d0749eec4fec69acd9c58e610645'

CHAMPIONS = {'acf': {'plugins': ['BP', 'Energy', 'Viterbi'],
         'threshold': 0.45,
         'frame_duration_ms': 30.0,
         'hop_duration_ms': 10.0,
         'clipping_ratio': 0.4},
 'amdf': {'plugins': ['Energy', 'Viterbi'],
          'threshold': 0.5,
          'frame_duration_ms': 30.0,
          'hop_duration_ms': 10.0,
          'clipping_ratio': 0.4},
 'cepstrum': {'plugins': ['BP', 'Clip', 'Energy', 'Viterbi'],
              'threshold': 0.1,
              'frame_duration_ms': 30.0,
              'hop_duration_ms': 10.0,
              'clipping_ratio': 0.4}}
