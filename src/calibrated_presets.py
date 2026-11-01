"""Train-selected presets frozen before stage 4 test evaluation; see docs/STAGE4_RESULTS.md."""

FROZEN_CONFIG_SHA256 = '83dfda76c5a2ce34846a8c4eff9132296e2c09846e4d8e2cf44bbac5be884dc5'

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
