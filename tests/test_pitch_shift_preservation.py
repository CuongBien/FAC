"""The original Pitch-Shift source and pipeline remain an independent baseline."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np

from src.analysis.ssl_pitch_shift import analyze_signal_ssl_pitch_shift


class PitchShiftPreservationTests(unittest.TestCase):
    def test_original_source_is_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        expected=json.loads((root/'tests/fixtures/pitch_shift_original.json').read_bytes())
        # Normalize EOL for Windows/Linux checkouts, while preserving all source text.
        source=(root/'src/analysis/ssl_pitch_shift.py').read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(hashlib.sha256(source).hexdigest(),expected['canonical_source_sha256'])




if __name__=='__main__':unittest.main()
