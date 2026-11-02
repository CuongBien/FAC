"""Each Pitch-Shift branch must use its own estimator on both views."""
import unittest
from unittest.mock import patch
import numpy as np
from src.analysis.ssl_pitch_shift import analyze_signal_ssl_pitch_shift
from src.analysis.ssl_pitch_shift_methods import analyze_signal_pitch_shift_by_method


class NativePitchShiftTests(unittest.TestCase):
    def signal(self):return np.sin(2*np.pi*150*np.arange(1600)/16000)

    def test_acf_dispatch_is_identical_to_preserved_original(self):
        before=analyze_signal_ssl_pitch_shift(self.signal(),16000)
        after=analyze_signal_pitch_shift_by_method(self.signal(),16000,'acf')
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_amdf_and_cepstrum_never_call_acf_and_use_own_metric(self):
        for method,expected in (('amdf',.8),('cepstrum',2/3)):
            def own_estimator(frame,rate,low,high,**kwargs):
                return (150. if len(frame)==480 else 187.5),.2,100
            with self.subTest(method=method), \
                 patch('src.analysis.ssl_pitch_shift.find_f0_acf',side_effect=AssertionError('ACF leak')), \
                 patch('src.analysis.ssl_pitch_shift_methods.find_f0_acf',side_effect=AssertionError('ACF leak')), \
                 patch('src.analysis.ssl_pitch_shift_methods.find_f0_'+method,side_effect=own_estimator) as own:
                result=analyze_signal_pitch_shift_by_method(self.signal(),16000,method)
                np.testing.assert_allclose(result['probabilities'],expected)
                np.testing.assert_array_equal(result['labels'],np.full(len(result['labels']),'v'))
                self.assertEqual(own.call_count,2*len(result['labels']))



    def test_real_signal_and_silence_are_finite_and_aligned(self):
        for method in ('amdf','cepstrum'):
            result=analyze_signal_pitch_shift_by_method(self.signal(),16000,method)
            self.assertEqual(len(result['labels']),8)
            self.assertTrue(np.all(np.isfinite(result['probabilities'])))
            np.testing.assert_array_equal(result['f0_contour'][result['labels']!='v'],0.)
            silent=analyze_signal_pitch_shift_by_method(np.zeros(1600),16000,method)
            self.assertFalse(np.any(silent['labels']=='v'))

    def test_invalid_settings_are_rejected(self):
        for method in ('unknown','amdf','cepstrum'):
            with self.assertRaises(ValueError):
                analyze_signal_pitch_shift_by_method(self.signal(),16000,method,alpha=0)


if __name__=='__main__':unittest.main()
