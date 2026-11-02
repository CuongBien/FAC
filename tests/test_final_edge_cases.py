"""LAB number formats, early pitch validation, and non-destructive reporting."""
import contextlib
from copy import deepcopy
import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from src.core.lab_parser import parse_lab_file
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.core.pitch_detector import PitchDetector
from src.plugins import PluginPitchDetector
from src.analysis.validation import freeze_json, sha256, write_json, write_source_snapshot
from src.calibrated_presets import CHAMPIONS

ROOT = Path(__file__).resolve().parents[1]


class LabFormatTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'a.lab'
            path.write_text(text, encoding='utf-8')
            return parse_lab_file(path)

    def test_scientific_notation_bom_and_segment_labels(self):
        result = self.parse('\ufeffF0mean +1.8E+2\nF0std .1e2 # reference\nF0num 2.e2\n0.0 0.1 v\n')
        self.assertEqual((result['f0_mean'], result['f0_std'], result['f0_num']), (180., 10., 200.))
        self.assertEqual(result['segments'], [(0., .1, 'v')])

    def test_missing_statistics_and_plain_decimals_remain_supported(self):
        result = self.parse('f0MEAN 180.\nF0num 0\n')
        self.assertEqual(result['f0_mean'], 180.)
        self.assertIsNone(result['f0_std'])
        self.assertEqual(result['f0_num'], 0.)

    def test_invalid_statistics_cannot_be_partially_parsed(self):
        for key in ('F0mean', 'F0std', 'F0num'):
            for value in ('1.8e', '180Hz', '1.2.3', 'nan', 'inf', '1e309', '-1', ''):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, r'a\.lab:1:'):
                    self.parse(key+' '+value+'\n')

    def test_fractional_frame_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer count'):
            self.parse('F0num 2.5\n')

    def test_cli_uses_complete_scientific_references_with_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder/'tone.wav'
            t = np.arange(2400)/16000
            wavfile.write(path, 16000, (6000*np.sin(2*np.pi*180*t)).astype(np.int16))
            path.with_suffix('.lab').write_text('\ufeffF0mean 1.8e2\nF0std 1e1\nF0num 2e2\n', encoding='utf-8')
            out = folder/'out'
            run = subprocess.run([sys.executable, '-B', str(ROOT/'predict.py'), '-i', str(path),
                                  '-m', 'baseline', '-o', str(out)], cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((out/'tone_pitch_contour.png').exists())
            with (out/'evaluation_summary_metrics.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(float(row['Ref_F0mean']), 180.)
                self.assertEqual(float(row['Ref_F0std']), 10.)
                self.assertEqual(float(row['Ref_F0num']), 200.)
                self.assertNotEqual(row['Composite_Score_pct'], '')

    def test_current_eight_lab_references_are_unchanged(self):
        expected = {
            'phone_F1': (215.6, 20.6, 148), 'phone_M1': (123.7, 16.8, 232),
            'studio_F1': (229.6, 36.8, 127), 'studio_M1': (116.9, 26.4, 82),
            'phone_F2': (150.2, 30.7, 219), 'phone_M2': (130.2, 15.4, 123),
            'studio_F2': (198.6, 47.8, 139), 'studio_M2': (154.7, 30.3, 116),
        }
        for name, values in expected.items():
            folder = 'TinHieuHuanLuyen' if name.endswith('1') else 'TinHieuKiemThu'
            result = parse_lab_file(ROOT/folder/(name+'.lab'))
            self.assertEqual(tuple(result[key] for key in ('f0_mean', 'f0_std', 'f0_num')), values)


class PitchRangeTests(unittest.TestCase):
    def test_low_level_invalid_parameters_fail_even_for_empty_or_silent_audio(self):
        inputs = (np.array([]), np.zeros(400), np.sin(2*np.pi*180*np.arange(400)/16000))
        parameters = [(16000, 400, 70), (16000, 0, 400), (16000, 70, 70),
                      (16000, 70, 9000), (16000, np.nan, 400), (16000, 70, np.inf),
                      (0, 70, 400), (-16000, 70, 400), (np.nan, 70, 400)]
        for extract in (find_f0_acf, find_f0_amdf, find_f0_cepstrum):
            for frame in inputs:
                for rate, low, high in parameters:
                    with self.subTest(method=extract.__name__, length=len(frame), params=(rate, low, high)), \
                         self.assertRaisesRegex(ValueError, 'Pitch range'):
                        extract(frame, rate, f0_min=low, f0_max=high)

    def test_detector_silence_gate_cannot_hide_invalid_pitch_configuration(self):
        for method in ('acf', 'amdf', 'cepstrum'):
            for wrapped in (False, True):
                base = PitchDetector(method=method, f0_min=400, f0_max=70)
                detector = PluginPitchDetector(base) if wrapped else base
                with self.subTest(method=method, wrapped=wrapped), self.assertRaisesRegex(ValueError, 'Pitch range'):
                    detector.process_signal(np.zeros(1600), 16000)

    def test_valid_range_still_accepts_silent_audio(self):
        for extract in (find_f0_acf, find_f0_amdf, find_f0_cepstrum):
            self.assertEqual(extract(np.zeros(400), 16000)[0], 0.)
        for method in ('acf', 'amdf', 'cepstrum'):
            result = PitchDetector(method).process_signal(np.zeros(1600), 16000)
            self.assertEqual(result['num_voiced'], 0)
            self.assertEqual(result['num_silence'], result['num_frames'])


class ReportOwnershipTests(unittest.TestCase):
    def fixture(self, directory):
        spec = importlib.util.spec_from_file_location('stage4_report', ROOT/'scripts/19_report_validation.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        root = Path(directory)
        (root/'source.py').write_text('recorded_source = True\n', encoding='utf-8')
        provenance = {'base_commit': 'recorded', 'source_sha256': {'source.py': sha256(root/'source.py')}}
        folder = root/'run'; (folder/'train').mkdir(parents=True)
        chosen = deepcopy(CHAMPIONS)
        freeze_json(folder/'frozen_config.json', {'train_manifest': {}, 'provenance': provenance,
            'selected': chosen, 'train_metrics': {m: {'final_score': .2} for m in chosen},
            'num_trials': 3, 'num_pipeline_file_pairs': 3, 'search_space': {'frame_grid_ms': [30.]}})
        freeze_json(folder/'frozen_models.json', {'train_manifest': {}})
        pipelines, protocols = [], []
        for dataset in ('train', 'test'):
            for method in chosen:
                for strategy in ('baseline', 'historical_champion', 'selected_champion'):
                    pipelines.append({'dataset': dataset, 'method': method, 'strategy': strategy,
                        'metrics': {'final_score': .2}, 'files': [{'file': 'a.wav', 'pred_f0_mean': 180.,
                            'pred_f0_std': 10., 'pred_f0_num': 20, 'final_score': .2}]})
                protocols.append({'dataset': dataset, 'method': method, 'strategy': 'median_train_active',
                                  'metrics': {'final_score': .3}, 'files': []})
        write_json(folder/'evaluation.json', {'config_sha256': sha256(folder/'frozen_config.json'),
            'models_sha256': sha256(folder/'frozen_models.json'), 'pipelines': pipelines, 'protocols': protocols})
        write_source_snapshot(root, folder/'training_source_snapshot.tar.gz', provenance)
        write_json(folder/'runtime.json', {'snapshot_sha256': sha256(folder/'training_source_snapshot.tar.gz')})
        trace = {'source_revision': 'caaf751', 'plugin_detector_override': '427c0a4',
                 'replay': {'detector_function_calls': {'acf': 3, 'cepstrum': 0}}}
        write_json(folder/'cepstrum_trace.json', trace)
        write_json(folder/'legacy_with_cepstrum_branch/cepstrum_trace.json', trace)
        for method in chosen:
            write_json(folder/'train'/f'{method}_trials.json', [{'spec': chosen[method], 'metrics': {'final_score': .2}}])
        for name in ('evaluation_pipelines_per_file.csv', 'evaluation_protocols_per_file.csv', 'train/all_trials_per_file.csv'):
            (folder/name).write_text('file,final_score\na.wav,0.2\n', encoding='utf-8')
        write_json(folder/'train/plugin_sweep.json', [])
        (root/'docs').mkdir()
        owned = {root/'README.md': report.README_MARKERS, root/'docs/STAGE4_RESULTS.md': report.REPORT_MARKERS,
                 **{root/'docs'/name: report.CHAPTER_MARKERS for name in report.CHAPTERS}}
        for path, markers in owned.items():
            path.write_text('# Newer manual content\n\nCHAMPION_UNSUPERVISED_RESULTS.md; 288 cases; 108 tests.\n'
                            +markers[0]+'\nold generated block\n'+markers[1]+'\n\nNew future section: giữ nguyên.\n', encoding='utf-8')
        return report, root, folder, owned

    def test_regeneration_preserves_every_manual_region_and_is_idempotent(self):
        from src.reporting.publish import preview, apply
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report, root, folder, owned = self.fixture(directory)
            before = {path: path.read_text(encoding='utf-8') for path in owned}
            bundle = report.main(['--output-dir', str(folder), '--report-dir', str(root/'report')])
            self.assertEqual(before, {path: path.read_text(encoding='utf-8') for path in owned})
            plan = preview(bundle, root)
            self.assertTrue(plan.diff())
            apply(plan)
            after = {path: path.read_text(encoding='utf-8') for path in owned}
            apply(preview(bundle, root))
            self.assertEqual(bundle, report.main(['--output-dir', str(folder), '--report-dir', str(bundle)]))
            for path, (start, end) in owned.items():
                self.assertEqual(before[path].split(start)[0], after[path].split(start)[0])
                self.assertEqual(before[path].split(end)[1], after[path].split(end)[1])
                self.assertEqual(after[path], path.read_text(encoding='utf-8'))
            self.assertIn('CHAMPION_UNSUPERVISED_RESULTS.md', after[root/'README.md'])
            self.assertNotIn('phần tích hợp Champion còn thiếu bằng chứng', after[root/'README.md'])
            self.assertTrue((bundle/'summary.json').is_file())
            self.assertFalse((folder/'report_generation.json').exists())

    def test_missing_markers_fail_before_any_document_is_modified(self):
        from src.reporting.publish import preview
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report, root, folder, owned = self.fixture(directory)
            bundle = report.main(['--output-dir', str(folder), '--report-dir', str(root/'report')])
            path = root/'README.md'
            path.write_text(path.read_text(encoding='utf-8').replace(report.README_MARKERS[1], ''), encoding='utf-8')
            before = {p: p.read_bytes() for p in owned}
            with self.assertRaisesRegex(ValueError, 'generated block'):
                preview(bundle, root)
            self.assertEqual(before, {p: p.read_bytes() for p in owned})
            self.assertFalse((folder/'figures').exists())

    def test_ambiguous_and_reversed_markers_are_rejected(self):
        spec = importlib.util.spec_from_file_location('report_markers', ROOT/'scripts/19_report_validation.py')
        report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
        for source in ('<end><start>', '<start><start><end>', '<start><end><end>', 'no markers'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                report.replace_generated_block(source, 'new', '<start>', '<end>')


if __name__ == '__main__':
    unittest.main()
