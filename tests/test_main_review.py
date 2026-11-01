"""Regression checks for the main review: Otsu, partial LAB and evidence integrity."""
import csv
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from src.analysis.unsupervised_threshold import compute_otsu_threshold
from src.analysis.validation import (
    freeze_json, read_evaluation_bundle, sha256, write_json,
    write_source_snapshot, verify_source_snapshot,
)
from src.calibrated_presets import CHAMPIONS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('artifact_verifier', ROOT/'scripts/20_verify_validation_artifacts.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class OtsuTests(unittest.TestCase):
    def test_returned_boundary_preserves_the_optimized_histogram_split(self):
        values = np.array([.1, .4, .6, .9])
        threshold = compute_otsu_threshold(values, n_bins=2)
        self.assertAlmostEqual(threshold, .5)
        np.testing.assert_array_equal(values < threshold, [True, True, False, False])
        np.testing.assert_array_equal(values >= threshold, [False, False, True, True])

    def test_constant_values_never_use_an_artificial_histogram_endpoint(self):
        for value in (-3., 0., .5, 2.):
            with self.subTest(value=value):
                self.assertEqual(compute_otsu_threshold(np.full(10, value)), value)

    def test_missing_observations_and_invalid_histograms(self):
        self.assertEqual(compute_otsu_threshold(np.array([np.nan, np.inf])), .5)
        self.assertEqual(compute_otsu_threshold(np.array([np.nan, .8])), .8)
        for n_bins in (0, 1, 2.5):
            with self.subTest(n_bins=n_bins), self.assertRaises(ValueError):
                compute_otsu_threshold(np.array([.1, .9]), n_bins=n_bins)
        with self.assertRaises(ValueError):
            compute_otsu_threshold(np.ones((2, 2)))


class PartialLabCliTests(unittest.TestCase):
    def test_missing_std_still_exports_plot_and_undefined_metric_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder/'tone.wav'
            time = np.arange(6400)/16000
            wavfile.write(path, 16000, (6000*np.sin(2*np.pi*180*time)).astype(np.int16))
            path.with_suffix('.lab').write_text('F0mean 180\nF0num 30\n', encoding='utf-8')
            output = folder/'out'
            run = subprocess.run([sys.executable, '-B', str(ROOT/'predict.py'), '-i', str(path),
                                  '-m', 'baseline', '-o', str(output)], cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((output/'tone_pitch_contour.png').is_file())
            self.assertIn('N/A', run.stdout)
            with (output/'evaluation_summary_metrics.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            for row in rows:
                for key in ('Ref_F0std', 'MAPE_F0std_pct', 'Composite_Score_pct'):
                    self.assertEqual(row[key], '')


class ArtifactIntegrityTests(unittest.TestCase):
    def bundle(self, root):
        source = root/'src'/'a.py'
        source.parent.mkdir()
        source.write_bytes(b'x = 1\n')
        provenance = {'base_commit': 'recorded-revision', 'source_sha256': {'src/a.py': sha256(source)}}
        folder = root/'run'
        freeze_json(folder/'frozen_config.json', {
            'train_manifest': {}, 'provenance': provenance,
            'selected': {method: {'method': method, **preset} for method, preset in CHAMPIONS.items()},
        })
        freeze_json(folder/'frozen_models.json', {'train_manifest': {}})
        result = {'config_sha256': sha256(folder/'frozen_config.json'),
                  'models_sha256': sha256(folder/'frozen_models.json'), 'pipelines': [], 'protocols': []}
        write_json(folder/'evaluation.json', result)
        write_source_snapshot(root, folder/'training_source_snapshot.tar.gz', provenance)
        write_json(folder/'runtime.json', {'snapshot_sha256': sha256(folder/'training_source_snapshot.tar.gz')})
        return folder, provenance

    def test_complete_bundle_and_exported_presets_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, _ = self.bundle(Path(directory))
            with patch.object(verifier, 'FROZEN_CONFIG_SHA256', sha256(folder/'frozen_config.json')):
                report = verifier.verify(folder)
                self.assertTrue(report['freeze_and_export_match'])
                self.assertEqual(report['snapshot_source_files'], 1)
            with patch.object(verifier, 'FROZEN_CONFIG_SHA256', 'wrong'):
                with self.assertRaisesRegex(ValueError, 'Exported preset checksum'):
                    verifier.verify(folder)

    def test_individually_valid_freezes_cannot_be_mixed_with_an_old_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, _ = self.bundle(Path(directory))
            replacement = folder/'other'
            freeze_json(replacement/'frozen_models.json', {'train_manifest': {}, 'new_model': True})
            for suffix in ('.json', '.sha256'):
                (folder/('frozen_models'+suffix)).write_bytes((replacement/('frozen_models'+suffix)).read_bytes())
            with self.assertRaisesRegex(ValueError, 'different frozen models'):
                read_evaluation_bundle(folder)

    def test_snapshot_checks_member_bytes_and_runtime_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, provenance = self.bundle(Path(directory))
            write_json(folder/'runtime.json', {'snapshot_sha256': 'wrong'})
            with self.assertRaisesRegex(ValueError, 'Runtime record'):
                verifier.verify(folder, check_presets=False)
            with tarfile.open(folder/'training_source_snapshot.tar.gz', 'w:gz') as archive:
                entry = tarfile.TarInfo('src/a.py')
                entry.size = len(b'stale bytes')
                archive.addfile(entry, io.BytesIO(b'stale bytes'))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_source_snapshot(folder/'training_source_snapshot.tar.gz', provenance)
            # A readable archive with the correct member name can still be stale.
            with tarfile.open(folder/'training_source_snapshot.tar.gz', 'w:gz') as archive:
                archive.add(Path(directory)/'src/a.py', arcname='different.py')
            with self.assertRaisesRegex(ValueError, 'member list'):
                verify_source_snapshot(folder/'training_source_snapshot.tar.gz', provenance)

    def test_repair_preserves_old_evidence_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = self.bundle(root)
            archive = folder/'training_source_snapshot.tar.gz'
            with tarfile.open(archive, 'w:gz'):
                pass
            write_json(folder/'runtime.json', {'python': 'recorded', 'snapshot_sha256': sha256(archive)})
            write_json(folder/'validation.json', {'old_record': True})
            old_digest = sha256(archive)
            with patch.object(verifier, 'ROOT', root):
                verifier.repair_snapshot(folder)
                repaired_digest = sha256(archive)
                verifier.repair_snapshot(folder)
            self.assertNotEqual(repaired_digest, old_digest)
            self.assertEqual(sha256(archive), repaired_digest)
            self.assertEqual(sha256(folder/'historical_source_snapshot/training_source_snapshot.tar.gz'), old_digest)
            runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
            self.assertEqual(runtime['python'], 'recorded')
            self.assertEqual(runtime['snapshot_reconstruction']['previous_snapshot_sha256'], old_digest)
            self.assertTrue(json.loads((folder/'historical_source_snapshot/validation.json').read_text())['old_record'])

    def test_reconstruction_requires_the_exact_recorded_git_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b'x = 1\r\n'
            provenance = {'base_commit': 'saved', 'source_sha256': {
                'src/a.py': hashlib.sha256(original).hexdigest()}}
            with patch('src.analysis.validation.subprocess.check_output', return_value=b'x = 1\n') as git:
                target = root/'snapshot.tar.gz'
                self.assertEqual(write_source_snapshot(root, target, provenance), 1)
                git.assert_called_once_with(['git', 'show', 'saved:src/a.py'], cwd=root)
            with tarfile.open(target) as archive:
                self.assertEqual(archive.extractfile('src/a.py').read(), original)
            with self.assertRaises(FileExistsError):
                write_source_snapshot(root, target, provenance)
            with patch('src.analysis.validation.subprocess.check_output', return_value=b'wrong'):
                with self.assertRaisesRegex(ValueError, 'Cannot reconstruct'):
                    write_source_snapshot(root, root/'failed.tar.gz', provenance)
                self.assertFalse((root/'failed.tar.gz').exists())

    def test_snapshot_paths_cannot_escape_the_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../outside.py', '/outside.py', 'C:/outside.py', 'src\\a.py'):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'snapshot path'):
                    write_source_snapshot(root, root/'bad.tar.gz', {
                        'base_commit': 'saved', 'source_sha256': {name: 'hash'}})


if __name__ == '__main__':
    unittest.main()
