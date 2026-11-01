"""Portable experiment contracts, immutable stores and read-only legacy indexing."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import ExperimentSpec, RunStore, RunRegistry
from src.experiments.legacy import freeze_json, inspect_legacy_run, read_frozen
from src.experiments.manifest import (
    capture_source, capture_dataset, verify_source_scope, sha256,
)
from src.experiments.serialization import write_json
from src.experiments.snapshots import write_source_snapshot
from src.experiments.store import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def spec_document():
    return {'schema_version': 1, 'name': 'portable-tone', 'metric_version': 'legacy-v1',
            'pipelines': {'acf': {'frame_duration_ms': 30., 'hop_duration_ms': 10.,
                'f0_min': 70., 'f0_max': 400., 'ste_silence_ratio': .008,
                'reference_threshold': .45, 'median_size': 3, 'use_median_filter': True,
                'pre_processors': [], 'post_processors': []}},
            'fit': {'mode': 'frozen', 'population': 'train_active', 'strategies': ['otsu']}}


def inputs(root):
    (root/'code.py').write_bytes(b'x = 1\n')
    (root/'report.py').write_bytes(b'title = "one"\n')
    (root/'data.wav').write_bytes(b'raw audio bytes')
    source = capture_source(root, {'train': ['code.py'], 'predict': ['code.py'], 'report': ['report.py']})
    dataset = {'train': capture_dataset(root, ['data.wav'])}
    return dataset, source


def legacy_fixture(root):
    folder = root/'outputs'/'old'
    folder.mkdir(parents=True)
    (root/'source.py').write_bytes(b'x = 1\r\n')
    provenance = {'base_commit': 'recorded', 'source_sha256': {'source.py': sha256(root/'source.py')}}
    # Fix original byte identity to CRLF regardless of the test host platform.
    for name, value in {
        'frozen_config': {'selected': {'acf': {'threshold': .45}}, 'provenance': provenance,
                          'train_manifest': {}, 'frozen_at_utc': '2026-11-01T00:00:00+00:00'},
        'frozen_models': {'train_manifest': {}},
    }.items():
        data = (json.dumps(value, indent=2)+'\n').replace('\n', '\r\n').encode()
        (folder/(name+'.json')).write_bytes(data)
        (folder/(name+'.sha256')).write_text(sha256(folder/(name+'.json'))+'\n', encoding='ascii')
    write_json(folder/'evaluation.json', {
        'config_sha256': (folder/'frozen_config.sha256').read_text().strip(),
        'models_sha256': (folder/'frozen_models.sha256').read_text().strip(),
        'pipelines': [{'files': [{'file': 'a.wav'}]}], 'protocols': []})
    write_source_snapshot(root, folder/'training_source_snapshot.tar.gz', provenance)
    write_json(folder/'runtime.json', {'snapshot_sha256': sha256(folder/'training_source_snapshot.tar.gz')})
    write_json(folder/'report_generation.json', {'evaluation_sha256': sha256(folder/'evaluation.json')})
    return folder


class SpecTests(unittest.TestCase):
    def test_spec_identity_is_immutable_and_independent_of_key_order(self):
        document = spec_document()
        spec = ExperimentSpec(document)
        expected = spec.sha256
        document['pipelines']['acf']['f0_min'] = 100
        returned = spec.to_dict(); returned['name'] = 'changed'
        self.assertEqual(spec.sha256, expected)
        self.assertEqual(ExperimentSpec(dict(reversed(list(spec.to_dict().items())))).sha256, expected)

    def test_spec_rejects_hidden_defaults_invalid_modes_and_nonfinite_values(self):
        cases = []
        for field in ('reference_threshold', 'hop_duration_ms', 'median_size', 'pre_processors'):
            value = spec_document(); del value['pipelines']['acf'][field]; cases.append(value)
        value = spec_document(); value['fit']['population'] = 'input_active'; cases.append(value)
        value = spec_document(); value['pipelines']['acf']['f0_max'] = float('nan'); cases.append(value)
        value = spec_document(); value['pipelines']['acf']['post_processors'] = ['invalid']; cases.append(value)
        value = spec_document(); value['schema_version'] = True; cases.append(value)
        cases.append([])
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExperimentSpec(value)


class ManifestTests(unittest.TestCase):
    def test_eol_changes_preserve_canonical_identity_but_real_code_changes_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = inputs(root)
            (root/'code.py').write_bytes(b'x = 1\r\n')
            second = capture_source(root, {'train': ['code.py'], 'predict': ['code.py'], 'report': ['report.py']})
            verify_source_scope(first, second, 'train')
            self.assertNotEqual(first['stages']['train']['raw_sha256'], second['stages']['train']['raw_sha256'])
            (root/'code.py').write_bytes(b'x = 2\r\n')
            third = capture_source(root, {'train': ['code.py']})
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                verify_source_scope(first, third, 'train')

    def test_renderer_changes_do_not_invalidate_train_and_git_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _, first = inputs(root)
            (root/'report.py').write_bytes(b'title = "two"\n')
            with patch('src.experiments.manifest.subprocess.check_output', side_effect=FileNotFoundError):
                second = capture_source(root, {'train': ['code.py'], 'predict': ['code.py'], 'report': ['report.py']})
            self.assertIsNone(second['git'])
            verify_source_scope(first, second, 'train')
            verify_source_scope(first, second, 'predict')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                verify_source_scope(first, second, 'report')

    def test_dataset_identity_survives_relocation_and_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root/'left', root/'right'; left.mkdir(); right.mkdir()
            for path in (left, right): (path/'a.wav').write_bytes(b'audio')
            first, second = capture_dataset(left, ['a.wav']), capture_dataset(right, ['a.wav'])
            self.assertEqual(first, second)
            (right/'a.wav').write_bytes(b'other')
            self.assertNotEqual(first['identity_sha256'], capture_dataset(right, ['a.wav'])['identity_sha256'])

    def test_source_scope_must_be_explicit_and_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            for scopes in ({}, {'unknown': ['a.py']}, {'train': []}, {'train': ['../outside.py']}):
                with self.subTest(scopes=scopes), self.assertRaises(ValueError):
                    capture_source(Path(directory), scopes)


class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.datasets, self.source = inputs(self.root)
        self.spec = ExperimentSpec(spec_document())
        self.store = RunStore(self.root/'outputs/runs')

    def test_run_identity_ignores_report_revision_and_artifact_json_is_lf(self):
        first = self.store.create(self.spec, self.datasets, self.source, 'one')
        (self.root/'report.py').write_bytes(b'title = "two"\r\n')
        updated = capture_source(self.root, {'train': ['code.py'], 'predict': ['code.py'], 'report': ['report.py']})
        self.store.create(self.spec, self.datasets, updated, 'two')
        self.assertEqual(self.store.verify('one')['experiment_id'], self.store.verify('two')['experiment_id'])
        self.assertNotIn(b'\r', (first/'manifest.json').read_bytes())
        self.assertNotIn(str(self.root), (first/'manifest.json').read_text())
        with self.assertRaises(FileExistsError): self.store.create(self.spec, self.datasets, self.source, 'one')

    def test_immutable_artifacts_and_tampering_detection(self):
        folder = self.store.create(self.spec, self.datasets, self.source, 'one')
        path = self.store.add_json('one', 'models/model.json', {'threshold': .4}, 'model')
        self.assertEqual(len(self.store.verify('one')['artifacts']), 2)
        with self.assertRaises(FileExistsError): self.store.add_json('one', 'models/model.json', {}, 'model')
        path.write_bytes(b'{"threshold":0.5}\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'): self.store.verify('one')
        self.assertTrue((folder/'manifest.json').exists())

    def test_locked_writer_and_invalid_paths_do_not_write_artifacts(self):
        folder = self.store.create(self.spec, self.datasets, self.source, 'one')
        (folder/'.writer.lock').write_text('another writer')
        with self.assertRaises(FileExistsError): self.store.add_json('one', 'models/a.json', {}, 'model')
        self.assertFalse((folder/'models/a.json').exists())
        (folder/'.writer.lock').unlink()
        for name in ('../outside.json', 'manifest.json', 'C:/outside.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.add_json('one', name, {}, 'model')

    def test_atomic_failure_leaves_previous_document_and_no_temporary_files(self):
        path = self.root/'index.json'; path.write_bytes(b'{"old":true}\n')
        with patch('src.experiments.store.os.replace', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError): atomic_write_json(path, {'new': True})
        self.assertEqual(path.read_bytes(), b'{"old":true}\n')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_invalid_metadata_and_run_ids_are_rejected_before_creation(self):
        changed = deepcopy(self.datasets); changed['train']['files']['data.wav']['size'] += 1
        with self.assertRaisesRegex(ValueError, 'manifest checksum'): self.store.create(self.spec, changed, self.source, 'bad')
        self.assertFalse((self.store.root/'bad').exists())
        with self.assertRaises(ValueError): self.store.create(self.spec, self.datasets, self.source, '../bad')


class LegacyRegistryTests(unittest.TestCase):
    def test_legacy_eol_reconstruction_is_read_only_and_keeps_original_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = legacy_fixture(Path(directory))
            original = inspect_legacy_run(folder)
            for name in ('frozen_config.json', 'frozen_models.json', 'evaluation.json'):
                path = folder/name; path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
            before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
            result = inspect_legacy_run(folder)
            self.assertEqual(result['run_id'], original['run_id'])
            self.assertIn('frozen_config.json', result['eol_reconstructed_in_memory'])
            self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()})
            with self.assertRaises(ValueError): read_frozen(folder/'frozen_config.json')

    def test_legacy_reader_rejects_real_tampering_even_with_different_eol(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = legacy_fixture(Path(directory))
            path = folder/'frozen_config.json'
            path.write_bytes(path.read_bytes().replace(b'0.45', b'0.46').replace(b'\r\n', b'\n'))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'): inspect_legacy_run(folder)

    def test_pinned_evaluation_is_checked_and_invalid_runs_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = legacy_fixture(root)
            value = json.loads((folder/'evaluation.json').read_bytes()); value['protocols'] = [{'files': []}]
            write_json(folder/'evaluation.json', value)
            registry = RunRegistry(root)
            entries = registry.verify()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]['status'], 'invalid')
            self.assertIn('checksum mismatch', entries[0]['error'])

    def test_mixed_registry_survives_relocation_and_cannot_write_inside_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'one'; root.mkdir()
            legacy_fixture(root); datasets, source = inputs(root)
            RunStore(root/'outputs/runs').create(ExperimentSpec(spec_document()), datasets, source, 'new')
            registry = RunRegistry(root)
            before = {path.relative_to(root): sha256(path) for path in (root/'outputs').rglob('*') if path.is_file()}
            entries = registry.verify()
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(entry['status'] == 'verified' for entry in entries))
            self.assertEqual(before, {path.relative_to(root): sha256(path) for path in (root/'outputs').rglob('*') if path.is_file()})
            with self.assertRaises(ValueError): registry.write_index('outputs/old/index.json')
            registry.write_index()
            relocated = Path(directory)/'two'; shutil.copytree(root, relocated)
            self.assertEqual(registry.scan(), RunRegistry(relocated).scan())
            self.assertEqual(registry.verify('new')[0]['phase'], 'prepared')

    def test_cli_list_verify_and_no_detector_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); legacy_fixture(root)
            command = [sys.executable, '-B', '-m', 'src.experiments', '--root', str(root)]
            result = subprocess.run(command+['list', '--json'], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['runs']), 1)
            (root/'outputs/old/runtime.json').write_text('{}')
            result = subprocess.run(command+['verify'], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            probe = subprocess.run([sys.executable, '-B', '-c',
                'import sys; import src.experiments; print("src.analysis" in sys.modules)'],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(probe.stdout.strip(), 'False')

    def test_malformed_native_manifest_does_not_hide_valid_legacy_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); legacy_fixture(root)
            folder = root/'outputs/runs/broken'; folder.mkdir(parents=True)
            for invalid in (None, [], {'format': 'fac.run.v1', 'schema_version': 1, 'artifacts': []}):
                atomic_write_json(folder/'manifest.json', invalid)
                entries = RunRegistry(root).verify()
                self.assertEqual(len(entries), 2)
                self.assertEqual(sum(row['status'] == 'verified' for row in entries), 1)
                self.assertEqual(sum(row['status'] == 'invalid' for row in entries), 1)


if __name__ == '__main__':
    unittest.main()
