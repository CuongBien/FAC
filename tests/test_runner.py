"""Phase integrity, process-crash recovery, relocation and source separation."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from src.experiments import champion
from src.experiments.dependencies import source_scopes
from src.experiments.legacy import freeze_json, read_frozen
from src.experiments.manifest import sha256, dataset_manifest
from src.experiments.protocol_specs import default_spec, validate_protocol
from src.experiments.registry import RunRegistry
from src.experiments.runner import RunService, run_phase, ROOT
from src.experiments.serialization import write_json
from src.experiments.spec import ExperimentSpec
from src.experiments.store import verify_run
from src.experiments.transactions import run_lock


def audio(folder, name='tone', frequency=180):
    folder.mkdir(parents=True, exist_ok=True)
    time = np.arange(3840)/16000
    values = .3*np.sin(2*np.pi*frequency*time)+.025*np.sin(2*np.pi*311*time)
    values[:800] *= .01
    wavfile.write(folder/(name+'.wav'), 16000, (values*32767).astype(np.int16))
    (folder/(name+'.lab')).write_text('F0mean 180\nF0std 10\nF0num 15\n', encoding='utf-8')


def fake_train(args, *, context):
    folder = Path(args.output_dir)
    common = {'train_manifest': dataset_manifest(args.train_dir),
              'provenance': context.provenance}
    freeze_json(folder/'frozen_config.json', {**common, 'experiment': {}})
    freeze_json(folder/'frozen_models.json', {**common, 'models': {}})


def fake_evaluate(args, *, context):
    folder = Path(args.output_dir)
    write_json(folder/'evaluation.json', {'config_sha256': sha256(folder/'frozen_config.json'),
        'models_sha256': sha256(folder/'frozen_models.json'), 'pipelines': [], 'protocols': [],
        'test_manifest': dataset_manifest(args.test_dir)})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train, self.test = self.root/'train', self.root/'test'
        audio(self.train)
        self.folder = self.root/'outputs/runs/test-run'
        self.service = RunService(self.folder)
        self.service.prepare(default_spec('champion'), self.train)

    def train_fake(self):
        with patch.object(champion, 'train', side_effect=fake_train):
            return self.service.train(self.train)

    def test_training_never_reads_test_directory_or_lab_statistics(self):
        class TrainArguments:
            train_dir = str(self.train)
            output_dir = str(self.root/'second-run')
            @property
            def test_dir(self):
                raise AssertionError('Training attempted to inspect test')
        with redirect_stdout(io.StringIO()), patch.object(champion, 'evaluate_result', side_effect=AssertionError('LAB statistics used')):
            run_phase('champion', 'train', TrainArguments())
        self.assertFalse(self.test.exists())
        manifest = verify_run(self.root/'second-run')
        self.assertEqual(manifest['phase'], 'trained')
        self.assertEqual(set(manifest['datasets']), {'train'})

    def test_completed_train_resume_verifies_without_refitting(self):
        self.train_fake()
        original = sha256(self.folder/'frozen_models.json')
        with patch.object(champion, 'train', side_effect=AssertionError('refit')):
            self.assertEqual(self.service.train(self.train, resume=True)['phase'], 'trained')
            with self.assertRaises(FileExistsError):
                self.service.train(self.train)
        self.assertEqual(sha256(self.folder/'frozen_models.json'), original)

    def test_failed_computation_keeps_phase_and_resume_restarts_uncommitted_work(self):
        def interrupted(args, *, context):
            (Path(args.output_dir)/'partial.json').write_text('{}')
            raise RuntimeError('interrupted computation')
        with patch.object(champion, 'train', side_effect=interrupted), self.assertRaises(RuntimeError):
            self.service.train(self.train)
        self.assertEqual(verify_run(self.folder)['phase'], 'prepared')
        self.assertFalse((self.folder/'frozen_models.json').exists())
        with patch.object(champion, 'train', side_effect=fake_train):
            self.service.train(self.train, resume=True)
        self.assertFalse((self.folder/'partial.json').exists())
        self.assertGreater(len(list((self.folder/'.attempts').iterdir())), 0)

    def test_interrupted_publication_resumes_without_recomputing(self):
        original_replace = os.replace
        def interrupted(source, destination):
            if Path(destination) == self.folder/'frozen_models.json':
                raise OSError('interrupted publication')
            return original_replace(source, destination)
        with patch.object(champion, 'train', side_effect=fake_train), \
             patch('src.experiments.transactions.os.replace', side_effect=interrupted), self.assertRaises(OSError):
            self.service.train(self.train)
        self.assertEqual(verify_run(self.folder)['phase'], 'prepared')
        self.assertTrue((self.folder/'.transaction.json').exists())
        journal = json.loads((self.folder/'.transaction.json').read_bytes())
        pending_model = self.folder/journal['work']/'frozen_models.json'
        saved = pending_model.read_bytes()
        pending_model.write_bytes(b'corrupted interrupted output')
        with patch.object(champion, 'train', side_effect=AssertionError('recomputed')), \
             self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.service.train(self.train, resume=True)
        pending_model.write_bytes(saved)
        with patch.object(champion, 'train', side_effect=AssertionError('recomputed')):
            self.service.train(self.train, resume=True)
        self.assertEqual(verify_run(self.folder)['phase'], 'trained')
        self.assertFalse((self.folder/'.transaction.json').exists())

    def test_failure_after_manifest_commit_is_recoverable(self):
        original_unlink = Path.unlink
        def interrupted(path, *args, **kwargs):
            if path == self.folder/'.transaction.json':
                raise OSError('crash after manifest commit')
            return original_unlink(path, *args, **kwargs)
        with patch.object(champion, 'train', side_effect=fake_train), \
             patch('src.experiments.transactions.Path.unlink', new=interrupted), self.assertRaises(OSError):
            self.service.train(self.train)
        self.assertEqual(verify_run(self.folder)['phase'], 'trained')
        with patch.object(champion, 'train', side_effect=AssertionError('refit')):
            self.service.train(self.train, resume=True)
        self.assertFalse((self.folder/'.transaction.json').exists())

    def test_missing_required_output_cannot_publish_a_completed_stage(self):
        with patch.object(champion, 'train', return_value=None), self.assertRaisesRegex(ValueError, 'produce'):
            self.service.train(self.train)
        self.assertEqual(verify_run(self.folder)['phase'], 'prepared')
        with self.assertRaisesRegex(ValueError, 'requires completed train'):
            self.service.evaluate(self.train, self.test)

    def test_backend_cannot_mutate_a_prepared_input(self):
        original = sha256(self.folder/'config.json')
        def bad(args, *, context):
            fake_train(args, context=context)
            (Path(args.output_dir)/'config.json').write_text('{}')
        with patch.object(champion, 'train', side_effect=bad), self.assertRaisesRegex(ValueError, 'committed input'):
            self.service.train(self.train)
        self.assertEqual(sha256(self.folder/'config.json'), original)
        self.assertEqual(verify_run(self.folder)['phase'], 'prepared')

    def test_invalid_freeze_links_do_not_publish_completion(self):
        def bad(args, *, context):
            fake_train(args, context=context)
            (Path(args.output_dir)/'frozen_models.sha256').write_text('0'*64)
        with patch.object(champion, 'train', side_effect=bad), self.assertRaisesRegex(ValueError, 'checksum'):
            self.service.train(self.train)
        self.assertEqual(verify_run(self.folder)['phase'], 'prepared')
        self.assertTrue((self.folder/'.transaction.json').exists())

    def test_changed_train_bytes_block_resume_before_compute(self):
        self.train_fake()
        (self.train/'tone.lab').write_text('F0mean 181\nF0std 10\nF0num 15\n')
        with patch.object(champion, 'train', side_effect=AssertionError('compute')), self.assertRaisesRegex(ValueError, 'Training data changed'):
            self.service.train(self.train, resume=True)

    def test_changed_test_data_blocks_completed_evaluation_resume(self):
        self.train_fake()
        audio(self.test)
        with patch.object(champion, 'evaluate', side_effect=fake_evaluate):
            self.service.evaluate(self.train, self.test)
        (self.test/'tone.lab').write_text('F0mean 200\nF0std 10\nF0num 15\n')
        with patch.object(champion, 'evaluate', side_effect=AssertionError('compute')), self.assertRaisesRegex(ValueError, 'inputs changed'):
            self.service.evaluate(self.train, self.test, resume=True)

    def test_evaluation_failure_preserves_freeze_and_hides_work_from_registry(self):
        self.train_fake()
        audio(self.test)
        original = sha256(self.folder/'frozen_models.json')
        def interrupted(args, *, context):
            fake_evaluate(args, context=context)
            raise RuntimeError('evaluation interrupted')
        with patch.object(champion, 'evaluate', side_effect=interrupted), self.assertRaises(RuntimeError):
            self.service.evaluate(self.train, self.test)
        self.assertEqual(verify_run(self.folder)['phase'], 'trained')
        self.assertEqual(sha256(self.folder/'frozen_models.json'), original)
        entries = RunRegistry(self.root).verify()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['phase'], 'trained')
        with patch.object(champion, 'evaluate', side_effect=fake_evaluate):
            self.service.evaluate(self.train, self.test, resume=True)

    def test_run_and_data_can_move_without_changing_identity_or_refitting(self):
        self.train_fake()
        identity = verify_run(self.folder)['experiment_id']
        moved = self.root/'relocated'
        shutil.copytree(self.folder, moved/'run')
        shutil.copytree(self.train, moved/'data')
        with patch.object(champion, 'train', side_effect=AssertionError('refit')):
            result = RunService(moved/'run').train(moved/'data', resume=True)
        self.assertEqual(result['experiment_id'], identity)
        self.assertNotIn(str(self.root), (moved/'run/manifest.json').read_text())

    def test_source_is_optional_git_and_only_renderer_changes_are_allowed(self):
        source = self.root/'source'
        files = {name for names in source_scopes(ROOT).values() for name in names}
        for name in files:
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
        service = RunService(self.root/'portable-run', source_root=source)
        service.prepare(default_spec('champion'), self.train)
        self.assertIsNone(verify_run(service.folder)['source']['git'])
        with patch.object(champion, 'train', side_effect=fake_train):
            service.train(self.train)
        renderer = source/'src/experiments/champion_report.py'
        renderer.write_bytes(renderer.read_bytes()+b'\n# renderer-only change\n')
        formatting = source/'src/metrics/formatting.py'
        formatting.write_bytes(formatting.read_bytes()+b'\n# presentation-only change\n')
        with patch.object(champion, 'train', side_effect=AssertionError('refit')):
            service.train(self.train, resume=True)
        core = source/'src/core/acf.py'
        core.write_bytes(core.read_bytes()+b'\n# scientific source changed\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            service.train(self.train, resume=True)

    def test_lf_crlf_source_change_is_allowed_without_rewriting_snapshot(self):
        source = self.root/'source'
        for name in {name for names in source_scopes(ROOT).values() for name in names}:
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
        service = RunService(self.root/'portable-run', source_root=source)
        service.prepare(default_spec('champion'), self.train)
        original = sha256(service.folder/'training_source_snapshot.tar.gz')
        core = source/'src/core/acf.py'
        lock = source/'uv.lock'
        for path in (core, lock):
            before = path.read_bytes()
            lf = before.replace(b'\r\n', b'\n')
            after = lf if b'\r\n' in before else lf.replace(b'\n', b'\r\n')
            self.assertNotEqual(before, after)
            path.write_bytes(after)
        with patch.object(champion, 'train', side_effect=fake_train):
            service.train(self.train)
        self.assertEqual(sha256(service.folder/'training_source_snapshot.tar.gz'), original)

    def test_supplied_source_root_must_match_executing_code(self):
        source = self.root/'source'
        for name in {name for names in source_scopes(ROOT).values() for name in names}:
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
        core = source/'src/core/acf.py'
        core.write_bytes(core.read_bytes()+b'\n# different source than executing interpreter\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            RunService(self.root/'different-source', source_root=source).prepare(default_spec('champion'), self.train)
        self.assertFalse((self.root/'different-source').exists())

    def test_os_lock_excludes_another_process_and_recovers_after_kill(self):
        # uv's venv executable can be a launcher. Kill the actual interpreter,
        # not a launcher that would leave a lock-owning grandchild alive.
        code = ('import sys,json; sys.path[:0]=json.loads(sys.argv[2]); '
                'from pathlib import Path; from src.experiments.transactions import run_lock; '
                'context=run_lock(Path(sys.argv[1])); context.__enter__(); '
                'print("locked",flush=True); sys.stdin.buffer.read(1)')
        executable = getattr(sys, '_base_executable', None) or sys.executable
        process = subprocess.Popen([executable, '-B', '-c', code, str(self.folder), json.dumps(sys.path)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(process.stdout.readline().strip(), b'locked')
            with self.assertRaises(FileExistsError):
                with run_lock(self.folder):
                    pass
            process.kill()
            process.wait(timeout=10)
            with run_lock(self.folder):
                self.assertTrue((self.folder/'.run.lock').exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    def test_config_overrides_and_wrong_runner_are_not_silently_ignored(self):
        bad = default_spec('champion').to_dict()
        bad['pipelines']['acf']['reference_threshold'] = .9
        with self.assertRaisesRegex(ValueError, 'unsupported overrides'):
            validate_protocol(ExperimentSpec(bad))
        bad = default_spec('champion').to_dict()
        bad['fit']['parameters']['gmm']['prior'] = False
        with self.assertRaisesRegex(ValueError, 'unsupported overrides'):
            validate_protocol(ExperimentSpec(bad))
        args = SimpleNamespace(train_dir=str(self.train), output_dir=str(self.folder), test_dir='unused')
        with self.assertRaisesRegex(ValueError, 'different protocol'):
            run_phase('stage4', 'evaluate', args)

    def test_both_shipped_specs_match_the_executable_protocols(self):
        for name in ('champion_unsupervised.json', 'stage4_calibration.json'):
            spec = ExperimentSpec(json.loads((ROOT/'configs/experiments'/name).read_bytes()))
            self.assertIn(validate_protocol(spec), ('champion', 'stage4'))


if __name__ == '__main__':
    unittest.main()
