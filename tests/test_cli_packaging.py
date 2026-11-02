"""Public entrypoints and logical source resources; wheel execution is separate."""
import ast
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from src.cli.main import main
from src.experiments.manifest import source_fingerprint
from src.experiments.resources import source_path
from src.experiments.serialization import json_bytes
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot

ROOT = Path(__file__).resolve().parents[1]


class CliPackagingTests(unittest.TestCase):
    def resource(self, root, data=b'fixed lock\n'):
        resource = root/'src/_runtime'
        blob = resource/'files/test.txt'
        blob.parent.mkdir(parents=True)
        blob.write_bytes(data)
        record = {'path': 'files/test.txt', 'sha256': hashlib.sha256(data).hexdigest()}
        (resource/'manifest.json').write_bytes(json_bytes({'schema_version': 1, 'files': {'uv.lock': record}}))
        return blob

    def test_root_help_and_version_do_not_import_scientific_dependencies(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('numpy', 'scipy', 'matplotlib', 'src.core', 'src.experiments')):
            raise AssertionError('unnecessary import: '+fullname)
sys.meta_path.insert(0, Block())
from src.cli.main import main
main([sys.argv[1]])
'''
        for argument in ('--help', '--version'):
            result = subprocess.run([sys.executable, '-B', '-c', code, argument], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.strip())

    def test_unknown_command_is_an_argument_error(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['unknown'])
        self.assertEqual(caught.exception.code, 2)

    def test_dispatch_preserves_subcommand_arguments_and_exit_code(self):
        from src.experiments import __main__ as experiments
        from src.reporting import __main__ as reporting
        from src.cli import predict
        for command, target, expected in (
            ('predict', predict, ['--help']), ('report', reporting, ['--help']),
            ('runs', experiments, ['--help']), ('experiment', experiments, ['run', '--help'])):
            with patch.object(target, 'main', return_value=7) as run:
                self.assertEqual(main([command, '--help']), 7)
                run.assert_called_once_with(expected)

    def test_prediction_functions_and_main_body_preserve_pre_refactor_ast(self):
        with tarfile.open(ROOT/'outputs/refactor_phase5/source_snapshot.tar.gz') as archive:
            original = ast.parse(archive.extractfile('predict.py').read())
        current = ast.parse((ROOT/'src/cli/predict.py').read_bytes())
        functions = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        old, new = functions(original), functions(current)
        for name in ('get_champion_detectors', 'get_baseline_detectors', 'process_single_file'):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]))
        adjusted = deepcopy(new['main'])
        adjusted.args = old['main'].args
        for node in ast.walk(adjusted):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'parse_args':
                node.args = []
        self.assertEqual(ast.dump(old['main']), ast.dump(adjusted))

    def test_legacy_prediction_imports_are_the_same_public_functions(self):
        import predict
        from src.cli import predict as installed
        for name in ('main', 'process_single_file', 'get_champion_detectors', 'get_baseline_detectors'):
            self.assertIs(getattr(predict, name), getattr(installed, name))

    def test_source_checkout_needs_no_resource_index(self):
        self.assertEqual(source_path(ROOT, 'uv.lock'), ROOT/'uv.lock')

    def test_wheel_resource_has_priority_over_unrelated_site_packages_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = self.resource(root)
            (root/'uv.lock').write_bytes(b'another distribution\n')
            self.assertEqual(source_path(root, 'uv.lock'), blob.resolve())
            record = source_fingerprint(root, ['uv.lock'])
            self.assertEqual(record['files']['uv.lock']['raw_sha256'], hashlib.sha256(blob.read_bytes()).hexdigest())

    def test_tampered_resource_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = self.resource(root)
            blob.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                source_path(root, 'uv.lock')

    def test_resource_paths_cannot_escape_the_source_or_blob_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.resource(root)
            for name in ('../uv.lock', 'D:/uv.lock', 'src\\..\\uv.lock'):
                with self.assertRaises(ValueError):
                    source_path(root, name)
            index = root/'src/_runtime/manifest.json'
            value = json.loads(index.read_bytes())
            value['files']['uv.lock']['path'] = '../outside.txt'
            index.write_bytes(json_bytes(value))
            with self.assertRaises(ValueError):
                source_path(root, 'uv.lock')

    def test_snapshot_keeps_logical_names_and_exact_resource_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = self.resource(root)
            provenance = {'base_commit': 'unavailable', 'source_sha256': {'uv.lock': hashlib.sha256(blob.read_bytes()).hexdigest()}}
            archive_path = root/'source.tar.gz'
            self.assertEqual(write_source_snapshot(root, archive_path, provenance), 1)
            self.assertEqual(verify_source_snapshot(archive_path, provenance), 1)
            with tarfile.open(archive_path) as archive:
                self.assertEqual(archive.getnames(), ['uv.lock'])
                self.assertEqual(archive.extractfile('uv.lock').read(), blob.read_bytes())

    def test_template_export_creates_new_files_without_overwriting(self):
        from src.cli.config import main as export
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            path = Path(directory)/'nested/champion.json'
            self.assertEqual(export(['champion', '--output', str(path)]), 0)
            before = path.read_bytes()
            self.assertEqual(json.loads(before)['protocol'], 'champion')
            with self.assertRaises(SystemExit) as caught:
                export(['stage4', '--output', str(path)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
