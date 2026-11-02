"""Immutable reports, detached rendering and marker publication failure cases."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.experiments.manifest import sha256
from src.experiments.registry import RunRegistry
from src.experiments.runner import RunService
from src.experiments.serialization import json_bytes
from src.reporting.bundle import create_report, verify_report
from src.reporting.catalog import catalogue
from src.reporting.publish import preview, apply, CHAPTERS, README_MARKERS, REPORT_MARKERS, CHAPTER_MARKERS
from src.reporting.source import load_source
from src.reporting.view import ReportView

ROOT = Path(__file__).resolve().parents[1]


def files(folder):
    return {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob('*') if p.is_file()}


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.stage = create_report(ROOT/'outputs/stage4_validation', cls.root/'stage')
        cls.champion = create_report(ROOT/'outputs/champion_unsupervised_2026-11-01', cls.root/'champion')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def documents(self, root):
        (root/'docs').mkdir(parents=True)
        owned = {'README.md': README_MARKERS, 'docs/STAGE4_RESULTS.md': REPORT_MARKERS,
                 **{'docs/'+name: CHAPTER_MARKERS for name in CHAPTERS}}
        for name, (start, end) in owned.items():
            (root/name).write_bytes(b'\xef\xbb\xbfManual before\r\n'+start.encode()+b'\r\nold\r\n'+end.encode()+
                                    '\r\nManual after; tám phương pháp.\r\n'.encode('utf-8'))
        return owned

    def test_render_reuses_identical_revision_without_rewriting_source_or_bundle(self):
        source = ROOT/'outputs/stage4_validation'
        before_source, before_report = files(source), files(self.stage)
        with patch('src.reporting.stage4.render', side_effect=AssertionError('rerender')):
            self.assertEqual(create_report(source, self.stage), self.stage)
        self.assertEqual(files(source), before_source)
        self.assertEqual(files(self.stage), before_report)

    def test_new_destination_has_identical_tables_and_portable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = create_report(ROOT/'outputs/stage4_validation', Path(directory)/'new location')
            for name in ('README.md', 'sections.json', 'tables.csv', 'view.json', 'summary.json'):
                self.assertEqual((bundle/name).read_bytes(), (self.stage/name).read_bytes())
            self.assertEqual(verify_report(bundle)['report_id'], verify_report(self.stage)['report_id'])

    def test_view_is_immutable_and_rejects_duplicate_rows(self):
        document = json.loads((self.stage/'view.json').read_bytes())
        view = ReportView(document)
        document['evaluation']['pipelines'][0]['metrics']['final_score'] = 9999
        exported = view.to_dict()
        exported['source']['run_id'] = 'changed'
        self.assertNotEqual(view.to_dict()['evaluation'], document['evaluation'])
        self.assertNotEqual(view.to_dict()['source']['run_id'], 'changed')
        duplicate = view.to_dict()
        duplicate['evaluation']['pipelines'].append(deepcopy(duplicate['evaluation']['pipelines'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ReportView(duplicate)

    def test_portable_bundle_rebuild_needs_no_original_run_or_dataset(self):
        from src.reporting.bundle import rebuild_report
        with tempfile.TemporaryDirectory() as directory, patch('src.reporting.bundle.load_source', side_effect=AssertionError('original run read')):
            output = rebuild_report(self.champion, Path(directory)/'detached')
            for name in ('README.md', 'view.json', 'tables.csv', 'sections.json'):
                self.assertEqual((output/name).read_bytes(), (self.champion/name).read_bytes())
            self.assertEqual(verify_report(output)['report_id'], verify_report(self.champion)['report_id'])
        # Execute the archived renderer itself outside the checkout. This also
        # checks package __init__ imports in the recorded dependency closure.
        import tarfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root/'code'
            code.mkdir()
            with tarfile.open(self.champion/'renderer_source.tar.gz') as archive:
                archive.extractall(code, filter='data')
            result = subprocess.run([sys.executable, '-B', '-m', 'src.reporting', 'rebuild', str(self.champion),
                                     '--output', str(root/'rebuilt')], cwd=code,
                                    env={**os.environ, 'PYTHONPATH': str(code)}, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(verify_report(root/'rebuilt')['report_id'], verify_report(self.champion)['report_id'])
            self.assertEqual((root/'rebuilt/tables.csv').read_bytes(), (self.champion/'tables.csv').read_bytes())

    def test_failed_render_does_not_publish_a_partial_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'failed'
            with patch('src.reporting.stage4.render', side_effect=OSError('render failed')), self.assertRaises(OSError):
                create_report(ROOT/'outputs/stage4_validation', output)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(output.parent.glob('.failed.*.tmp'))), 1)

    def test_detached_renderer_and_cli_do_not_import_or_execute_detectors(self):
        code = '''
import importlib.abc, sys, tempfile
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('src.core', 'src.analysis.champion_unsupervised', 'src.analysis.protocols', 'src.experiments.champion', 'src.experiments.calibration')):
            raise AssertionError('scientific module imported: '+fullname)
sys.meta_path.insert(0, Block())
from src.reporting.bundle import create_report
with tempfile.TemporaryDirectory() as directory:
    create_report(Path(sys.argv[1]), Path(directory)/'report')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT/'outputs/champion_unsupervised_2026-11-01')],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = [sys.executable, '-B', '-m', 'src.reporting', 'verify', str(self.champion)]
        result = subprocess.run(command, cwd=self.root, env={**os.environ, 'PYTHONPATH': str(ROOT)},
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['format'], 'fac.report.v1')

    def test_report_service_does_not_require_scientific_source_runtime_or_datasets(self):
        with patch.object(RunService, '_inputs', side_effect=AssertionError('WAV/source/runtime read')):
            result = RunService(ROOT/'outputs/stage4_validation').report(output=self.stage)
        self.assertEqual(result, self.stage)

    def test_bundle_detects_tampered_or_extra_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)/'copy'
            shutil.copytree(self.stage, copy)
            (copy/'tables.csv').write_text('fake')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_report(copy)
            shutil.copyfile(self.stage/'tables.csv', copy/'tables.csv')
            (copy/'extra.txt').write_text('unregistered')
            with self.assertRaisesRegex(ValueError, 'population'):
                verify_report(copy)

    def test_portable_bundle_checks_local_links_even_if_output_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)/'copy'
            shutil.copytree(self.stage, copy)
            with (copy/'README.md').open('a', encoding='utf-8') as stream:
                stream.write('\n[missing](inputs/missing.json)\n')
            manifest = json.loads((copy/'report_manifest.json').read_bytes())
            manifest['artifacts']['README.md'] = sha256(copy/'README.md')
            (copy/'report_manifest.json').write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, 'Broken report link'):
                verify_report(copy)

    def test_changed_renderer_creates_new_revision_and_refuses_overwrite(self):
        from src.reporting.bundle import renderer_record
        changed = renderer_record()
        changed['canonical_sha256'] = '0'*64
        with patch('src.reporting.bundle.renderer_record', return_value=changed):
            with self.assertRaisesRegex(FileExistsError, 'changed'):
                create_report(ROOT/'outputs/stage4_validation', self.stage)

    def test_preview_preserves_manual_bytes_rebases_links_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = self.documents(root)
            before = {name: (root/name).read_bytes() for name in owned}
            plan = preview(self.stage, root)
            self.assertTrue(plan.diff())
            self.assertNotIn('\r', plan.diff())
            self.assertEqual(before, {name: (root/name).read_bytes() for name in owned})
            apply(plan)
            for name, (start, end) in owned.items():
                after = (root/name).read_bytes()
                self.assertEqual(after.split(start.encode())[0], before[name].split(start.encode())[0])
                self.assertEqual(after.split(end.encode())[1], before[name].split(end.encode())[1])
                self.assertIn(b'evaluation.json)', after)
                self.assertNotIn(b'](inputs/', after)
            self.assertEqual(preview(self.stage, root).diff(), '')

    def test_last_invalid_marker_prevents_every_document_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = self.documents(root)
            last = root/('docs/'+CHAPTERS[-1])
            last.write_bytes(last.read_bytes().replace(CHAPTER_MARKERS[1].encode(), b''))
            before = {name: (root/name).read_bytes() for name in owned}
            with self.assertRaisesRegex(ValueError, 'generated block'):
                preview(self.stage, root)
            self.assertEqual(before, {name: (root/name).read_bytes() for name in owned})

    def test_stale_preview_rejects_all_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = self.documents(root)
            plan = preview(self.stage, root)
            with (root/'README.md').open('ab') as stream:
                stream.write(b'new manual text')
            before = {name: (root/name).read_bytes() for name in owned}
            with self.assertRaisesRegex(ValueError, 'changed'):
                apply(plan)
            self.assertEqual(before, {name: (root/name).read_bytes() for name in owned})

    def test_io_failure_rolls_back_documents_already_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = self.documents(root)
            plan = preview(self.stage, root)
            before = {name: (root/name).read_bytes() for name in owned}
            replace = os.replace
            def fail(source, target):
                if Path(target) == root/'docs/STAGE4_RESULTS.md':
                    raise OSError('publication failure')
                return replace(source, target)
            with patch('src.reporting.publish.os.replace', side_effect=fail), self.assertRaises(OSError):
                apply(plan)
            self.assertEqual(before, {name: (root/name).read_bytes() for name in owned})
            self.assertFalse(list(root.rglob('*.tmp')))

    def test_missing_metrics_render_as_na_in_both_protocols(self):
        for source, module in ((self.stage, 'stage4'), (self.champion, 'champion')):
            data = json.loads((source/'view.json').read_bytes())
            for row in data['evaluation']['pipelines']+data['evaluation']['protocols']:
                row['metrics']['final_score'] = None
                for file in row['files']:
                    file['final_score'] = None
            with tempfile.TemporaryDirectory() as directory:
                from importlib import import_module
                import_module('src.reporting.'+module).render(ReportView(data), Path(directory))
                self.assertIn('N/A', (Path(directory)/'README.md').read_text(encoding='utf-8'))

    def test_registry_ignores_copied_freezes_and_catalogue_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'outputs/legacy'
            shutil.copytree(self.stage/'inputs', run)
            shutil.copytree(self.stage, run/'reports/revision')
            before = files(run)
            entries = RunRegistry(root).scan()
            self.assertEqual(len(entries), 1, entries)
            self.assertEqual(entries[0]['status'], 'verified')
            text = catalogue(root, root/'outputs/REPORTS.md')
            self.assertIn('report-', text)
            self.assertIn('reports/revision/README.md', text)
            self.assertEqual(files(run), before)
            with self.assertRaisesRegex(ValueError, 'managed run'):
                catalogue(root, run/'README.md')

    def test_plotting_functions_are_unchanged_and_old_imports_are_compatible(self):
        import tarfile
        with tarfile.open(ROOT/'outputs/refactor_phase4/source_snapshot.tar.gz') as archive:
            before = ast.parse(archive.extractfile('src/visualization/plotter.py').read())
        after = ast.parse((ROOT/'src/reporting/plots.py').read_text(encoding='utf-8'))
        funcs = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(funcs(before), funcs(after))
        from src.visualization import plotter
        from src.reporting import plots
        for name in funcs(before):
            self.assertIs(getattr(plotter, name), getattr(plots, name))


if __name__ == '__main__':
    unittest.main()
