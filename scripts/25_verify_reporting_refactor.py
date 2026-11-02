"""Compare report tables with the pre-refactor capture; never rerun a detector."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.manifest import source_manifest, sha256
from src.experiments.serialization import json_bytes
from src.experiments.snapshots import write_source_snapshot
from src.experiments.registry import RunRegistry
from src.reporting.bundle import create_report, rebuild_report, verify_report
from src.reporting.publish import preview, apply, CHAPTERS, README_MARKERS, REPORT_MARKERS, CHAPTER_MARKERS


def hashes(folder):
    return {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob('*') if p.is_file()}


def tables(text):
    return [line for line in text.splitlines() if line.startswith('|')]


def expected_section(reference, name, markers):
    original = subprocess.check_output(['git', 'show', reference['base_commit']+':'+name], cwd=ROOT).decode('utf-8')
    before = tables(original.split(markers[0])[0])
    after = tables(original.split(markers[1])[1])
    recorded = reference['tables'][name]
    assert recorded[:len(before)] == before
    if after:
        assert recorded[-len(after):] == after
    return recorded[len(before):len(recorded)-len(after) if after else None]


def verify(output=None):
    reference_path = ROOT/'outputs/refactor_phase5/reference.json'
    reference = json.loads(reference_path.read_bytes())
    provenance = source_manifest(ROOT)
    old_folders = [ROOT/'outputs/stage4_validation', ROOT/'outputs/champion_unsupervised_2026-11-01']
    before = [hashes(folder) for folder in old_folders]
    owned = {'README.md': README_MARKERS, 'docs/STAGE4_RESULTS.md': REPORT_MARKERS,
             **{'docs/'+name: CHAPTER_MARKERS for name in CHAPTERS}}
    documents_before = {name: (ROOT/name).read_bytes() for name in owned}
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        project = temp/'project'
        project.mkdir()
        shutil.copytree(ROOT/'docs', project/'docs')
        shutil.copyfile(ROOT/'README.md', project/'README.md')
        stage = create_report(old_folders[0], project/'outputs/reports/stage4')
        champion = create_report(old_folders[1], project/'outputs/reports/champion')
        sections = json.loads((stage/'sections.json').read_bytes())
        count = 0
        for name, markers in owned.items():
            expected = expected_section(reference, name, markers)
            assert tables(sections[name]) == expected, name
            count += len(expected)
        assert tables((champion/'README.md').read_text(encoding='utf-8')) == reference['tables']['champion']
        count += len(reference['tables']['champion'])
        assert json.loads((stage/'summary.json').read_bytes()) == reference['stage4_summary']
        for bundle in (stage, champion):
            rebuilt = rebuild_report(bundle, temp/('rebuilt-'+bundle.name))
            for name in ('README.md', 'tables.csv', 'sections.json', 'view.json'):
                assert (bundle/name).read_bytes() == (rebuilt/name).read_bytes(), name
            assert verify_report(bundle)['report_id'] == verify_report(rebuilt)['report_id']
        code = temp/'archived-renderer'
        code.mkdir()
        with tarfile.open(champion/'renderer_source.tar.gz') as archive:
            archive.extractall(code, filter='data')
        detached = temp/'archived-code-rebuild'
        completed = subprocess.run([sys.executable, '-B', '-m', 'src.reporting', 'rebuild', str(champion),
                                    '--output', str(detached)], cwd=code, env={**os.environ, 'PYTHONPATH': str(code)},
                                   capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        assert verify_report(detached)['report_id'] == verify_report(champion)['report_id']
        for name in ('README.md', 'tables.csv', 'sections.json', 'view.json'):
            assert (detached/name).read_bytes() == (champion/name).read_bytes(), name
        plan = preview(stage, project)
        diff = plan.diff()
        originals = {name: (project/name).read_bytes() for name in owned}
        apply(plan)
        for name, (start, end) in owned.items():
            published = (project/name).read_bytes()
            assert published.split(start.encode())[0] == originals[name].split(start.encode())[0]
            assert published.split(end.encode())[1] == originals[name].split(end.encode())[1]
        assert preview(stage, project).diff() == ''
        # No snapshots/input copies in the report are mistaken for extra runs.
        assert RunRegistry(project).scan() == []
        bundles = {}
        if output:
            output.mkdir(parents=True, exist_ok=True)
            required = ['validation.json', 'source_snapshot.tar.gz', 'stage4_report.tar.gz', 'champion_report.tar.gz']
            if any((output/name).exists() for name in required):
                raise FileExistsError('Use a fresh proof output; existing evidence is immutable')
            gallery = output/'figures'
            gallery.mkdir(exist_ok=True)
        for name, bundle in (('stage4', stage), ('champion', champion)):
            manifest = verify_report(bundle)
            bundles[name] = {'report_id': manifest['report_id'], 'source': manifest['source'],
                             'artifact_count': len(manifest['artifacts']), 'manifest_sha256': sha256(bundle/'report_manifest.json')}
            if output:
                archive_path = output/(name+'_report.tar.gz')
                with tarfile.open(archive_path, 'w:gz') as archive:
                    for path in sorted(bundle.rglob('*')):
                        if path.is_file():
                            archive.add(path, arcname=name+'/'+path.relative_to(bundle).as_posix(), recursive=False)
                bundles[name]['archive_sha256'] = sha256(archive_path)
                for image in (bundle/'figures').glob('*.png'):
                    shutil.copyfile(image, gallery/image.name)
        if output:
            (output/'publication_preview.diff').write_bytes(diff.encode('utf-8'))
    assert [hashes(folder) for folder in old_folders] == before
    assert {name: (ROOT/name).read_bytes() for name in owned} == documents_before
    with tarfile.open(ROOT/'outputs/refactor_phase4/source_snapshot.tar.gz') as archive:
        old = ast.parse(archive.extractfile('src/visualization/plotter.py').read())
    new = ast.parse((ROOT/'src/reporting/plots.py').read_bytes())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert functions(old) == functions(new)
    paths = [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/test_*.py'), ROOT/'predict.py']
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 9))
    assert source_manifest(ROOT)['source_sha256'] == provenance['source_sha256']
    proof = {'schema_version': 1, 'reference_sha256': sha256(reference_path), 'source_provenance': provenance,
             'table_lines_matched': count, 'stage4_summary_matched': True, 'portable_rebuild_matched': True,
             'archived_renderer_outside_git_rebuild_matched': True,
             'publication_idempotent': True, 'manual_regions_byte_unchanged': True, 'legacy_artifacts_unchanged': True,
             'plot_functions_ast_unchanged': len(functions(old)), 'python_39_ast_files': len(paths), 'bundles': bundles,
             'test_source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'tests').glob('test_*.py'))},
             'scope': 'Stored metrics/trials only; no detector, model fit, preset selection or legacy artifact rewrite.'}
    if output:
        snapshot = output/'source_snapshot.tar.gz'
        proof['source_snapshot_files'] = write_source_snapshot(ROOT, snapshot, provenance)
        proof['source_snapshot_sha256'] = sha256(snapshot)
        (output/'validation.json').write_bytes(json_bytes(proof))
    print(json.dumps({key: value for key, value in proof.items() if key not in ('source_provenance', 'test_source_sha256')}, indent=2))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Explicit fresh proof directory; default is read-only verification')
    args = parser.parse_args()
    verify(args.output)


if __name__ == '__main__':
    main()
