"""Build sdist/wheel, install into a clean venv and exercise public commands."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.manifest import source_manifest, sha256
from src.experiments.serialization import json_bytes
from src.experiments.snapshots import write_source_snapshot


def execute(arguments, *, cwd=ROOT, env=None):
    value = subprocess.run(list(map(str, arguments)), cwd=cwd, env=env, capture_output=True,
                           text=True, encoding='utf-8', errors='replace', timeout=300)
    if value.returncode:
        raise RuntimeError('Command failed: '+' '.join(map(str, arguments))+'\n'+value.stderr[-5000:]+'\n'+value.stdout[-1000:])
    return value


def verify(output=None, *, python=None, offline=False, reference=True):
    if output and output.exists() and any((output/name).exists() for name in ('validation.json', 'source_snapshot.tar.gz', 'wheel_smoke.json')):
        raise FileExistsError('Use a new proof directory; existing evidence is immutable')
    provenance = source_manifest(ROOT)
    names = ('pyproject.toml', 'setup.py', 'build_support.py', 'MANIFEST.in', 'uv.lock', 'README.md',
             'configs/experiments/champion_unsupervised.json', 'configs/experiments/stage4_calibration.json')
    build_inputs = {name: sha256(ROOT/name) for name in names}
    test_inputs = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'tests').glob('*.py'))}
    optional = ['--offline'] if offline else []
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        distributions = work/'dist'
        # Default uv build makes the wheel from the sdist, checking both layouts.
        execute(['uv', 'build', '--out-dir', distributions, *optional])
        wheel = next(distributions.glob('*.whl'))
        source = next(distributions.glob('*.tar.gz'))
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            assert not any(name.startswith(('outputs/', 'TinHieu', 'tests/', 'archive/')) for name in members)
            index = json.loads(archive.read('src/_runtime/manifest.json'))
            assert len(index['files']) == 7
            for name, item in index['files'].items():
                content = archive.read('src/_runtime/'+item['path'])
                assert hashlib.sha256(content).hexdigest() == item['sha256']
                assert content == (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
        requirements = work/'locked-dependencies.txt'
        execute(['uv', 'export', '--locked', '--no-emit-project', '--format', 'requirements-txt',
                 '--output-file', requirements, *optional])
        environment = work/'venv'
        execute(['uv', 'venv', '--python', python or sys.executable, environment, *optional])
        interpreter = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        execute(['uv', 'pip', 'install', '--python', interpreter, '--require-hashes', '-r', requirements, *optional])
        execute(['uv', 'pip', 'install', '--python', interpreter, '--no-deps', wheel, *optional])
        record = work/'wheel_smoke.json'
        arguments = [interpreter, '-I', '-B', ROOT/'tests/wheel_smoke.py', '--workspace', work/'outside',
                     '--checkout', ROOT, '--record', record]
        if reference:
            arguments += ['--reference', ROOT/'outputs/refactor_phase6/reference.json']
        env = {**os.environ, 'MPLBACKEND': 'Agg'}
        env.pop('PYTHONPATH', None)
        execute(arguments, cwd=work, env=env)
        smoke = json.loads(record.read_bytes())
        # Validate the documented checkout command too, without modifying the
        # user's .venv or allowing uv.lock to change.
        editable = work/'editable'
        execute(['uv', 'sync', '--locked', '--python', python or sys.executable, *optional],
                env={**os.environ, 'UV_PROJECT_ENVIRONMENT': str(editable)})
        editable_bin = editable/('Scripts' if os.name == 'nt' else 'bin')
        assert execute([editable_bin/('fac.exe' if os.name == 'nt' else 'fac'), '--version'], cwd=work).stdout.strip() == '0.1.0'
        proof = {'schema_version': 1, 'scope': 'Clean non-editable wheel; public CLI outside Git/checkout; no historical artifact writes.',
                 'source_provenance': provenance, 'build_input_sha256': build_inputs, 'test_source_sha256': test_inputs,
                 'wheel': {'name': wheel.name, 'sha256': sha256(wheel), 'members': len(members), 'support_files': 7},
                 'sdist': {'name': source.name, 'sha256': sha256(source)}, 'smoke': smoke,
                 'uv_version': execute(['uv', '--version']).stdout.strip(), 'lock_sha256': sha256(ROOT/'uv.lock'),
                 'locked_editable_install_checked': True}
        for path in [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'), ROOT/'predict.py', ROOT/'setup.py', ROOT/'build_support.py']:
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 9))
        assert source_manifest(ROOT)['source_sha256'] == provenance['source_sha256']
        assert all(sha256(ROOT/name) == digest for name, digest in build_inputs.items())
        assert all(sha256(ROOT/name) == digest for name, digest in test_inputs.items())
        if output:
            output.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wheel, output/wheel.name)
            shutil.copyfile(source, output/source.name)
            shutil.copyfile(record, output/'wheel_smoke.json')
            shutil.copyfile(requirements, output/'locked-dependencies.txt')
            sources = {**provenance, 'source_sha256': {**provenance['source_sha256'], **build_inputs}}
            snapshot = output/'source_snapshot.tar.gz'
            proof['source_snapshot_files'] = write_source_snapshot(ROOT, snapshot, sources)
            proof['source_snapshot_sha256'] = sha256(snapshot)
            (output/'validation.json').write_bytes(json_bytes(proof))
        print(json.dumps({key: proof[key] for key in ('wheel', 'sdist', 'smoke', 'uv_version')}, indent=2))
        return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--python', help='Python interpreter/version for the clean venv; default is current Python')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--skip-reference', action='store_true', help='For cross-runtime CI: exercise behavior without exact same-environment golden comparison')
    args = parser.parse_args()
    verify(args.output, python=args.python, offline=args.offline, reference=not args.skip_reference)
