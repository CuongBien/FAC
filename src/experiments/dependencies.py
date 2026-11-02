"""Declared conservative runtime dependencies, separated from the renderer."""
from .manifest import capture_source


def source_scopes(root):
    files = []
    for directory in ('src/core', 'src/pipeline', 'src/plugins', 'src/metrics'):
        files.extend(path.relative_to(root).as_posix() for path in (root/directory).rglob('*.py')
                     if path != root/'src/metrics/formatting.py')
    files += ['src/__init__.py', 'src/config.py', 'src/calibrated_presets.py', 'src/pipelines.py',
              'src/analysis/__init__.py', 'src/analysis/threshold.py', 'src/analysis/evaluation.py',
              'src/analysis/validation.py', 'src/analysis/unsupervised_threshold.py',
              'src/analysis/advanced_unsupervised.py', 'src/analysis/protocols.py',
              'src/analysis/champion_unsupervised.py', 'pyproject.toml', 'uv.lock',
              'scripts/17_validate_and_calibrate.py', 'scripts/21_benchmark_champion_unsupervised.py']
    files += [path.relative_to(root).as_posix() for path in (root/'src/experiments').glob('*.py')
              if path.name != 'champion_report.py']
    files = sorted(set(files))
    return {'train': files, 'predict': files, 'evaluate': files,
            # Snapshot support files are outside scientific fingerprints: the
            # actual executable spec is already pinned as config.json per run.
            'report': ['src/experiments/champion_report.py', 'src/metrics/formatting.py', 'README.md',
                       'configs/experiments/champion_unsupervised.json',
                       'configs/experiments/stage4_calibration.json']}


def capture_runtime_source(root, expected=None):
    scopes = ({stage: list(value['files']) for stage, value in expected['stages'].items()}
              if expected else source_scopes(root))
    return capture_source(root, scopes)
