"""Declared runtime dependency closure for reproducible experiments."""
from .manifest import capture_source

def source_scopes(root):
    files = []
    for directory in ('src/core', 'src/pipeline', 'src/plugins', 'src/analysis', 'src/metrics', 'src/experiments'):
        files.extend(path.relative_to(root).as_posix() for path in (root/directory).rglob('*.py')
                     if path.name not in ('formatting.py', 'champion_report.py'))
    files += ['src/__init__.py', 'src/config.py', 'src/calibrated_presets.py', 'src/pipelines.py',
              'pyproject.toml', 'uv.lock', 'scripts/21_benchmark_champion_unsupervised.py']
    files = sorted(set(files))
    return {'train': files, 'predict': files, 'evaluate': files,
            'report': [*(p.relative_to(root).as_posix() for p in (root/'src/reporting').glob('*.py')),
                       'src/experiments/champion_report.py', 'src/metrics/formatting.py', 'README.md',
                       'configs/experiments/champion_unsupervised.json']}

def capture_runtime_source(root, expected=None):
    scopes = ({stage: list(value['files']) for stage, value in expected['stages'].items()}
              if expected else source_scopes(root))
    return capture_source(root, scopes)
