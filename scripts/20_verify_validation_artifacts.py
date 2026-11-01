"""Verify Stage 4 freezes, evaluation links, exported presets and source archive.

--repair-source-snapshot preserves the old evidence before reconstructing exact
recorded bytes. No training, selection or evaluation result is modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrated_presets import CHAMPIONS, FROZEN_CONFIG_SHA256
from src.analysis.validation import (
    read_evaluation_bundle, sha256, verify_source_snapshot, write_source_snapshot, write_json,
    source_manifest,
)


def verify(folder, check_presets=True):
    folder = Path(folder)
    frozen, _, result = read_evaluation_bundle(folder)
    count = verify_source_snapshot(folder/'training_source_snapshot.tar.gz', frozen['provenance'])
    runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
    if runtime['snapshot_sha256'] != sha256(folder/'training_source_snapshot.tar.gz'):
        raise ValueError('Runtime record uses a different source snapshot')
    if check_presets:
        if FROZEN_CONFIG_SHA256 != sha256(folder/'frozen_config.json'):
            raise ValueError('Exported preset checksum differs from frozen config')
        selected = {method: {key: value for key, value in spec.items() if key != 'method'}
                    for method, spec in frozen['selected'].items()}
        if selected != CHAMPIONS:
            raise ValueError('Exported preset values differ from frozen selection')
    return {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': sha256(folder/'frozen_config.json'),
        'models_sha256': sha256(folder/'frozen_models.json'),
        'evaluation_matches_freezes': True,
        'training_source_snapshot_verified': True,
        'snapshot_source_files': count,
        'runtime_snapshot_checksum_matches': True,
        'freeze_and_export_match': True if check_presets else None,
        'evaluation_pipeline_file_pairs': sum(len(row['files']) for row in result['pipelines']),
        'evaluation_protocol_file_pairs': sum(len(row['files']) for row in result['protocols']),
        'scope': 'Artifact integrity; historical scores are not a rerun of the current algorithm.',
        'verification_source_provenance': source_manifest(ROOT),
    }


def repair_snapshot(folder):
    folder = Path(folder)
    frozen, _, _ = read_evaluation_bundle(folder)
    snapshot = folder/'training_source_snapshot.tar.gz'
    try:
        verify_source_snapshot(snapshot, frozen['provenance'])
        return
    except ValueError:
        pass
    backup = folder/'historical_source_snapshot'
    backup.mkdir(exist_ok=True)
    for name in ('training_source_snapshot.tar.gz', 'runtime.json', 'validation.json'):
        source, target = folder/name, backup/name
        if target.exists():
            if sha256(source) != sha256(target):
                raise FileExistsError(f'Historical evidence already exists: {target}')
        else:
            shutil.copy2(source, target)
    # Validate all recorded source bytes before replacing the inconsistent archive.
    replacement = folder/'training_source_snapshot_rebuilt.tar.gz'
    write_source_snapshot(ROOT, replacement, frozen['provenance'])
    replacement.replace(snapshot)
    runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
    runtime.update({
        'snapshot_sha256': sha256(snapshot),
        'snapshot_reconstruction': {
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_revision': frozen['provenance']['base_commit'],
            'method': 'Recorded SHA-256 matched against working tree or Git bytes (LF/CRLF).',
            'previous_snapshot_sha256': sha256(backup/'training_source_snapshot.tar.gz'),
            'note': 'Source evidence repaired; no training or score was recomputed.',
        },
    })
    write_json(folder/'runtime.json', runtime)
    (backup/'README.md').write_text(
        '# Historical source evidence\n\n'
        'These files belong to the 30 October run. The 31 October regeneration changed '
        'freezes and provenance but retained this older snapshot/runtime/validation. '
        'They are preserved here for history and do not validate the regenerated freezes.\n',
        encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/stage4_validation')
    parser.add_argument('--repair-source-snapshot', action='store_true')
    parser.add_argument('--write-report', action='store_true')
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args()
    if args.repair_source_snapshot:
        repair_snapshot(args.output_dir)
    report = verify(args.output_dir)
    if args.run_tests:
        run = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-q'],
                             cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        if run.returncode:
            raise SystemExit(run.returncode)
        match = re.search(r'Ran (\d+) tests?', run.stderr)
        if not match:
            raise ValueError('Cannot determine actual regression test count')
        report['total_regression_tests'] = int(match.group(1))
        report['regression_tests_passed'] = True
    if args.write_report:
        write_json(args.output_dir/'validation.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
