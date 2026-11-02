"""Run with an isolated, non-editable wheel interpreter, outside the checkout."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys


def run(work, checkout, reference=None):
    import numpy as np
    from scipy.io import wavfile
    import src
    from src.cli import predict
    from src.experiments.manifest import sha256
    from src.experiments.store import verify_run
    from src.reporting.bundle import verify_report
    package = Path(src.__file__).resolve()
    assert checkout not in package.parents, package
    assert Path(sys.prefix).resolve() in package.parents, package
    assert not (work/'.git').exists()
    binary = Path(sys.executable).parent
    console = binary/('fac.exe' if os.name == 'nt' else 'fac')
    assert console.is_file()
    env = {**os.environ, 'PATH': str(binary), 'MPLBACKEND': 'Agg', 'PYTHONNOUSERSITE': '1'}
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)

    def command(*arguments, expected=0):
        value = subprocess.run([str(console), *map(str, arguments)], cwd=work, env=env,
                               capture_output=True, text=True, encoding='utf-8', timeout=120)
        assert value.returncode == expected, (arguments, value.returncode, value.stderr, value.stdout[-1000:])
        return value

    assert command('--version').stdout.strip() == '0.1.0'
    assert 'report' in command('--help').stdout
    config = work/'champion.json'
    command('config', 'champion', '--output', config)
    assert json.loads(config.read_bytes())['protocol'] == 'champion'
    assert 'protocol' in command('config', 'stage4').stdout
    refused = command('config', 'champion', '--output', config, expected=2)
    assert 'Traceback' not in refused.stderr

    records = []
    inputs = work/'inputs'
    inputs.mkdir()
    for rate in (16000, 44100):
        t = np.arange(round(rate*.42))/rate
        rng = np.random.default_rng(102)
        signal = .3*np.sin(2*np.pi*180*t)+.04*np.sin(2*np.pi*360*t)+rng.normal(0, .002, len(t))
        signal[:round(rate*.05)] = signal[-round(rate*.05):] = 0
        path = inputs/('tone_'+str(rate)+'.wav')
        wavfile.write(path, rate, (signal*32767).astype(np.int16))
        path.with_suffix('.lab').write_text('F0mean 1.8e2\n'+('F0std 10\n' if rate == 16000 else '')+'F0num 25\n', encoding='utf-8-sig')
        detectors = {**predict.get_champion_detectors(), **predict.get_baseline_detectors()}
        target = work/str(rate)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = predict.process_single_file(str(path), detectors, str(target), save_plot=False)
        with (target/(path.stem+'_pitch_prediction.csv')).open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        records.append({'rate': rate, 'metrics': metrics, 'contour_rows': rows})
    if reference:
        assert records == json.loads(reference.read_bytes())['cases'], 'Wheel prediction differs from pre-refactor capture'
    cli_prediction = work/'cli-prediction'
    command('predict', '-i', inputs, '-m', 'all', '-o', cli_prediction)
    with (cli_prediction/'evaluation_summary_metrics.csv').open(encoding='utf-8', newline='') as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 12
    assert any(row['Composite_Score_pct'] == '' for row in summary)
    assert len(list(cli_prediction.glob('*.png'))) == 2

    # Fit only after creating train; test does not exist at training time.
    train, test = work/'train', work/'test'
    train.mkdir()
    import shutil
    shutil.copyfile(inputs/'tone_16000.wav', train/'a.wav')
    (train/'a.lab').write_text('F0mean 180\nF0std 10\nF0num 25\n', encoding='utf-8')
    output = work/'outputs/runs/wheel-run'
    command('experiment', 'champion', 'train', '--train-dir', train, '--spec', config, '--output-dir', output)
    assert not test.exists()
    frozen = sha256(output/'frozen_models.json')
    test.mkdir()
    shutil.copyfile(inputs/'tone_44100.wav', test/'b.wav')
    (test/'b.lab').write_text('F0mean 180\nF0std 11\nF0num 25\n', encoding='utf-8')
    command('experiment', 'champion', 'evaluate', '--train-dir', train, '--test-dir', test, '--output-dir', output)
    manifest = verify_run(output)
    assert manifest['source']['git'] is None
    assert all(Path(name).is_absolute() is False for name in manifest['source']['stages']['train']['files'])
    assert len(json.loads((output/'metrics.json').read_bytes())['series']) == 72
    evaluation_sha = sha256(output/'evaluation.json')
    manifest_sha = sha256(output/'manifest.json')
    command('experiment', 'champion', 'train', '--train-dir', train, '--output-dir', output, '--resume')
    command('experiment', 'champion', 'evaluate', '--train-dir', train, '--test-dir', test, '--output-dir', output, '--resume')
    report = work/'outputs/reports/wheel-report'
    command('experiment', 'champion', 'report', '--output-dir', output, '--report-dir', report)
    report_info = json.loads(command('report', 'verify', report).stdout)
    assert report_info['source']['evaluation_sha256'] == evaluation_sha
    rebuilt = work/'outputs/reports/rebuilt'
    command('report', 'rebuild', report, '--output', rebuilt)
    assert verify_report(report)['report_id'] == verify_report(rebuilt)['report_id']
    assert (report/'tables.csv').read_bytes() == (rebuilt/'tables.csv').read_bytes()
    comparison = json.loads(command('compare', output, 'test/acf/gmm', output, 'test/acf/hmm').stdout)
    assert comparison['metric_version'] == 'legacy-v1'
    refused = command('compare', output, 'test/acf/baseline', output, 'test/acf/gmm', expected=2)
    assert 'feature_contract' in refused.stderr and 'Traceback' not in refused.stderr
    runs = json.loads(command('runs', '--root', work, 'list', '--json').stdout)['runs']
    assert len(runs) == 1 and runs[0]['status'] == 'verified', runs
    command('runs', '--root', work, 'verify', 'outputs/runs/wheel-run')
    command('report', 'index', '--root', work, '--output', work/'outputs/REPORT_INDEX.md')
    assert sha256(output/'manifest.json') == manifest_sha
    assert sha256(output/'evaluation.json') == evaluation_sha
    assert sha256(output/'frozen_models.json') == frozen
    assert verify_run(output)['phase'] == 'evaluated'
    from importlib.metadata import version
    return {'python': sys.version.split()[0], 'platform': sys.platform,
            'dependencies': {name: version(name) for name in ('numpy', 'scipy', 'matplotlib')},
            'prediction_cases': 12, 'pre_refactor_prediction_exact': reference is not None,
            'native_metric_series': 72, 'git_unavailable': True, 'source_outside_checkout': True,
            'train_without_test': True, 'resume_without_rewrite': True, 'report_rebuild_matched': True,
            'frozen_model_sha256': frozen, 'evaluation_sha256': evaluation_sha,
            'report_id': report_info['report_id'], 'metric_version': 'legacy-v1',
            'prediction_digest': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=False)
    os.chdir(args.workspace)
    result = run(args.workspace.resolve(), args.checkout.resolve(), args.reference)
    args.record.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
