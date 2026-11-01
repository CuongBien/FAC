"""Train/freeze, evaluate, and report eight decisions inside native Champions.

No preset is selected here. Primary models fit active train frames after the
Champion's native pre-processing. Two per-input adaptive controls are separate.
"""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.analysis.champion_unsupervised import (
    PRIMARY, ADAPTIVE, experiment_configuration,
    fit_champion_models, predict_decisions,
)
from src.analysis.protocols import threshold_labels
from src.analysis.validation import evaluate_result, aggregate
from src.pipeline.cache import PitchCache
from src.pipeline.champion import extract_file_features
from src.experiments.manifest import source_manifest, dataset_manifest, verify_dataset, sha256
from src.experiments.serialization import write_json
from src.experiments.legacy import freeze_json, read_frozen, read_evaluation_bundle
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot
from src.analysis.evaluation import format_metric
from src.pipelines import build_champion, build_baseline

METHODS = ('acf', 'amdf', 'cepstrum')


def save_frames(path, result, initial=None, decision_threshold=None, features=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = result if features is None else features
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame_index', 'time_sec', 'active', 'periodicity', 'zcr', 'candidate_f0',
                         'decision_threshold', 'initial_label', 'final_label', 'f0_hz'])
        for i, (time, label, f0) in enumerate(zip(result['frame_times'], result['labels'], result['f0_contour'])):
            writer.writerow([i, float(time), bool(fd.get('is_active', result['labels'] != 'sil')[i]),
                             float(fd['peak_values'][i]),
                             float(fd['zcr_vals'][i]) if 'zcr_vals' in fd else '',
                             float(fd['candidate_f0'][i]) if 'candidate_f0' in fd else '',
                             float(decision_threshold[i]) if decision_threshold is not None else '',
                             str(initial[i]) if initial is not None else '', str(label), float(f0)])


def train(args):
    folder = Path(args.output_dir)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Use a new/empty output directory; training artifacts are immutable')
    manifest = dataset_manifest(args.train_dir)
    paths = sorted(Path(args.train_dir).glob('*.wav'))
    settings = experiment_configuration()
    provenance = source_manifest(ROOT)
    write_source_snapshot(ROOT, folder/'training_source_snapshot.tar.gz', provenance)
    write_json(folder/'runtime.json', {'python': sys.version.split()[0],
        **{name: version(name) for name in ('numpy', 'scipy', 'matplotlib')},
        'snapshot_sha256': sha256(folder/'training_source_snapshot.tar.gz')})
    models = {}
    cache = PitchCache()
    for method in METHODS:
        features = []
        for path in paths:
            fd = extract_file_features(path, method, settings, cache=cache)
            features.append(fd)
            save_frames(folder/'train_features'/method/(path.stem+'.csv'), fd)
        models[method] = fit_champion_models(features, method, settings)
        print('TRAIN FIT', method, models[method]['num_train_active'], 'active frames', flush=True)
    # Check data/source did not change during fitting. No test directory is used.
    verify_dataset(args.train_dir, manifest)
    if source_manifest(ROOT)['source_sha256'] != provenance['source_sha256']:
        raise ValueError('Source changed during training; choose a new run directory')
    common = {'schema_version': 1, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'train_manifest': manifest, 'provenance': provenance}
    freeze_json(folder/'frozen_models.json', {**common, 'models': models})
    freeze_json(folder/'frozen_config.json', {**common, 'experiment': settings,
        'test_policy': 'Train-only fitting; no test-based selection. Test has historical exposure.',
        'selection': 'No new Champion or strategy is selected; all predefined strategies are reported.'})
    print('FROZEN before test evaluation:', folder, flush=True)


def checked_freezes(folder, train_dir):
    frozen = read_frozen(folder/'frozen_config.json')
    models = read_frozen(folder/'frozen_models.json')
    if frozen['train_manifest'] != models['train_manifest'] or frozen['provenance'] != models['provenance']:
        raise ValueError('Frozen config/models come from different training runs')
    verify_dataset(train_dir, frozen['train_manifest'])
    verify_source_snapshot(folder/'training_source_snapshot.tar.gz', frozen['provenance'])
    runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
    if runtime['snapshot_sha256'] != sha256(folder/'training_source_snapshot.tar.gz'):
        raise ValueError('Runtime/source snapshot mismatch')
    if experiment_configuration() != frozen['experiment']:
        raise ValueError('Native pipeline configuration changed after freeze')
    if source_manifest(ROOT)['source_sha256'] != frozen['provenance']['source_sha256']:
        raise ValueError('Source changed after freeze; rerun from the recorded snapshot or use a new run')
    return frozen, models


def evaluate(args):
    folder = Path(args.output_dir)
    if (folder/'evaluation.json').exists():
        raise FileExistsError('Evaluation exists; do not overwrite a recorded run')
    frozen, models = checked_freezes(folder, args.train_dir)
    settings = frozen['experiment']
    test_manifest = dataset_manifest(args.test_dir)
    pipelines, protocols = [], []
    comparison_count = 0
    cache = PitchCache()
    for dataset, data_dir in [('train', args.train_dir), ('test', args.test_dir)]:
        paths = sorted(Path(data_dir).glob('*.wav'))
        for method in METHODS:
            rows = {}
            for path in paths:
                fd = extract_file_features(path, method, settings, cache=cache)
                decisions, thresholds, adaptive_models = predict_decisions(
                    fd, models['models'][method], settings, include_input_adaptation=True)
                write_json(folder/'input_models'/dataset/method/(path.stem+'.json'), adaptive_models)
                control = threshold_labels(fd, method, settings['champions'][method]['threshold'])
                decisions = {'champion': control, **decisions}
                detector = build_champion(method, cache=cache)
                native = detector.process_file(str(path))
                for strategy, labels in decisions.items():
                    result = detector.process_feature_labels(fd, labels)
                    if strategy == 'champion':
                        for key in ('labels', 'frame_times', 'f0_contour'):
                            if not np.array_equal(result[key], native[key]):
                                raise AssertionError(f'Native Champion replay mismatch: {method}/{path.name}/{key}')
                        comparison_count += 1
                    row = evaluate_result(result, path)
                    row.update({'num_frames': result['num_frames'],
                                'initial_voiced': int(np.sum(labels == 'v')),
                                'energy_guard_threshold': detector.base_detector.threshold,
                                'post_added_voiced': int(np.sum((labels != 'v') & (result['labels'] == 'v'))),
                                'post_removed_voiced': int(np.sum((labels == 'v') & (result['labels'] != 'v')))})
                    contour = Path('contours')/dataset/method/strategy/(path.stem+'.csv')
                    save_frames(folder/contour, result, initial=labels,
                                decision_threshold=thresholds.get(strategy), features=fd)
                    row['contour_path'] = contour.as_posix()
                    rows.setdefault(strategy, []).append(row)
                baseline = build_baseline(method, cache=cache).process_file(str(path))
                row = evaluate_result(baseline, path)
                contour = Path('contours')/dataset/method/'baseline'/(path.stem+'.csv')
                save_frames(folder/contour, baseline)
                row.update({'num_frames': baseline['num_frames'], 'contour_path': contour.as_posix()})
                rows.setdefault('baseline', []).append(row)
            for strategy, records in rows.items():
                group = 'control' if strategy in ('baseline', 'champion') else (
                    'input_adaptive' if strategy in ADAPTIVE else 'train_fitted')
                item = {'dataset': dataset, 'method': method, 'strategy': strategy,
                        'group': group, 'metrics': aggregate(records), 'files': records}
                (pipelines if group == 'control' else protocols).append(item)
            print('EVALUATED', dataset, method, len(rows), 'strategies', flush=True)
    cache_record = dict(cache)
    verify_dataset(args.train_dir, frozen['train_manifest'])
    verify_dataset(args.test_dir, test_manifest)
    if source_manifest(ROOT)['source_sha256'] != frozen['provenance']['source_sha256']:
        raise ValueError('Source changed during evaluation; do not publish mixed-source results')
    result = {'config_sha256': sha256(folder/'frozen_config.json'),
              'models_sha256': sha256(folder/'frozen_models.json'),
              'provenance': source_manifest(ROOT), 'test_manifest': test_manifest,
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
              'pipelines': pipelines, 'protocols': protocols,
              'native_champion_replays_matched': comparison_count, 'cache': cache_record}
    write_json(folder/'evaluation.json', result)
    fields = ['dataset', 'method', 'strategy', 'group', 'file', 'pred_f0_mean', 'ref_f0_mean',
              'abs_error_mean', 'mape_f0', 'pred_f0_std', 'ref_f0_std', 'abs_error_std', 'mape_std',
              'pred_f0_num', 'ref_f0_num', 'abs_error_num', 'mape_num', 'final_score',
              'num_frames', 'initial_voiced', 'post_added_voiced', 'post_removed_voiced',
              'energy_guard_threshold', 'contour_path']
    with (folder/'per_file.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in pipelines+protocols:
            for row in item['files']:
                writer.writerow({key: (row.get(key) if key in row else item.get(key)) for key in fields})
    print('EVALUATION COMPLETE; no models or presets were changed', flush=True)


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(args.output_dir)
    frozen, _, result = read_evaluation_bundle(folder)
    verify_source_snapshot(folder/'training_source_snapshot.tar.gz', frozen['provenance'])
    all_rows = result['pipelines']+result['protocols']
    rows = {(r['dataset'], r['method'], r['strategy']): r for r in all_rows}
    names = ('baseline', 'champion')+PRIMARY+ADAPTIVE
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, dataset in zip(axes, ('train', 'test')):
        matrix = np.asarray([[rows[dataset, method, name]['metrics']['final_score'] for method in METHODS]
                             for name in names], dtype=float)
        plot = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(3)); ax.set_xticklabels([m.upper() for m in METHODS])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.axhline(1.5, c='black', lw=1); ax.axhline(9.5, c='black', lw=1)
        for i in range(len(names)):
            for j in range(3):
                value = matrix[i, j]
                ax.text(j, i, format_metric(value), ha='center', va='center', fontsize=9,
                        color='white' if np.isfinite(value) and value > np.nanmax(matrix)*.55 else 'black')
        ax.set_title(dataset.upper()+' Final Score (%); lower is better')
        fig.colorbar(plot, ax=ax, fraction=.045)
    fig.suptitle('Native Champion preprocessing + fixed Energy/Viterbi\n'
                 'Eight train-fitted decisions; input-adaptive controls below the second line')
    fig.tight_layout()
    figure = folder/'figures/score_matrix.png'
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure, dpi=180); plt.close(fig)
    lines = ['# Tám phương pháp không giám sát trong pipeline Champion', '',
             'Kết quả sinh tự động từ freeze và evaluation. Ngày chạy: '+result['evaluated_at_utc']+'.', '',
             '## Giao thức', '',
             '- Không chọn lại Champion. Fit tám mô hình từ các khung active của bốn file train, sau đúng BP/Clipping và khung/hop của Champion.',
             '- Median/Otsu/Valley/K-Means/GMM, chuẩn hóa/centroid 2D và Gaussian emission HMM được chốt trước test. HMM giữ A/pi cố định, reset qua silence và ranh giới file.',
             '- ZCR được tính trên frame pitch đã tiền xử lý và bỏ DC. Ngưỡng động dùng GMM train, cửa sổ 50 khung/alpha 0,30 trong từng đoạn active; có dùng ngữ cảnh tương lai, không phải bộ giải mã realtime.',
             '- Tất cả quyết định 30 ms dùng cùng silence gate, median, Energy, Viterbi. Ngưỡng periodicity của Energy cố định theo Champion; không đổi theo ngưỡng quyết định mới.',
             '- Hai dòng input_active fit 2D/HMM trên từng input, được lưu mô hình riêng. Chúng không phải kết quả suy luận thuần từ mô hình train.',
             '- Baseline là không plugin 25 ms; Champion và các dòng không giám sát dùng 30/10 ms. Final Score thấp hơn tốt hơn; chưa có nhãn thời gian để tính Accuracy/F1.',
             '- Test đã được quan sát trong lịch sử. Các thứ hạng dưới chỉ mô tả; không dùng để chọn lại preset hoặc suy ra tỷ lệ V/UV 50/50.', '',
             '## Đối sánh test', '', '| Nhóm | Phương pháp | ACF (%) | AMDF (%) | Cepstrum (%) |',
             '|---|---|---:|---:|---:|']
    for name in names:
        group = rows['test', 'acf', name]['group']
        values = [rows['test', m, name]['metrics']['final_score'] for m in METHODS]
        lines.append('| '+group+' | '+name+' | '+' | '.join(format_metric(v, '.3f') for v in values)+' |')
    lines += ['', '![Train và test](figures/score_matrix.png)', '', '## Tham số đã fit trên train', '',
              '| Thuật toán | Plugin | Frame/hop (ms) | Champion / Energy guard | Median | Otsu | Valley | K-Means | GMM |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    models = read_frozen(folder/'frozen_models.json')['models']
    for method in METHODS:
        spec, fit = frozen['experiment']['champions'][method], models[method]['thresholds']
        values = [fit[name] for name in ('median', 'otsu', 'valley', 'kmeans')]+[fit['gmm']['threshold']]
        lines.append(f"| {method.upper()} | {' + '.join(spec['plugins'])} | {spec['frame_duration_ms']:g}/{spec['hop_duration_ms']:g} | {spec['threshold']:g} | "+' | '.join(format_metric(v, '.6f') for v in values)+' |')
    count = sum(len(row['files']) for row in all_rows)
    lines += ['', '## Bằng chứng và giới hạn', '',
              f"- {count} kết quả từng file; {result['native_champion_replays_matched']} đối chứng Champion khớp từng label/time/F0 với đường chạy native.",
              '- [frozen_config.json](frozen_config.json), [frozen_models.json](frozen_models.json) và SHA sidecar được tạo bởi pha train trước khi đọc test.',
              '- [evaluation.json](evaluation.json), [per_file.csv](per_file.csv), `contours/`, `train_features/` và `input_models/` chứa đủ nguồn bảng và các quyết định trước/sau hậu xử lý.',
              '- `training_source_snapshot.tar.gz` và `runtime.json` giữ mã/phiên bản đã dùng. Evaluate từ chối nếu source, pipeline hoặc dữ liệu train thay đổi sau freeze.',
              '- Đây là giao thức mới có đặc trưng train/predict đồng nhất và Energy guard cố định. Không xem là tái lập giao thức chưa lưu của các mốc 14,31% / 14,40% / 2,17% trước đây.',
              '- Chỉ có bốn train/bốn test. Điểm mean/std/count không chứng minh đúng contour hoặc hết lỗi octave; chưa có test mới độc lập.', '',
              '```powershell',
              'uv run python -B scripts/21_benchmark_champion_unsupervised.py train --output-dir outputs/champion_unsupervised_recheck',
              'uv run python -B scripts/21_benchmark_champion_unsupervised.py evaluate --output-dir outputs/champion_unsupervised_recheck',
              'uv run python -B scripts/21_benchmark_champion_unsupervised.py report --output-dir outputs/champion_unsupervised_recheck',
              '```', '', 'Dùng thư mục mới khi chạy lại. Pha report chỉ ghi báo cáo/hình trong thư mục run.', '']
    (folder/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(folder/'report_manifest.json', {'config_sha256': result['config_sha256'],
        'models_sha256': result['models_sha256'], 'evaluation_sha256': sha256(folder/'evaluation.json'),
        'per_file_sha256': sha256(folder/'per_file.csv'), 'report_sha256': sha256(folder/'README.md'),
        'figure_sha256': sha256(figure), 'file_results': count})
    print('REPORT:', folder/'README.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate', 'report'))
    parser.add_argument('--train-dir', default=str(ROOT/'TinHieuHuanLuyen'))
    parser.add_argument('--test-dir', default=str(ROOT/'TinHieuKiemThu'))
    parser.add_argument('--output-dir', default=str(ROOT/'outputs/champion_unsupervised_2026-11-01'))
    args = parser.parse_args()
    {'train': train, 'evaluate': evaluate, 'report': report}[args.phase](args)


if __name__ == '__main__':
    main()
