"""Compatibility renderer retained until the reporting refactor."""
from pathlib import Path
import numpy as np
from src.analysis.champion_unsupervised import PRIMARY, ADAPTIVE
from src.analysis.evaluation import format_metric
from src.experiments.legacy import read_evaluation_bundle, read_frozen
from src.experiments.snapshots import verify_source_snapshot
from src.experiments.manifest import sha256
from src.experiments.serialization import write_json

METHODS = ('acf', 'amdf', 'cepstrum')

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
