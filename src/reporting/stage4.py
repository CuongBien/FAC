"""Stage 4 tables/figures; only stored trials and evaluations are consumed."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.experiments.serialization import write_json
from .common import source_note, stamp_figure, finite_max
from src.metrics.formatting import format_metric
from .publish import CHAPTERS


def render(view, folder):
    data = view.to_dict()
    frozen, result = data['frozen'], data['evaluation']
    pipelines = {(r['dataset'], r['method'], r['strategy']): r for r in result['pipelines']}
    protocols = {(r['dataset'], r['method'], r['strategy']): r for r in result['protocols']}
    methods = ['acf', 'amdf', 'cepstrum']
    folder = Path(folder)
    figures = folder/'figures'; figures.mkdir(exist_ok=True)
    sections = {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, split in zip(axes, ['train', 'test']):
        for i, (strategy, label) in enumerate([('baseline', 'Historical baseline'),
                                               ('historical_champion', 'Before calibration'),
                                               ('selected_champion', 'Frozen train selection')]):
            values = [pipelines[split, m, strategy]['metrics']['final_score'] for m in methods]
            bars = ax.bar(np.arange(3)+(i-1)*.25, [np.nan if v is None else v for v in values], width=.25, label=label)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, (value if value is not None else 0)+.4, format_metric(value, '.3f'), ha='center', fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylabel('Final Score (%); lower is better')
        ax.set_title(split.upper()); ax.set_ylim(0, max(57, finite_max([r['metrics']['final_score'] for r in result['pipelines']])*1.15))
        ax.grid(axis='y', alpha=.25)
    axes[0].legend(fontsize=8)
    stamp_figure(fig, data); fig.tight_layout(); fig.savefig(figures/'pipelines_train_test.png', dpi=180); plt.close(fig)

    names = [row['strategy'] for row in result['protocols'] if row['dataset']=='test' and row['method']=='acf']
    matrix = np.array([[protocols['test', m, name]['metrics']['final_score'] for m in methods] for name in names], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels([m.upper() for m in methods])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    for row in range(len(names)):
        for column in range(3):
            ax.text(column, row, format_metric(matrix[row,column]), ha='center', va='center', fontsize=9,
                    color='white' if matrix[row,column] > finite_max(matrix.flat)*.6 else 'black')
    ax.set_title('TEST: fixed train models vs per-input adaptation\n25 ms frames / 10 ms hop, common silence gate and segment median')
    fig.colorbar(image, ax=ax, label='Final Score (%); lower is better')
    stamp_figure(fig, data); fig.tight_layout(); fig.savefig(figures/'protocols_test.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, method in zip(axes, methods):
        trials = data['extras']['train/'+method+'_trials.json']
        selected = frozen['selected'][method]
        for frame in frozen['search_space']['frame_grid_ms']:
            own = sorted([row for row in trials if row['spec']['plugins']==selected['plugins']
                          and row['spec']['frame_duration_ms']==frame], key=lambda r:r['spec']['threshold'])
            ax.plot([r['spec']['threshold'] for r in own], [r['metrics']['final_score'] for r in own],
                    marker='.', label=f'{frame:g} ms')
        ax.scatter([selected['threshold']], [frozen['train_metrics'][method]['final_score']],
                   marker='*', s=150, c='black', label='Frozen train choice', zorder=5)
        ax.set_title(method.upper()+'\n'+' + '.join(selected['plugins']), fontsize=10)
        ax.set_xlabel('Threshold'); ax.set_ylabel('TRAIN Final Score (%)')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    stamp_figure(fig, data); fig.tight_layout(); fig.savefig(figures/'train_threshold_curves.png', dpi=180); plt.close(fig)

    lines = ['Kết quả từ cấu hình đóng băng và dữ liệu đánh giá đã lưu; không chạy lại thí nghiệm khi dựng báo cáo.', '',
             '## Phạm vi và giao thức', '',
             f"- Khảo sát {frozen['num_trials']} cấu hình, {frozen['num_pipeline_file_pairs']} lượt cấu hình/file trên {len(frozen['train_manifest'])} WAV train.",
             '- Pha 1: đủ 16 tổ hợp BP/Clip/Energy/Viterbi ở cấu hình lịch sử. Pha 2: hai tổ hợp tốt nhất cộng Champion lịch sử, khung 20/25/30 ms và lưới ngưỡng định trước.',
             '- Hop cố định 10 ms, F0 70–400 Hz, clipping 40%, hệ số Energy/Viterbi giữ nguyên. Không khẳng định đã tìm tối ưu toàn bộ không gian tham số.',
             '- Tiêu chí: trung bình Final Score các file train; hòa điểm ưu tiên sai số file tệ nhất, số plugin, khoảng cách tới khung 25 ms, tên plugin và ngưỡng.',
             '- Dùng tham chiếu thống kê toàn file để hiệu chỉnh; không gọi đây là huấn luyện Gaussian từ nhãn V/UV. Accuracy/F1 từng khung vẫn không có.',
             '- `frozen_config.json` và `frozen_models.json` có checksum, được ghi trước bước đọc/evaluate test. Các tham số không được chọn lại theo điểm test.',
             '- Test từng được xem trong các lần rà soát trước; đây là lựa chọn theo train với cấu hình cố định khi đánh giá, không phải một test set hoàn toàn chưa từng được quan sát.', '',
             '## Preset đã chốt', '', '| Thuật toán | Plugin | Frame/hop (ms) | Ngưỡng | Train trước → sau | Test trước → sau |',
             '|---|---|---:|---:|---:|---:|']
    for method in methods:
        spec = frozen['selected'][method]
        values = [pipelines[split, method, strategy]['metrics']['final_score']
                  for split in ['train', 'test'] for strategy in ['historical_champion', 'selected_champion']]
        lines.append(f"| {method.upper()} | {' + '.join(spec['plugins'])} | {spec['frame_duration_ms']:g}/{spec['hop_duration_ms']:g} | {spec['threshold']:g} | {format_metric(values[0], '.3f')} → {format_metric(values[1], '.3f')} | {format_metric(values[2], '.3f')} → {format_metric(values[3], '.3f')} |")
    lines += ['', 'Bảng giữ đúng điểm trước/sau đã lưu. Không đổi lựa chọn trên train sau khi xem test; số liệu của một tập nhỏ chưa chứng minh khả năng tổng quát.', '',
              '![So sánh train/test](figures/pipelines_train_test.png)', '',
              '![Khảo sát trên train](figures/train_threshold_curves.png)', '',
              '## Đối chứng các giao thức không giám sát', '',
              'Mọi dòng dưới dùng khung 25/10 ms, cùng silence gate, ứng viên F0 và median theo từng đoạn. `train_all` fit toàn bộ train có silence; `train_active` fit phần active của train. `input_active` fit riêng active từng file đầu vào; không dùng LAB để fit nhưng là thích nghi trên chính dữ liệu đánh giá. HMM giữ A/pi cố định và reset qua silence/file boundary. Ngưỡng động dùng GMM train và median/STE cục bộ, không ước lượng SNR.', '',
              '| Giao thức | ACF test (%) | AMDF test (%) | Cepstrum test (%) |', '|---|---:|---:|---:|']
    for name in names:
        values = [protocols['test', method, name]['metrics']['final_score'] for method in methods]
        lines.append('| '+name+' | '+' | '.join(format_metric(value, '.3f') for value in values)+' |')
    lines += ['', 'Các dòng này là đối chứng mô tả; không dùng thứ hạng test để chọn lại preset. Mô hình 2D/HMM fit trên train có kết quả riêng với bản thích nghi theo input.', '',
              '![Đối chứng giao thức](figures/protocols_test.png)', '',
              '## Artifact và tái lập', '',
              '- [Cấu hình đã đóng băng](inputs/frozen_config.json), [mô hình đã fit trên train](inputs/frozen_models.json).',
              '- [Toàn bộ trial theo file](inputs/train/all_trials_per_file.csv), [16 tổ hợp mỗi thuật toán](inputs/train/plugin_sweep.json).',
              '- [Đánh giá đầy đủ](inputs/evaluation.json), [pipeline theo file](inputs/evaluation_pipelines_per_file.csv), [giao thức theo file](inputs/evaluation_protocols_per_file.csv).',
              '- `training_source_snapshot.tar.gz` chứa đúng source theo checksum lúc bắt đầu train; `runtime.json` ghi phiên bản thư viện. Base commit, branch, checksum WAV/LAB và mã đang dùng nằm trong provenance.', '',
              '```powershell',
              '.\\.venv\\Scripts\\python.exe -B scripts/17_validate_and_calibrate.py train --output-dir outputs/stage4_recheck',
              '.\\.venv\\Scripts\\python.exe -B scripts/17_validate_and_calibrate.py evaluate --output-dir outputs/stage4_recheck',
              '```', '',
              'Lệnh train từ chối ghi đè freeze; evaluate kiểm tra checksum và dữ liệu train, từ chối ghi đè evaluation đã có. Cache chỉ lưu kết quả tính theo khung và cấu hình chính xác; mọi quyết định/hậu xử lý/DP vẫn chạy bình thường và đã được đối chiếu với đường không cache.', '',
              '## Giới hạn còn lại', '',
              '- Không có nhãn V/UV theo thời gian hoặc F0 chuẩn từng khung; Final Score chỉ đo thống kê toàn file.',
              '- Kết quả tốt về mean/std/count không chứng minh đúng contour hoặc hết octave errors. Việc tự gán nhãn dữ liệu cũ không thể thay bằng chứng tham chiếu.',
              '- Chưa có tập kiểm thử mới độc lập sau các vòng phát triển; chưa đủ cơ sở khẳng định preset tối ưu trên mọi giọng/môi trường.',
              '- Word, slide và phép đo T0 thủ công là các đầu ra môn học riêng, chưa được tạo trong đợt sửa code này.', '']
    if all(name in data['extras'] for name in ('cepstrum_trace.json', 'legacy_with_cepstrum_branch/cepstrum_trace.json')):
        original = data['extras']['cepstrum_trace.json']
        corrected = data['extras']['legacy_with_cepstrum_branch/cepstrum_trace.json']
        lines += ['', '## Truy vết Cepstrum lịch sử', '',
                  'Hai replay là bằng chứng riêng đã lưu; không chạy lại khi render.',
                  f"- Replay `{original['source_revision']}`: {original['replay']['detector_function_calls']}",
                  f"- Chỉ sửa nhánh `{corrected['plugin_detector_override']}`: {corrected['replay']['detector_function_calls']}",
                  '- [Replay gốc](inputs/cepstrum_trace.json), [replay sửa nhánh](inputs/legacy_with_cepstrum_branch/cepstrum_trace.json).', '']
    sections['docs/STAGE4_RESULTS.md'] = '\n'.join(lines)
    # Preserve historical chapters; clearly mark their old tables and append
    # current figures directly from the recorded data, with repeatable markers.
    for index, filename in enumerate(CHAPTERS):
        path = 'docs/'+filename
        supplement = ['## Kết quả từ run đã xác minh', '',
                      'Nguồn: cấu hình đóng băng và evaluation đã ghim trong gói báo cáo. Final Score là sai số thống kê toàn file; thấp hơn tốt hơn. Accuracy/F1 từng khung không xác định vì thiếu nhãn thời gian.', '']
        if index < 3:
            method = methods[index]
            spec = frozen['selected'][method]
            supplement += [f"Preset: {' + '.join(spec['plugins'])}, frame/hop {spec['frame_duration_ms']:g}/{spec['hop_duration_ms']:g} ms, T={spec['threshold']:g}.", '',
                           '| Cấu hình | Train (%) | Test (%) |', '|---|---:|---:|']
            for strategy in ['baseline', 'historical_champion', 'selected_champion']:
                values = [pipelines[split, method, strategy]['metrics']['final_score'] for split in ['train','test']]
                supplement.append(f'| {strategy} | {format_metric(values[0], ".3f")} | {format_metric(values[1], ".3f")} |')
            supplement += ['', '| File test | Mean dự đoán (Hz) | Std dự đoán (Hz) | Khung voiced | Final Score (%) |', '|---|---:|---:|---:|---:|']
            for row in pipelines['test', method, 'selected_champion']['files']:
                supplement.append(f"| {row['file']} | {format_metric(row['pred_f0_mean'])} | {format_metric(row['pred_f0_std'])} | {row['pred_f0_num']} | {format_metric(row['final_score'])} |")
            if method == 'cepstrum' and 'cepstrum_trace.json' in data['extras']:
                supplement += ['', '**Đính chính:** mốc 4,43%/7,62% lịch sử được tạo khi detector rơi vào ACF nhưng Viterbi dùng Cepstrum. Replay đúng commit và chỉ thay nhánh Cepstrum đã xác nhận nguyên nhân. Preset hiện hành giảm lỗi train nhưng tăng lỗi test so với trước hiệu chỉnh; không gọi đây là cải thiện tổng quát.']
        elif index == 3:
            supplement += ['| Thuật toán | Baseline test (%) | Trước hiệu chỉnh (%) | Sau chọn trên train (%) |', '|---|---:|---:|---:|']
            for method in methods:
                values = [pipelines['test',method,name]['metrics']['final_score'] for name in ['baseline','historical_champion','selected_champion']]
                supplement.append('| '+method.upper()+' | '+' | '.join(format_metric(v, '.3f') for v in values)+' |')
            supplement += ['', '![Kết quả hiện hành](figures/pipelines_train_test.png)']
        else:
            supplement += ['Các giao thức dưới đây thuộc evaluation đã lưu; không chạy lại Otsu hoặc các mô hình bằng mã hiện tại.', '',
                           'Giao thức `train_all` gồm silence, `train_active` chỉ active train, `input_active` fit riêng từng file đầu vào. Tất cả dùng cùng frame/hop 25/10 ms, silence gate và median theo đoạn.', '',
                           '| Giao thức | ACF test (%) | AMDF test (%) | Cepstrum test (%) |', '|---|---:|---:|---:|']
            for name in names:
                values = [protocols['test',method,name]['metrics']['final_score'] for method in methods]
                supplement.append('| '+name+' | '+' | '.join(format_metric(v, '.3f') for v in values)+' |')
            supplement += ['', 'Các thứ hạng test là mô tả; không được dùng để chọn lại tham số.', '',
                           '![Giao thức hiện hành](figures/protocols_test.png)']
        supplement += ['', '[Chi tiết và giới hạn kiểm chứng](README.md).']
        sections[path] = '\n'.join(supplement)

    readme = 'README.md'
    current = ['| Champion | Ngưỡng | Train Final Score (%) | Test Final Score (%) |', '|---|---:|---:|---:|']
    for method in methods:
        spec = frozen['selected'][method]
        values = [pipelines[split,method,'selected_champion']['metrics']['final_score'] for split in ['train','test']]
        current.append(f"| {method.upper()} | {spec['threshold']:g} | {format_metric(values[0], '.3f')} | {format_metric(values[1], '.3f')} |")
    sections[readme] = '\n'.join(current)
    write_json(folder/'summary.json', {'num_trials': frozen['num_trials'], 'num_train_pipeline_file_pairs': frozen['num_pipeline_file_pairs'],
                                       'selected': frozen['selected'], 'pipelines': [{k:r[k] for k in ('dataset','method','strategy','metrics')} for r in result['pipelines']],
                                       'num_protocols_per_method': len(names), 'test_was_not_used_for_selection': True})
    for target in sections:
        sections[target] = source_note(data)+'\n\n'+sections[target]
    write_json(folder/'sections.json', sections)
    (folder/'README.md').write_text('# Stage 4 — kết quả đã lưu\n\n'+sections['docs/STAGE4_RESULTS.md']+'\n', encoding='utf-8')
