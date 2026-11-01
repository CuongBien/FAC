"""Render stage 4 tables/figures from frozen choices and recorded evaluations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.legacy import read_evaluation_bundle
from src.experiments.snapshots import verify_source_snapshot
from src.experiments.manifest import sha256
from src.experiments.serialization import write_json

CHAPTERS = ('01_acf_theory_and_experiments.md', '02_amdf_theory_and_experiments.md',
            '03_cepstrum_theory_and_experiments.md', '04_test_set_benchmark_and_analysis.md',
            '05_unsupervised_thresholding_analysis.md')
README_MARKERS = ('<!-- STAGE4_README_START -->', '<!-- STAGE4_README_END -->')
REPORT_MARKERS = ('<!-- STAGE4_REPORT_START -->', '<!-- STAGE4_REPORT_END -->')
CHAPTER_MARKERS = ('<!-- STAGE4_CURRENT_START -->', '<!-- STAGE4_CURRENT_END -->')


def replace_generated_block(source, content, start_marker, end_marker):
    """Only replace an explicitly owned block; refuse missing/ambiguous markers."""
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError(f'Expected exactly one generated block: {start_marker} / {end_marker}')
    begin, end = source.index(start_marker), source.index(end_marker)
    if end < begin:
        raise ValueError(f'Generated block markers are reversed: {start_marker}')
    block = start_marker+'\n'+content.rstrip()+'\n'+end_marker
    return source[:begin]+block+source[end+len(end_marker):]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT/'outputs/stage4_validation'))
    args = parser.parse_args()
    folder = Path(args.output_dir)
    frozen, _, result = read_evaluation_bundle(folder)
    verify_source_snapshot(folder/'training_source_snapshot.tar.gz', frozen['provenance'])
    runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
    if runtime['snapshot_sha256'] != sha256(folder/'training_source_snapshot.tar.gz'):
        raise ValueError('Runtime record uses a different source snapshot')
    artifact = json.loads((folder/'cepstrum_trace.json').read_text(encoding='utf-8'))
    corrected = json.loads((folder/'legacy_with_cepstrum_branch/cepstrum_trace.json').read_text(encoding='utf-8'))
    owned = {ROOT/'README.md': README_MARKERS, ROOT/'docs/STAGE4_RESULTS.md': REPORT_MARKERS,
             **{ROOT/'docs'/name: CHAPTER_MARKERS for name in CHAPTERS}}
    sources = {path: path.read_text(encoding='utf-8') for path in owned}
    # Validate every boundary before writing figures, documents or summaries.
    for path, markers in owned.items():
        replace_generated_block(sources[path], '', *markers)
    pipelines = {(r['dataset'], r['method'], r['strategy']): r for r in result['pipelines']}
    protocols = {(r['dataset'], r['method'], r['strategy']): r for r in result['protocols']}
    methods = ['acf', 'amdf', 'cepstrum']
    figures = folder/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, split in zip(axes, ['train', 'test']):
        for i, (strategy, label) in enumerate([('baseline', 'Historical baseline'),
                                               ('historical_champion', 'Before calibration'),
                                               ('selected_champion', 'Frozen train selection')]):
            values = [pipelines[split, m, strategy]['metrics']['final_score'] for m in methods]
            bars = ax.bar(np.arange(3)+(i-1)*.25, values, width=.25, label=label)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+.4, f'{value:.3f}', ha='center', fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylabel('Final Score (%); lower is better')
        ax.set_title(split.upper()); ax.set_ylim(0, 57)
        ax.grid(axis='y', alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures/'pipelines_train_test.png', dpi=180); plt.close(fig)

    names = [row['strategy'] for row in result['protocols'] if row['dataset']=='test' and row['method']=='acf']
    matrix = np.array([[protocols['test', m, name]['metrics']['final_score'] for m in methods] for name in names])
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels([m.upper() for m in methods])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    for row in range(len(names)):
        for column in range(3):
            ax.text(column, row, f'{matrix[row,column]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if matrix[row,column] > float(np.max(matrix))*.6 else 'black')
    ax.set_title('TEST: fixed train models vs per-input adaptation\n25 ms frames / 10 ms hop, common silence gate and segment median')
    fig.colorbar(image, ax=ax, label='Final Score (%); lower is better')
    fig.tight_layout(); fig.savefig(figures/'protocols_test.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, method in zip(axes, methods):
        trials = json.loads((folder/'train'/f'{method}_trials.json').read_text(encoding='utf-8'))
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
    fig.tight_layout(); fig.savefig(figures/'train_threshold_curves.png', dpi=180); plt.close(fig)

    lines = ['Ngày 30/10/2026. Kết quả sinh từ `scripts/19_report_validation.py`, cấu hình đóng băng và dữ liệu đánh giá đã lưu.', '',
             '## Phạm vi và giao thức', '',
             f"- Khảo sát {frozen['num_trials']} cấu hình, {frozen['num_pipeline_file_pairs']} lượt cấu hình/file trên bốn WAV train.",
             '- Pha 1: đủ 16 tổ hợp BP/Clip/Energy/Viterbi ở cấu hình lịch sử. Pha 2: hai tổ hợp tốt nhất cộng Champion lịch sử, khung 20/25/30 ms và lưới ngưỡng định trước.',
             '- Hop cố định 10 ms, F0 70–400 Hz, clipping 40%, hệ số Energy/Viterbi giữ nguyên. Không khẳng định đã tìm tối ưu toàn bộ không gian tham số.',
             '- Tiêu chí: trung bình Final Score của bốn file train; hòa điểm ưu tiên sai số file tệ nhất, số plugin, khoảng cách tới khung 25 ms, tên plugin và ngưỡng.',
             '- Dùng tham chiếu thống kê toàn file để hiệu chỉnh; không gọi đây là huấn luyện Gaussian từ nhãn V/UV. Accuracy/F1 từng khung vẫn không có.',
             '- `frozen_config.json` và `frozen_models.json` có checksum, được ghi trước bước đọc/evaluate test. Các tham số không được chọn lại theo điểm test.',
             '- Test từng được xem trong các lần rà soát trước; đây là lựa chọn theo train với cấu hình cố định khi đánh giá, không phải một test set hoàn toàn chưa từng được quan sát.', '',
             '## Preset đã chốt', '', '| Thuật toán | Plugin | Frame/hop (ms) | Ngưỡng | Train trước → sau | Test trước → sau |',
             '|---|---|---:|---:|---:|---:|']
    for method in methods:
        spec = frozen['selected'][method]
        values = [pipelines[split, method, strategy]['metrics']['final_score']
                  for split in ['train', 'test'] for strategy in ['historical_champion', 'selected_champion']]
        lines.append(f"| {method.upper()} | {' + '.join(spec['plugins'])} | {spec['frame_duration_ms']:g}/{spec['hop_duration_ms']:g} | {spec['threshold']:g} | {values[0]:.3f} → {values[1]:.3f} | {values[2]:.3f} → {values[3]:.3f} |")
    lines += ['', '**ACF và AMDF giảm sai số test; Cepstrum tăng sai số test dù giảm trên train.** Giữ preset đã chốt, không đảo lại sau khi xem test. Chỉ có bốn file train nên khả năng tổng quát còn hạn chế.', '',
              '![So sánh train/test](../outputs/stage4_validation/figures/pipelines_train_test.png)', '',
              '![Khảo sát trên train](../outputs/stage4_validation/figures/train_threshold_curves.png)', '',
              '## Truy vết Cepstrum lịch sử', '',
              f"Replay commit `{artifact['source_revision'][:7]}` đã tái lập đúng 4,43% train / 7,62% test trong artifact. Bộ đếm xác nhận {artifact['replay']['detector_function_calls']['acf']} lời gọi ACF và {artifact['replay']['detector_function_calls']['cepstrum']} lời gọi Cepstrum khi plugin detector báo method là Cepstrum.", '',
              'Commit này thiếu nhánh `elif method == "cepstrum"` trong `src/plugins/plugin_detector.py`: ứng viên pitch và quyết định V/UV rơi vào nhánh ACF, sau đó hậu xử lý/Viterbi lại dùng Cepstrum. Kết quả cũ là một pipeline lai, không phải Cepstrum thuần theo mô tả.', '',
              f"Chỉ thay `plugin_detector.py` bằng bản có nhánh Cepstrum từ commit `{corrected['plugin_detector_override']}`, giữ mọi mã/dữ liệu khác, đổi số lời gọi thành ACF={corrected['replay']['detector_function_calls']['acf']}, Cepstrum={corrected['replay']['detector_function_calls']['cepstrum']}. Điểm thành 6,84% / 13,24%, khớp mốc trước sửa đợt 1.", '',
              'Vì vậy không dùng 4,43%/7,62% làm mốc chất lượng Cepstrum đúng thuật toán. Các artifact cũ được giữ nguyên để truy vết; không phục hồi lỗi nhánh ACF để lấy lại điểm thấp.', '',
              '## Đối chứng các giao thức không giám sát', '',
              'Mọi dòng dưới dùng khung 25/10 ms, cùng silence gate, ứng viên F0 và median theo từng đoạn. `train_all` fit toàn bộ train có silence; `train_active` fit phần active của train. `input_active` fit riêng active từng file đầu vào; không dùng LAB để fit nhưng là thích nghi trên chính dữ liệu đánh giá. HMM giữ A/pi cố định và reset qua silence/file boundary. Ngưỡng động dùng GMM train và median/STE cục bộ, không ước lượng SNR.', '',
              '| Giao thức | ACF test (%) | AMDF test (%) | Cepstrum test (%) |', '|---|---:|---:|---:|']
    for name in names:
        values = [protocols['test', method, name]['metrics']['final_score'] for method in methods]
        lines.append('| '+name+' | '+' | '.join(f'{value:.3f}' for value in values)+' |')
    lines += ['', 'Các dòng này là đối chứng mô tả; không dùng thứ hạng test để chọn lại preset. Mô hình 2D/HMM fit trên train có kết quả riêng với bản thích nghi theo input.', '',
              '![Đối chứng giao thức](../outputs/stage4_validation/figures/protocols_test.png)', '',
              '## Artifact và tái lập', '',
              '- [Cấu hình đã đóng băng](../outputs/stage4_validation/frozen_config.json), [mô hình đã fit trên train](../outputs/stage4_validation/frozen_models.json).',
              '- [Toàn bộ trial theo file](../outputs/stage4_validation/train/all_trials_per_file.csv), [16 tổ hợp mỗi thuật toán](../outputs/stage4_validation/train/plugin_sweep.json).',
              '- [Đánh giá đầy đủ](../outputs/stage4_validation/evaluation.json), [pipeline theo file](../outputs/stage4_validation/evaluation_pipelines_per_file.csv), [giao thức theo file](../outputs/stage4_validation/evaluation_protocols_per_file.csv).',
              '- [Replay gốc](../outputs/stage4_validation/cepstrum_trace.json), [replay chỉ sửa nhánh Cepstrum](../outputs/stage4_validation/legacy_with_cepstrum_branch/cepstrum_trace.json).',
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
    report_path = ROOT/'docs/STAGE4_RESULTS.md'
    report_path.write_text(replace_generated_block(sources[report_path], '\n'.join(lines), *REPORT_MARKERS), encoding='utf-8')
    # Preserve historical chapters; clearly mark their old tables and append
    # current figures directly from the recorded data, with repeatable markers.
    for index, filename in enumerate(CHAPTERS):
        path = ROOT/'docs'/filename
        supplement = ['## Kết quả kiểm chứng hiện hành — 30/10/2026', '',
                      'Nguồn: cấu hình đóng băng và `outputs/stage4_validation/evaluation.json`. Final Score là sai số thống kê toàn file; thấp hơn tốt hơn. Accuracy/F1 từng khung không xác định vì thiếu nhãn thời gian.', '']
        if index < 3:
            method = methods[index]
            spec = frozen['selected'][method]
            supplement += [f"Preset: {' + '.join(spec['plugins'])}, frame/hop {spec['frame_duration_ms']:g}/{spec['hop_duration_ms']:g} ms, T={spec['threshold']:g}.", '',
                           '| Cấu hình | Train (%) | Test (%) |', '|---|---:|---:|']
            for strategy in ['baseline', 'historical_champion', 'selected_champion']:
                values = [pipelines[split, method, strategy]['metrics']['final_score'] for split in ['train','test']]
                supplement.append(f'| {strategy} | {values[0]:.3f} | {values[1]:.3f} |')
            supplement += ['', '| File test | Mean dự đoán (Hz) | Std dự đoán (Hz) | Khung voiced | Final Score (%) |', '|---|---:|---:|---:|---:|']
            for row in pipelines['test', method, 'selected_champion']['files']:
                supplement.append(f"| {row['file']} | {row['pred_f0_mean']:.2f} | {row['pred_f0_std']:.2f} | {row['pred_f0_num']} | {row['final_score']:.2f} |")
            if method == 'cepstrum':
                supplement += ['', '**Đính chính:** mốc 4,43%/7,62% lịch sử được tạo khi detector rơi vào ACF nhưng Viterbi dùng Cepstrum. Replay đúng commit và chỉ thay nhánh Cepstrum đã xác nhận nguyên nhân. Preset hiện hành giảm lỗi train nhưng tăng lỗi test so với trước hiệu chỉnh; không gọi đây là cải thiện tổng quát.']
        elif index == 3:
            supplement += ['| Thuật toán | Baseline test (%) | Trước hiệu chỉnh (%) | Sau chọn trên train (%) |', '|---|---:|---:|---:|']
            for method in methods:
                values = [pipelines['test',method,name]['metrics']['final_score'] for name in ['baseline','historical_champion','selected_champion']]
                supplement.append('| '+method.upper()+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
            supplement += ['', '![Kết quả hiện hành](../outputs/stage4_validation/figures/pipelines_train_test.png)']
        else:
            supplement += ['> Đây là snapshot 17 giao thức Stage 4, gồm Otsu trước sửa ngày 01/11. Benchmark 15/16 sau sửa được liên kết trong [rà soát main](MAIN_REVIEW_FIXES.md).', '',
                           'Giao thức `train_all` gồm silence, `train_active` chỉ active train, `input_active` fit riêng từng file đầu vào. Tất cả dùng cùng frame/hop 25/10 ms, silence gate và median theo đoạn.', '',
                           '| Giao thức | ACF test (%) | AMDF test (%) | Cepstrum test (%) |', '|---|---:|---:|---:|']
            for name in names:
                values = [protocols['test',method,name]['metrics']['final_score'] for method in methods]
                supplement.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
            supplement += ['', 'Các thứ hạng test là mô tả; không được dùng để chọn lại tham số.', '',
                           '![Giao thức hiện hành](../outputs/stage4_validation/figures/protocols_test.png)']
        supplement += ['', '[Chi tiết và giới hạn kiểm chứng](STAGE4_RESULTS.md).']
        path.write_text(replace_generated_block(sources[path], '\n'.join(supplement), *CHAPTER_MARKERS), encoding='utf-8')

    readme = ROOT/'README.md'
    current = ['| Champion | Ngưỡng | Train Final Score (%) | Test Final Score (%) |', '|---|---:|---:|---:|']
    for method in methods:
        spec = frozen['selected'][method]
        values = [pipelines[split,method,'selected_champion']['metrics']['final_score'] for split in ['train','test']]
        current.append(f"| {method.upper()} | {spec['threshold']:g} | {values[0]:.3f} | {values[1]:.3f} |")
    readme.write_text(replace_generated_block(sources[readme], '\n'.join(current), *README_MARKERS), encoding='utf-8')
    write_json(folder/'summary.json', {'num_trials': frozen['num_trials'], 'num_train_pipeline_file_pairs': frozen['num_pipeline_file_pairs'],
                                       'selected': frozen['selected'], 'pipelines': [{k:r[k] for k in ('dataset','method','strategy','metrics')} for r in result['pipelines']],
                                       'num_protocols_per_method': len(names), 'test_was_not_used_for_selection': True})
    write_json(folder/'report_generation.json', {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'report_script_sha256': sha256(Path(__file__)),
        'config_sha256': result['config_sha256'], 'models_sha256': result['models_sha256'],
        'evaluation_sha256': sha256(folder/'evaluation.json'),
        'documents_sha256_at_generation': {path.relative_to(ROOT).as_posix(): sha256(path) for path in owned},
        'write_scope': 'Only STAGE4 marker blocks; all text outside them is preserved.',
    })
    print('Stage 4 figures and docs/STAGE4_RESULTS.md generated')


if __name__ == '__main__':
    main()
