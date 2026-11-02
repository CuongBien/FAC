"""Selected unsupervised decisions: tables and figures from frozen results."""
from pathlib import Path
import numpy as np
from src.metrics.formatting import format_metric
from .common import source_note, stamp_figure
from src.experiments.serialization import write_json
METHODS = ('acf', 'amdf', 'cepstrum')
PRIMARY = ('otsu', 'kmeans', 'kmeans2d', 'dynamic')
ADAPTIVE = ('kmeans2d_input_active',)

def render(view, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = view.to_dict()
    result = data['evaluation']
    folder = Path(folder)
    rows = {(r['dataset'], r['method'], r['strategy']): r for r in result['pipelines']+result['protocols']}
    names = ('baseline', 'champion')+PRIMARY+ADAPTIVE
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, dataset in zip(axes, ('train', 'test')):
        matrix = np.asarray([[rows[dataset, m, n]['metrics']['final_score'] for m in METHODS] for n in names], dtype=float)
        plot = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(3)); ax.set_xticklabels([m.upper() for m in METHODS])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(3):
                ax.text(j, i, format_metric(matrix[i,j], '.3f'), ha='center', va='center', fontsize=9)
        ax.set_title(dataset.upper()+' Final Score (%)')
        fig.colorbar(plot, ax=ax, fraction=.045)
    fig.suptitle('Otsu, K-means 1D/2D and dynamic thresholds')
    stamp_figure(fig, data); fig.tight_layout()
    (folder/'figures').mkdir(exist_ok=True)
    fig.savefig(folder/'figures/score_matrix.png', dpi=180); plt.close(fig)
    lines = ['# Otsu, K-means 1D/2D và ngưỡng động', '', source_note(data), '',
        'Ngày đánh giá: '+result['evaluated_at_utc'], '',
        '## Giao thức', '',
        '- ACF, AMDF và Cepstrum; khung 30 ms, hop 10 ms; vùng F0 70–400 Hz.',
        '- Các dòng otsu/kmeans/kmeans2d/dynamic fit đặc trưng active trên train, đóng băng trước test. Không dùng LAB để fit.',
        '- Dynamic dùng Otsu train làm ngưỡng nền, median cục bộ trong cửa sổ 50 khung, alpha 0,30, trọng số theo STE; không nối qua im lặng. Đây là xử lý offline.',
        '- kmeans2d_input_active fit lại trên chính từng input và được báo cáo riêng.',
        '- Giữ nguyên Champion: ACF dùng Bandpass + Energy + Viterbi; AMDF dùng Energy + Viterbi; Cepstrum dùng Bandpass + Center clipping + Energy + Viterbi. Baseline dùng 25/10 ms.',
        '- Các quyết định mới dùng đúng tiền xử lý, median, Energy và Viterbi của Champion. Ngưỡng guard Energy giữ cố định 0,45 / 0,50 / 0,10; chỉ bộ quyết định mới được fit không giám sát.',
        '- Final Score là trung bình ba sai số phần trăm F0mean/F0std/F0num rồi trung bình theo file; thấp hơn tốt hơn. LAB chỉ có thống kê nên Accuracy/F1 không xác định.',
        '- Tập test đã được quan sát trước đây; không dùng thứ hạng để chọn lại cấu hình.', '']
    for dataset in ('train', 'test'):
        lines += ['## '+dataset.upper(), '', '| Phương pháp | ACF (%) | AMDF (%) | Cepstrum (%) |', '|---|---:|---:|---:|']
        for name in names:
            lines.append('| '+name+' | '+' | '.join(format_metric(rows[dataset,m,name]['metrics']['final_score'], '.3f') for m in METHODS)+' |')
        lines.append('')
    lines += ['![Train và test](figures/score_matrix.png)', '', '## Ngưỡng fit trên train', '',
              '| Thuật toán | Otsu / nền dynamic | K-means 1D |', '|---|---:|---:|']
    for method in METHODS:
        fits = data['models']['models'][method]['thresholds']
        lines.append('| '+method+' | '+' | '.join(format_metric(fits[n], '.6f') for n in ('otsu','kmeans'))+' |')
    count = sum(len(row['files']) for row in rows.values())
    lines += ['', '## Kiểm chứng', '',
        f"- {count} kết quả từng file; {result['native_champion_replays_matched']} đối chứng khớp từng label/time/F0 với đường chạy trực tiếp.",
        '- [CSV từng file](inputs/per_file.csv), [evaluation](inputs/evaluation.json), [cấu hình](inputs/frozen_config.json), [mô hình](inputs/frozen_models.json).',
        '- Snapshot nguồn và hash dữ liệu/artefact được lưu cùng run.', '']
    text = '\n'.join(lines)
    (folder/'README.md').write_text(text, encoding='utf-8')
    write_json(folder/'sections.json', {'docs/CHAMPION_UNSUPERVISED_RESULTS.md': text})
