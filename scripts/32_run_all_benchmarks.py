"""Rerun all retained pipelines and publish one train/test results inventory."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprints():
    sources = sorted([*(ROOT/'src').rglob('*.py'), *(ROOT/'scripts').glob('*.py'),
                      *(ROOT/'tests').rglob('*.py'), ROOT/'pyproject.toml', ROOT/'uv.lock'])
    data = sorted([*(ROOT/'TinHieuHuanLuyen').glob('*'), *(ROOT/'TinHieuKiemThu').glob('*')])
    return {'source': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
            'data': {p.relative_to(ROOT).as_posix(): sha(p) for p in data if p.suffix.lower() in ('.wav', '.lab')}}


def command(folder, label, args):
    log = folder/'logs'/(label+'.log')
    log.parent.mkdir(exist_ok=True)
    print('START', label, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', *map(str, args)], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-12000:], flush=True)
        raise RuntimeError('Benchmark failed: '+label)
    print('DONE', label, flush=True)


def raw_pitch_shift(folder):
    """Exercise the current public script 27/28 APIs on both splits."""
    standalone = load_module('full_raw_pitch', 'scripts/27_evaluate_ssl_methods.py')
    champion = load_module('full_raw_matrix', 'scripts/28_benchmark_champion_ssl.py')
    output = folder/'raw_pitch_shift'
    output.mkdir()
    result = {'standalone': {}, 'champion': {}}
    for split, directory in (('train', 'TinHieuHuanLuyen'), ('test', 'TinHieuKiemThu')):
        print('START raw_pitch_shift', split, flush=True)
        result['standalone'][split] = standalone.evaluate_test_dataset(str(ROOT/directory))
        result['champion'][split] = champion.evaluate_champion_ssl_matrix(str(ROOT/directory))
        print('DONE raw_pitch_shift', split, flush=True)
    write_json(output/'results.json', result)
    rows = []
    for split in ('train', 'test'):
        for method, strategies in {'standalone': result['standalone'][split],
                                    **result['champion'][split]}.items():
            for strategy, record in strategies.items():
                rows.extend({'split': split, 'method': method, 'strategy': strategy, **row} for row in record['files'])
    fields = ['split', 'method', 'strategy', 'file', 'diff_f0', 'mape_f0', 'diff_std',
              'mape_std', 'diff_num', 'mape_num', 'final_score']
    with (output/'per_file.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def fmt(value):
    return 'N/A' if value is None else f'{value:.3f}'


def publish(folder, manifest, raw_count):
    from src.metrics.aggregation import FIELDS
    unsup = json.loads((folder/'unsupervised/evaluation.json').read_bytes())
    native = json.loads((folder/'native_pitch_shift/results.json').read_bytes())
    series, files = [], []
    for row in unsup['pipelines']+unsup['protocols']:
        item = {'family':'unsupervised', 'split':row['dataset'], 'method':row['method'],
                'strategy':row['strategy'], 'metrics':row['metrics']}
        series.append(item)
        files.extend({**{k:item[k] for k in ('family','split','method','strategy')}, **record} for record in row['files'])
    for family, data in (('native_pitch_shift', native),):
        for row in data['summary']:
            series.append({'family':family, 'split':row['split'],
                'method':row.get('method',row.get('pipeline')), 'strategy':row['strategy'], 'metrics':row['metrics']})
        for row in data['files']:
            files.append({'family':family,'split':row['split'],'method':row.get('method',row.get('pipeline')),
                          **{k:v for k,v in row.items() if k not in ('pipeline','method')}})
    write_json(folder/'summary.json', {'manifest':manifest, 'series':series,
        'canonical_file_records':len(files), 'raw_script_file_records':raw_count})
    fields = ['family','split','method','strategy',*FIELDS]
    with (folder/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in series:
            writer.writerow({**{k:row[k] for k in fields[:4]}, **{k:row['metrics'].get(k) for k in FIELDS}})
    with (folder/'per_file.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields[:4]+['file']+list(FIELDS),extrasaction='ignore')
        writer.writeheader();writer.writerows(files)
    lookup={(r['family'],r['split'],r['method'],r['strategy']):r['metrics'] for r in series}
    def table(family, strategies, methods=('acf','amdf','cepstrum')):
        lines=['| Phương pháp | '+' | '.join(m.upper()+' train → test' for m in methods)+' |',
               '|---|'+'---:|'*len(methods)]
        for key,label in strategies:
            cells=[]
            for method in methods:
                values=[lookup[family,split,method,key]['final_score'] for split in ('train','test')]
                cells.append(' → '.join(fmt(v) for v in values))
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        return lines+['']
    lines=['# Toàn bộ kết quả hiện hành — chạy lại train/test', '',
        'Thời gian bắt đầu (UTC+7): '+manifest['started_at_local'],
        'Nhánh: `'+manifest['branch']+'`. Mỗi tập có bốn WAV. Tất cả các bảng chính dùng metric legacy-v1: Final Score (%) càng thấp càng tốt.',
        '', '## Phạm vi và giao thức', '',
        '- Giữ nguyên các preset/plugin Champion. Baseline 25/10 ms; Champion và các biến thể 30/10 ms, F0 70–400 Hz.',
        '- Otsu, K-means 1D/2D và dynamic fit từ train rồi đóng băng; dynamic lấy Otsu train làm nền. K-means 2D thích nghi input được tách riêng.',
        '- Mọi nhánh chính dùng ACF–ACF, AMDF–AMDF hoặc Cepstrum–Cepstrum theo thuật toán. ACF-only cũ chỉ giữ riêng làm lịch sử/standalone ACF. Không chia sẻ nhãn ACF sang nhánh khác.',
        '- LAB chỉ có F0mean/F0std/F0num; chưa có Accuracy/F1 hoặc sai số contour từng khung. Test đã được xem trong các lần trước; đây không phải test độc lập mới.',
        '', '## 1. Baseline và Champion', '']
    lines+=table('unsupervised',[('baseline','Baseline'),('champion','Champion')])
    lines+=['## 2. Không giám sát trên Champion','']
    lines+=table('unsupervised',[('otsu','Otsu'),('kmeans','K-means 1D'),('kmeans2d','K-means 2D (train-fitted)'),
                                ('dynamic','Ngưỡng động (nền Otsu)'),('kmeans2d_input_active','K-means 2D (fit từng input)')])
    lines+=['## 3. Pitch-Shift không model','']
    lines+=table('native_pitch_shift',[('pitch_shift_own_method','Pitch-Shift theo từng thuật toán')])
    lines+=['**Kiểm chứng và dữ liệu đầy đủ**','',
        f'- {len(files)} kết quả từng file theo metric chung; {raw_count} bản ghi bổ sung từ API script 27/28.',
        '- Script 27/28 và CLI predict -m all chạy trên cả train/test. Toàn bộ unit test được chạy lại.',
        '- Bảng chính dùng legacy-v1; script 27/28 tính trực tiếp rồi lấy trung bình, lưu riêng để tránh khác biệt làm tròn.',
        '- Source và WAV/LAB được hash trước/sau toàn bộ lần chạy. Không thay các preset Champion.', '',
        '[Tổng hợp CSV](summary.csv) · [Từng file CSV](per_file.csv) · [JSON](summary.json) · [Manifest](manifest.json)', '',
        '[Benchmark không giám sát](unsupervised/evaluation.json) · [Pitch-Shift riêng từng thuật toán](native_pitch_shift/README.md)', '',
        '[API script 27/28](raw_pitch_shift/results.json) · [Dự đoán train](predictions/train/evaluation_summary_metrics.csv) · [Dự đoán test](predictions/test/evaluation_summary_metrics.csv) · [Log kiểm thử](logs/tests.log)','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    return series,len(files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    folder=args.output_dir.resolve();folder.mkdir(parents=True,exist_ok=False)
    started=fingerprints()
    manifest={'started_at_local':datetime.now(timezone(timedelta(hours=7))).isoformat(),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'runtime':{'python':sys.version,**{name:version(name) for name in ('numpy','scipy','matplotlib')}},
        'fingerprints':started,'status':'running'}
    write_json(folder/'manifest.json',manifest)
    for phase in ('train','evaluate','report'):
        command(folder,'unsupervised_'+phase,['scripts/21_benchmark_champion_unsupervised.py',phase,'--output-dir',folder/'unsupervised'])
    command(folder,'native_pitch_shift',['scripts/31_compare_native_pitch_shift.py','--output-dir',folder/'native_pitch_shift'])
    raw_count=raw_pitch_shift(folder)
    for split,directory in (('train','TinHieuHuanLuyen'),('test','TinHieuKiemThu')):
        command(folder,'predict_'+split,['predict.py','-i',directory,'-m','all','-o',folder/'predictions'/split])
    command(folder,'tests',['-m','unittest','discover','-s','tests'])
    if fingerprints()!=started:
        raise ValueError('Source or data changed during the full benchmark')
    manifest.update(status='complete',finished_at_local=datetime.now(timezone(timedelta(hours=7))).isoformat(),
                    source_data_unchanged=True)
    series,count=publish(folder,manifest,raw_count)
    write_json(folder/'manifest.json',manifest)
    print('COMPLETE',folder/'README.md',len(series),'series;',count,'canonical per-file results',flush=True)


if __name__=='__main__':
    if sys.platform=='win32':sys.stdout.reconfigure(encoding='utf-8')
    main()
