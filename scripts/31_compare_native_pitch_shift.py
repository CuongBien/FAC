"""Train/test comparison: original shared ACF vs each algorithm's Pitch-Shift."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analysis.ssl_pitch_shift import analyze_signal_ssl_pitch_shift
from src.analysis.ssl_pitch_shift_methods import analyze_signal_pitch_shift_by_method
from src.analysis.champion_unsupervised import experiment_configuration
from src.core.audio import load_wav
from src.pipeline.champion import extract_file_features
from src.pipelines import build_champion
from src.metrics.evaluation import evaluate_result
from src.metrics.aggregation import aggregate


def run(output, include_legacy_shared_acf=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    settings=experiment_configuration();records=[]
    for split,directory in (('train','TinHieuHuanLuyen'),('test','TinHieuKiemThu')):
        for path in sorted((ROOT/directory).glob('*.wav')):
            fs,signal,_=load_wav(str(path))
            shared=analyze_signal_ssl_pitch_shift(signal,fs) if include_legacy_shared_acf else None
            for method in ('acf','amdf','cepstrum'):
                native=analyze_signal_pitch_shift_by_method(signal,fs,method)
                detector=build_champion(method)
                features=extract_file_features(path,method,settings)
                results={'champion':detector.process_file(str(path)),
                    'pitch_shift_own_method':detector.process_feature_labels(features,native['labels'])}
                if include_legacy_shared_acf:
                    results['pitch_shift_shared_acf']=detector.process_feature_labels(features,shared['labels'])
                for strategy,result in results.items():
                    records.append({'split':split,'method':method,'strategy':strategy,**evaluate_result(result,path)})
                folder=output/'scores'/split/method;folder.mkdir(parents=True,exist_ok=True)
                with (folder/(path.stem+'.csv')).open('x',encoding='utf-8',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['frame','own_method_score','own_label'])
                    writer.writerows(zip(range(len(native['labels'])),native['probabilities'],native['labels']))
        print('DONE',split,flush=True)
    groups={}
    for row in records:groups.setdefault((row['split'],row['method'],row['strategy']),[]).append(row)
    summary=[{'split':k[0],'method':k[1],'strategy':k[2],'metrics':aggregate(v)} for k,v in groups.items()]
    files=['src/analysis/ssl_pitch_shift.py','src/analysis/ssl_pitch_shift_methods.py',
           'src/core/amdf.py','src/core/cepstrum.py','scripts/31_compare_native_pitch_shift.py']
    document={'settings':settings,'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        'strength':{'acf':'clip(r_max,0,1)','amdf':'clip(1-d_min,0,1)',
                    'cepstrum':'max(c_peak,0)/(max(c_peak,0)+frozen_champion_threshold_0.1)'},
        'legacy_shared_acf_opt_in':include_legacy_shared_acf,'alpha':1.25,'sigma':.12,'decision_threshold':.5,'summary':summary,'files':records}
    (output/'results.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'per_file.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lookup={(r['split'],r['method'],r['strategy']):r['metrics']['final_score'] for r in summary}
    lines=['# Pitch-Shift dùng đúng thuật toán của từng nhánh','',
        'Mặc định mỗi nhánh dùng ACF–ACF, AMDF–AMDF hoặc Cepstrum–Cepstrum cho hai khung trước/sau resample. Không dùng model/Otsu/K-means. Bản ACF-only lịch sử chỉ chạy khi chỉ định --include-legacy-shared-acf, không đưa vào bảng mặc định.',
        '', 'Giữ alpha=1,25, sigma=0,12, gate STE=0,01 và ngưỡng điểm 0,5. Điểm = exp(-relative_error²/(2 sigma²)) × độ tuần hoàn riêng. AMDF dùng 1-d_min; Cepstrum dùng c_positive/(c_positive+0,1), với 0,1 là ngưỡng Champion đã chốt, không fit theo test. ACF gọi trực tiếp bản cũ nên khớp chính xác.',
        '', 'Quyết định Pitch-Shift dùng khung tín hiệu gốc như pipeline cũ. F0 đầu ra và các plugin tiền/hậu xử lý của từng Champion giữ nguyên. Final Score (%) thấp hơn tốt hơn, metric legacy-v1.',
        '', '| Phương pháp | ACF train → test | AMDF train → test | Cepstrum train → test |','|---|---:|---:|---:|']
    for strategy in (['champion','pitch_shift_own_method']+(['pitch_shift_shared_acf'] if include_legacy_shared_acf else [])):
        cells=[' → '.join(f"{lookup[split,method,strategy]:.3f}" for split in ('train','test')) for method in ('acf','amdf','cepstrum')]
        lines.append('| '+strategy+' | '+' | '.join(cells)+' |')
    lines+=['','Không chọn lại tham số dựa vào test; điểm mới có thể kém đối chứng. [CSV từng file](per_file.csv) · [Kết quả đầy đủ](results.json).','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    if sys.platform=='win32':sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--include-legacy-shared-acf',action='store_true',help='Explicit historical cross-method control only')
    args=parser.parse_args();run(args.output_dir,args.include_legacy_shared_acf)
