#!/usr/bin/env python3
"""PREDICT & TEST CLI TOOL - DÀNH CHO BUỔI BẢO VỆ / THỰC NGHIỆM TRÊN DỮ LIỆU CỦA THẦY
Hỗ trợ kiểm thử tức thì trên 1 file .wav bất kỳ hoặc toàn bộ thư mục âm thanh.
Tự động kích hoạt các cấu hình Champion đã chọn trên train (ACF, AMDF, Cepstrum).
Tự động đối soát nhãn Ground Truth (.lab) nếu có cùng tên trong thư mục.

Ví dụ sử dụng:
1. Chạy trên 1 file âm thanh bất kỳ:
   uv run python predict.py --input path/to/audio.wav

2. Chạy trên toàn bộ thư mục âm thanh của thầy:
   uv run python predict.py --input path/to/folder_thay/

3. Chỉ chạy riêng thuật toán tốt nhất (Cepstrum hoặc ACF hoặc AMDF):
   uv run python predict.py --input audio.wav --method cepstrum
"""
import argparse
import csv
import glob
import os
import sys
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

# Đảm bảo in tiếng Việt chuẩn trên Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.audio import load_wav
from src.core.pitch_detector import PitchDetector
from src.pipelines import build_baseline, build_champion
from src.metrics.evaluation import evaluate_against_ground_truth
from src.metrics.aggregation import aggregate_rows, ROUNDED_MACRO
from src.metrics.numeric import round_metric, mean_metric
from src.metrics.formatting import format_metric, csv_value


def get_champion_detectors() -> Dict[str, PitchDetector]:
    """Frozen train-selected Champion configurations shared with benchmark 14."""
    return {f"{method.title() if method == 'cepstrum' else method.upper()} Champion": build_champion(method)
            for method in ('acf', 'amdf', 'cepstrum')}


def get_baseline_detectors() -> Dict[str, PitchDetector]:
    """Baseline configurations shared with the public API and benchmarks."""
    return {f"{method.title() if method == 'cepstrum' else method.upper()} Baseline": build_baseline(method)
            for method in ('acf', 'amdf', 'cepstrum')}


def process_single_file(
    wav_path: str,
    detectors: Dict[str, PitchDetector],
    output_dir: str,
    save_plot: bool = True,
    save_csv: bool = True,
) -> Dict[str, dict]:
    """Xử lý 1 file âm thanh và trả về kết quả dự đoán cùng đánh giá nếu có ground truth."""
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    lab_path = os.path.splitext(wav_path)[0] + ".lab"
    has_ground_truth = os.path.exists(lab_path)

    sample_rate, signal, duration = load_wav(wav_path)
    time_sig = np.arange(len(signal)) / sample_rate

    file_results = {}
    f0_contours = {}
    frame_times_by_method = {}
    labels_by_method = {}

    print(f"\n{'='*80}")
    print(f"  ĐANG PHÂN TÍCH: {os.path.basename(wav_path)} (Thời lượng: {format_metric(duration, '.2f')}s | Fs: {sample_rate} Hz)")
    if has_ground_truth:
        print(f"  [*] Đã tìm thấy nhãn Ground Truth: {os.path.basename(lab_path)}")
    else:
        print(f"  [*] Không có nhãn .lab đi kèm (Chế độ dự đoán mù / Blind Prediction)")
    print(f"{'='*80}")

    for name, detector in detectors.items():
        res = detector.process_signal(signal, sample_rate)
        frame_times_by_method[name] = res['frame_times']
        labels_by_method[name] = res['labels']

        f0_raw = res["f0_contour"]
        f0_contours[name] = f0_raw
        voiced_mask = res["labels"] == "v"
        voiced_f0 = f0_raw[voiced_mask]

        f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        voiced_ratio = (np.sum(voiced_mask) / len(voiced_mask)) * 100.0 if len(voiced_mask) > 0 else 0.0

        metrics = {
            "pred_f0_mean": round_metric(f0_mean, 2),
            "pred_f0_std": round_metric(f0_std, 2),
            "voiced_ratio_pct": round_metric(voiced_ratio, 2),
            "voiced_frames": int(np.sum(voiced_mask)),
            "total_frames": len(voiced_mask),
        }

        # Nếu có ground truth, tính sai số
        eval_res = None
        if has_ground_truth:
            eval_res = evaluate_against_ground_truth(res, lab_path)
            metrics["ref_f0_mean"] = round_metric(eval_res["ref_f0_mean"], 2)
            metrics["ref_f0_std"] = round_metric(eval_res["ref_f0_std"], 2)
            metrics["abs_error_mean"] = round_metric(eval_res["abs_error_mean"], 2)
            metrics["rel_error_mean_pct"] = round_metric(eval_res["rel_error_mean_pct"], 2)
            metrics["mape_f0"] = round_metric(eval_res["mape_f0"], 2)
            metrics["mape_mean"] = round_metric(eval_res["mape_mean"], 2)
            metrics["abs_error_std"] = round_metric(eval_res["abs_error_std"], 2)
            metrics["rel_error_std_pct"] = round_metric(eval_res["rel_error_std_pct"], 2)
            metrics["mape_std"] = round_metric(eval_res["mape_std"], 2)
            metrics["accuracy"] = round_metric(eval_res["classification_accuracy"], 2) if eval_res.get("classification_accuracy") is not None else None
            metrics["voiced_f1"] = round_metric(eval_res["voiced_f1"], 2) if eval_res.get("voiced_f1") is not None else None
            if "pred_f0_num" in eval_res and eval_res.get("ref_f0_num") is not None:
                metrics["pred_f0_num"] = eval_res["pred_f0_num"]
                metrics["ref_f0_num"] = eval_res["ref_f0_num"]
                metrics["abs_error_num"] = eval_res["abs_error_num"]
                metrics["mape_num"] = eval_res["mape_num"]
            if "composite_score" in eval_res:
                metrics["composite_score"] = eval_res["composite_score"]

        file_results[name] = metrics

        # In kết quả dạng liệt kê rõ ràng
        print(f"\n--- {name.upper()} ---")
        print(f"  * Tần số F0 dự đoán (Pred) : {format_metric(f0_mean, '.2f')} Hz ± {format_metric(f0_std, '.2f')} Hz (Mean ± Std)")
        print(f"  * Tỉ lệ Hữu thanh (V)       : {format_metric(voiced_ratio, '.1f')}% ({np.sum(voiced_mask)}/{len(voiced_mask)} khung)")
        if has_ground_truth:
            ref_m = metrics["ref_f0_mean"]
            ref_s = metrics["ref_f0_std"]
            print(f"  * Nhãn chuẩn .lab (GT)      : {format_metric(ref_m, '.2f')} Hz ± {format_metric(ref_s, '.2f')} Hz (F0mean ± F0std)")
            print(f"  * Sai số F0mean (|ΔMean|)   : {format_metric(metrics['abs_error_mean'], '.2f')} Hz (MAPE F0mean: {format_metric(metrics['mape_f0'], '.2f')}%)")
            print(f"  * Sai số F0std  (|ΔStd|)    : {format_metric(metrics['abs_error_std'], '.2f')} Hz (MAPE F0std : {format_metric(metrics['mape_std'], '.2f')}%)")
            if "ref_f0_num" in metrics and metrics["ref_f0_num"] is not None:
                print(f"  * Số khung Voiced (F0num)   : Pred={metrics['pred_f0_num']} vs Ref={metrics['ref_f0_num']} (Lệch: {metrics['abs_error_num']} khung | MAPE F0num: {format_metric(metrics['mape_num'], '.2f')}%)")
            if "composite_score" in metrics:
                print(f"  * ĐIỂM LỖI TỔNG HỢP (Composite Score): {format_metric(metrics['composite_score'], '.2f')}% (TB các MAPE)")
            if metrics.get("accuracy") is not None:
                print(f"  * Độ chính xác V/UV         : {format_metric(metrics['accuracy'], '.2f')}% | Voiced F1-Score: {format_metric(metrics['voiced_f1'], '.2f')}%")

    # 1. Xuất file CSV dự đoán
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{base_name}_pitch_prediction.csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Frame_Index', 'Time_Sec', 'Label', 'F0_Hz'])
            for name, contour in f0_contours.items():
                writer.writerows((name, i, format(float(t), '.6f'), label, format(float(f0), '.6f'))
                                 for i, (t, label, f0) in enumerate(zip(
                                     frame_times_by_method[name], labels_by_method[name], contour)))
        print(f"\n[+] Đã lưu bảng số liệu F0 chi tiết từng khung: {csv_path}")

    # 2. Xuất biểu đồ trực quan
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{base_name}_pitch_contour.png")
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.8], hspace=0.30)

        # Panel 1: Dạng sóng âm thanh
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(time_sig, signal, color="#2b5c8f", linewidth=0.8, alpha=0.9)
        ax0.set_title(f"Dạng sóng âm thanh: {base_name}.wav (Thời lượng: {format_metric(duration, '.2f')}s)", fontsize=11, fontweight="bold")
        ax0.set_xlabel("Thời gian (giây)", fontsize=10)
        ax0.set_ylabel("Biên độ", fontsize=10)
        ax0.set_xlim(0, max(time_sig))
        ax0.grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Quỹ đạo cao độ F0
        ax1 = fig.add_subplot(gs[1])
        colors = {"ACF Champion": "#1f77b4", "AMDF Champion": "#d62728", "Cepstrum Champion": "#2ca02c",
                  "ACF Baseline": "#aec7e8", "AMDF Baseline": "#ff9896", "Cepstrum Baseline": "#98df8a"}
        styles = {"ACF Champion": "-", "AMDF Champion": "--", "Cepstrum Champion": "-."}

        # Nếu có ground truth, vẽ vùng chuẩn
        if has_ground_truth and eval_res is not None:
            ref_f0 = eval_res["ref_f0_mean"]
            ref_std = eval_res["ref_f0_std"]
            for seg in eval_res.get("gt_segments", []):
                if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    start_s, end_s, lbl = seg[0], seg[1], seg[2]
                elif isinstance(seg, dict):
                    start_s, end_s, lbl = seg.get("start", 0), seg.get("end", 0), seg.get("label", "")
                else:
                    continue
                if lbl.lower() == "v":
                    ax1.axvspan(start_s, end_s, color="#2ca02c", alpha=0.10, label="Khoảng Hữu thanh Chuẩn (GT)" if "Khoảng Hữu thanh Chuẩn (GT)" not in ax1.get_legend_handles_labels()[1] else "")
            if ref_f0 is not None and ref_f0 > 0:
                ax1.axhline(ref_f0, color="black", linestyle="--", linewidth=1.5, label=f"GT F0mean = {format_metric(ref_f0, '.1f')} Hz")
                if ref_std is not None and ref_std > 0:
                    ax1.axhspan(max(0, ref_f0 - ref_std), ref_f0 + ref_std, color="black", alpha=0.06, label=f"GT Dải chuẩn F0mean ± F0std ({format_metric(ref_f0, '.1f')} ± {format_metric(ref_std, '.1f')} Hz)")

        for n, contour in f0_contours.items():
            c = colors.get(n, "#333333")
            s = styles.get(n, "-")
            v_times = frame_times_by_method[n]
            v_f0 = np.where(contour > 0, contour, np.nan)
            stat = file_results[n]
            lbl = f"{n}: {format_metric(stat['pred_f0_mean'], '.1f')} ± {format_metric(stat['pred_f0_std'], '.1f')} Hz"
            if has_ground_truth:
                f1_part = f", F1: {format_metric(stat['voiced_f1'], '.1f')}%" if stat.get('voiced_f1') is not None else ""
                score_part = f", Score: {format_metric(stat['composite_score'], '.1f')}%" if stat.get('composite_score') is not None else ""
                lbl += f" (|ΔM|: {format_metric(stat['abs_error_mean'], '.2f')}Hz, |ΔS|: {format_metric(stat['abs_error_std'], '.2f')}Hz{score_part}{f1_part})"
            ax1.plot(v_times, v_f0, s, color=c, linewidth=2.0, alpha=0.9, label=lbl)

        ax1.set_title("Quỹ đạo Tần số cơ bản F0 (Hz) qua các thuật toán", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Thời gian (giây)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("F0 (Hz)", fontsize=10, fontweight="bold")
        ax1.set_ylim(50, 420)
        ax1.set_xlim(0, max(time_sig))
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True, fontsize=9.5)

        plt.suptitle(f"KẾT QUẢ ĐO CAO ĐỘ PITCH: {base_name.upper()}", fontsize=13, fontweight="bold", y=0.98)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] Đã lưu ảnh biểu đồ trực quan: {plot_path}")

    return file_results


def main():
    parser = argparse.ArgumentParser(description="Pitch Prediction & Testing CLI on Custom Data.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Đường dẫn tới 1 file .wav hoặc thư mục chứa các file .wav")
    parser.add_argument("--method", "-m", type=str, default="champions", choices=["champions", "all", "acf", "amdf", "cepstrum", "baseline"], help="Chế độ thuật toán (mặc định: 'champions' đã chọn trên train)")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/predictions", help="Thư mục xuất kết quả (ảnh PNG và bảng CSV)")
    args = parser.parse_args()

    # Chọn tập thuật toán
    champs = get_champion_detectors()
    bases = get_baseline_detectors()

    if args.method == "champions":
        detectors = champs
    elif args.method == "baseline":
        detectors = bases
    elif args.method == "all":
        detectors = {**champs, **bases}
    elif args.method == "acf":
        detectors = {"ACF Champion": champs["ACF Champion"], "ACF Baseline": bases["ACF Baseline"]}
    elif args.method == "amdf":
        detectors = {"AMDF Champion": champs["AMDF Champion"], "AMDF Baseline": bases["AMDF Baseline"]}
    elif args.method == "cepstrum":
        detectors = {"Cepstrum Champion": champs["Cepstrum Champion"], "Cepstrum Baseline": bases["Cepstrum Baseline"]}
    else:
        detectors = champs

    # Tìm các file âm thanh
    if os.path.isfile(args.input):
        wav_files = [args.input]
    elif os.path.isdir(args.input):
        wav_files = sorted(list(set(glob.glob(os.path.join(args.input, "*.wav")) + glob.glob(os.path.join(args.input, "*.WAV")))))
        if len(wav_files) == 0:
            print(f"[!] Không tìm thấy file .wav nào trong thư mục: {args.input}")
            sys.exit(1)
    else:
        print(f"[!] Đường dẫn không tồn tại: {args.input}")
        sys.exit(1)

    print("\n" + "#" * 80)
    print(f"  HỆ THỐNG ĐO CAO ĐỘ PITCH TRACKING - CHẾ ĐỘ KIỂM THỬ TỔNG HỢP")
    print(f"  Tổng số file cần phân tích: {len(wav_files)}")
    print(f"  Các thuật toán tham gia: {', '.join(detectors.keys())}")
    print(f"  Thư mục xuất báo cáo: {os.path.abspath(args.output_dir)}")
    print("#" * 80)

    all_summary = {}
    for wav_f in wav_files:
        try:
            res = process_single_file(
                wav_path=wav_f,
                detectors=detectors,
                output_dir=args.output_dir,
                save_plot=True,
                save_csv=True,
            )
        except ValueError as exc:
            parser.exit(2, f"[!] {os.path.basename(wav_f)}: {exc}\n")
        all_summary[os.path.basename(wav_f)] = res

    # Xuất file CSV tổng hợp đánh giá (nếu có ground truth)
    summary_csv_path = os.path.join(args.output_dir, "evaluation_summary_metrics.csv")
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        headers = [
            "File", "Method", "Ref_F0mean", "Pred_F0mean", "Abs_Err_F0mean", "MAPE_F0mean_pct",
            "Ref_F0std", "Pred_F0std", "Abs_Err_F0std", "MAPE_F0std_pct",
            "Ref_F0num", "Pred_F0num", "Abs_Err_F0num", "MAPE_F0num_pct",
            "Composite_Score_pct", "Voiced_Ratio_pct", "V_UV_Accuracy", "Voiced_F1"
        ]
        f.write(",".join(headers) + "\n")
        for fname, fdict in all_summary.items():
            for mname, mval in fdict.items():
                row = [
                    fname,
                    mname,
                    csv_value(mval.get("ref_f0_mean", "")),
                    csv_value(mval.get("pred_f0_mean", "")),
                    csv_value(mval.get("abs_error_mean", "")),
                    csv_value(mval.get("mape_f0", mval.get("rel_error_mean_pct", ""))),
                    csv_value(mval.get("ref_f0_std", "")),
                    csv_value(mval.get("pred_f0_std", "")),
                    csv_value(mval.get("abs_error_std", "")),
                    csv_value(mval.get("mape_std", mval.get("rel_error_std_pct", ""))),
                    csv_value(mval.get("ref_f0_num", "")),
                    csv_value(mval.get("pred_f0_num", "")),
                    csv_value(mval.get("abs_error_num", "")),
                    csv_value(mval.get("mape_num", "")),
                    csv_value(mval.get("composite_score", "")),
                    csv_value(mval.get("voiced_ratio_pct", "")),
                    csv_value(mval.get("accuracy", "") if mval.get("accuracy") is not None else ""),
                    csv_value(mval.get("voiced_f1", "") if mval.get("voiced_f1") is not None else ""),
                ]
                f.write(",".join(row) + "\n")
    print(f"\n[+] Đã xuất file thống kê tổng hợp (F0mean, F0std, F0num & Composite Score): {summary_csv_path}")

    # Báo cáo tổng hợp nếu chạy nhiều hơn 1 file
    if len(wav_files) > 1:
        print("\n" + "=" * 80)
        print("  TỔNG KẾT TRUNG BÌNH TOÀN BỘ TẬP KIỂM THỬ (F0 MEAN, F0 STD & F0 NUM - KÈM MAPE)")
        print("=" * 80)
        for sys_name in detectors.keys():
            means = [all_summary[f][sys_name]["pred_f0_mean"] for f in all_summary.keys()]
            stds = [all_summary[f][sys_name]["pred_f0_std"] for f in all_summary.keys()]
            voiced_ratios = [all_summary[f][sys_name]["voiced_ratio_pct"] for f in all_summary.keys()]
            print(f"\n* {sys_name}:")
            print(f"  - Tần số F0 dự đoán trung bình  : {format_metric(np.mean(means), '.2f')} Hz ± {format_metric(np.mean(stds), '.2f')} Hz (Mean ± Std)")
            print(f"  - Tỉ lệ phân đoạn Voiced TB     : {format_metric(np.mean(voiced_ratios), '.2f')}%")

            rows = [all_summary[f][sys_name] for f in all_summary]
            if all("abs_error_mean" in row for row in rows):
                summary = aggregate_rows([{**row, 'final_score': row.get('composite_score')} for row in rows], ROUNDED_MACRO)
                for label, key in [("Sai số tuyệt đối F0mean TB (Hz)", "abs_error_mean"),
                                   ("MAPE F0mean TB (%)", "mape_f0"),
                                   ("Sai số tuyệt đối F0std TB (Hz)", "abs_error_std"),
                                   ("MAPE F0std TB (%)", "mape_std"),
                                   ("MAPE F0num TB (%)", "mape_num")]:
                    print(f"  - {label}: {format_metric(summary[key])}")
                score = summary['final_score']
                print(f"  >>> FINAL SCORE (%; TB đủ 3 MAPE, thấp hơn tốt hơn): {format_metric(score)} <<<")
                for label, key in [("Voiced F1", "voiced_f1"), ("V/UV Accuracy", "accuracy")]:
                    values = [row[key] for row in rows if row.get(key) is not None]
                    if values:
                        print(f"  - {label}: {format_metric(mean_metric(values))}% ({len(values)}/{len(rows)} file có nhãn thời gian)")
        print("=" * 80)


if __name__ == "__main__":
    main()
