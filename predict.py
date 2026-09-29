#!/usr/bin/env python3
"""PREDICT & TEST CLI TOOL - DÀNH CHO BUỔI BẢO VỆ / THỰC NGHIỆM TRÊN DỮ LIỆU CỦA THẦY
Hỗ trợ kiểm thử tức thì trên 1 file .wav bất kỳ hoặc toàn bộ thư mục âm thanh.
Tự động kích hoạt các cấu hình Quán quân tối ưu nhất (ACF Champion, AMDF Champion, YIN Champion).
Tự động đối soát nhãn Ground Truth (.lab) nếu có cùng tên trong thư mục.

Ví dụ sử dụng:
1. Chạy trên 1 file âm thanh bất kỳ:
   uv run python predict.py --input path/to/audio.wav

2. Chạy trên toàn bộ thư mục âm thanh của thầy:
   uv run python predict.py --input path/to/folder_thay/

3. Chỉ chạy riêng thuật toán tốt nhất (YIN hoặc ACF hoặc AMDF):
   uv run python predict.py --input audio.wav --method yin
"""
import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple
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
from src.plugins import (
    PluginPitchDetector,
    BandpassFilterPlugin,
    CenterClippingPlugin,
    EnergyExtensionPlugin,
    ViterbiTrackingPlugin,
)
from src.analysis.evaluation import evaluate_against_ground_truth


def get_champion_detectors() -> Dict[str, PitchDetector]:
    """Khởi tạo 3 hệ thống Quán quân tối ưu nhất của đề tài."""
    # 1. ACF Champion: Config 29 (Bandpass + Center Clipping + Energy Extension + Viterbi)
    acf_champ = PluginPitchDetector(
        base_detector=PitchDetector(method="acf", threshold=0.395257),
        plugins=[
            BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
            CenterClippingPlugin(clipping_ratio=0.40),
            EnergyExtensionPlugin(threshold_discount=0.20),
            ViterbiTrackingPlugin(w_freq=3.0, w_octave=2.0),
        ],
    )

    # 2. AMDF Champion: Config 13 (Center Clipping + Viterbi)
    amdf_champ = PluginPitchDetector(
        base_detector=PitchDetector(method="amdf", threshold=0.648843),
        plugins=[
            CenterClippingPlugin(clipping_ratio=0.40),
            ViterbiTrackingPlugin(w_freq=3.0, w_octave=2.0),
        ],
    )

    # 3. YIN Champion: Enhanced YIN (Viterbi Tracking)
    yin_champ = PluginPitchDetector(
        base_detector=PitchDetector(method="yin", threshold=0.250000),
        plugins=[
            ViterbiTrackingPlugin(w_freq=3.0, w_octave=2.0),
        ],
    )

    return {
        "YIN Champion": yin_champ,
        "ACF Champion": acf_champ,
        "AMDF Champion": amdf_champ,
    }


def get_baseline_detectors() -> Dict[str, PitchDetector]:
    """Khởi tạo 3 hệ thống Baseline nguyên bản (không plugin)."""
    return {
        "YIN Baseline": PitchDetector(method="yin", threshold=0.250000),
        "ACF Baseline": PitchDetector(method="acf", threshold=0.440788),
        "AMDF Baseline": PitchDetector(method="amdf", threshold=0.437976),
    }


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
    frame_times = None

    print(f"\n{'='*80}")
    print(f"  ĐANG PHÂN TÍCH: {os.path.basename(wav_path)} (Thời lượng: {duration:.2f}s | Fs: {sample_rate} Hz)")
    if has_ground_truth:
        print(f"  [*] Đã tìm thấy nhãn Ground Truth: {os.path.basename(lab_path)}")
    else:
        print(f"  [*] Không có nhãn .lab đi kèm (Chế độ dự đoán mù / Blind Prediction)")
    print(f"{'='*80}")

    for name, detector in detectors.items():
        res = detector.process_signal(signal, sample_rate)
        if frame_times is None:
            frame_times = res["frame_times"]

        f0_raw = res["f0_contour"]
        f0_contours[name] = f0_raw
        voiced_mask = res["labels"] == "v"
        voiced_f0 = f0_raw[voiced_mask]

        f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        voiced_ratio = (np.sum(voiced_mask) / len(voiced_mask)) * 100.0 if len(voiced_mask) > 0 else 0.0

        metrics = {
            "pred_f0_mean": round(f0_mean, 2),
            "pred_f0_std": round(f0_std, 2),
            "voiced_ratio_pct": round(voiced_ratio, 2),
            "voiced_frames": int(np.sum(voiced_mask)),
            "total_frames": len(voiced_mask),
        }

        # Nếu có ground truth, tính sai số
        eval_res = None
        if has_ground_truth:
            eval_res = evaluate_against_ground_truth(res, lab_path)
            metrics["ref_f0_mean"] = round(eval_res["ref_f0_mean"], 2)
            metrics["ref_f0_std"] = round(eval_res["ref_f0_std"], 2)
            metrics["abs_error_mean"] = round(eval_res["abs_error_mean"], 2)
            metrics["rel_error_mean_pct"] = round(eval_res["rel_error_mean_pct"], 2)
            metrics["mape_f0"] = round(eval_res["mape_f0"], 2)
            metrics["mape_mean"] = round(eval_res["mape_mean"], 2)
            metrics["abs_error_std"] = round(eval_res["abs_error_std"], 2)
            metrics["rel_error_std_pct"] = round(eval_res["rel_error_std_pct"], 2)
            metrics["mape_std"] = round(eval_res["mape_std"], 2)
            metrics["accuracy"] = round(eval_res["classification_accuracy"], 2) if eval_res.get("classification_accuracy") is not None else None
            metrics["voiced_f1"] = round(eval_res["voiced_f1"], 2) if eval_res.get("voiced_f1") is not None else None
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
        print(f"  * Tần số F0 dự đoán (Pred) : {f0_mean:.2f} Hz ± {f0_std:.2f} Hz (Mean ± Std)")
        print(f"  * Tỉ lệ Hữu thanh (V)       : {voiced_ratio:.1f}% ({np.sum(voiced_mask)}/{len(voiced_mask)} khung)")
        if has_ground_truth:
            ref_m = metrics["ref_f0_mean"]
            ref_s = metrics["ref_f0_std"]
            print(f"  * Nhãn chuẩn .lab (GT)      : {ref_m:.2f} Hz ± {ref_s:.2f} Hz (F0mean ± F0std)")
            print(f"  * Sai số F0mean (|ΔMean|)   : {metrics['abs_error_mean']:.2f} Hz (MAPE F0mean: {metrics['mape_f0']:.2f}%)")
            print(f"  * Sai số F0std  (|ΔStd|)    : {metrics['abs_error_std']:.2f} Hz (MAPE F0std : {metrics['mape_std']:.2f}%)")
            if "ref_f0_num" in metrics and metrics["ref_f0_num"] is not None:
                print(f"  * Số khung Voiced (F0num)   : Pred={metrics['pred_f0_num']} vs Ref={metrics['ref_f0_num']} (Lệch: {metrics['abs_error_num']} khung | MAPE F0num: {metrics['mape_num']:.2f}%)")
            if "composite_score" in metrics:
                print(f"  * ĐIỂM LỖI TỔNG HỢP (Composite Score): {metrics['composite_score']:.2f}% (TB các MAPE)")
            if metrics.get("accuracy") is not None:
                print(f"  * Độ chính xác V/UV         : {metrics['accuracy']:.2f}% | Voiced F1-Score: {metrics['voiced_f1']:.2f}%")

    # 1. Xuất file CSV dự đoán
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{base_name}_pitch_prediction.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            headers = ["Frame_Index", "Time_Sec"] + [f"{n}_F0_Hz" for n in detectors.keys()]
            f.write(",".join(headers) + "\n")
            for i, t in enumerate(frame_times):
                row = [str(i), f"{t:.3f}"]
                for n in detectors.keys():
                    val = f0_contours[n][i]
                    row.append(f"{val:.2f}")
                f.write(",".join(row) + "\n")
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
        ax0.set_title(f"Dạng sóng âm thanh: {base_name}.wav (Thời lượng: {duration:.2f}s)", fontsize=11, fontweight="bold")
        ax0.set_xlabel("Thời gian (giây)", fontsize=10)
        ax0.set_ylabel("Biên độ", fontsize=10)
        ax0.set_xlim(0, max(time_sig))
        ax0.grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Quỹ đạo cao độ F0
        ax1 = fig.add_subplot(gs[1])
        colors = {"YIN Champion": "#2ca02c", "ACF Champion": "#1f77b4", "AMDF Champion": "#d62728",
                  "YIN Baseline": "#98df8a", "ACF Baseline": "#aec7e8", "AMDF Baseline": "#ff9896"}
        styles = {"YIN Champion": "-", "ACF Champion": "--", "AMDF Champion": "-."}

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
            if ref_f0 > 0:
                ax1.axhline(ref_f0, color="black", linestyle="--", linewidth=1.5, label=f"GT F0mean = {ref_f0:.1f} Hz")
                if ref_std > 0:
                    ax1.axhspan(max(0, ref_f0 - ref_std), ref_f0 + ref_std, color="black", alpha=0.06, label=f"GT Dải chuẩn F0mean ± F0std ({ref_f0:.1f} ± {ref_std:.1f} Hz)")

        for n, contour in f0_contours.items():
            c = colors.get(n, "#333333")
            s = styles.get(n, "-")
            mask = contour > 0
            v_times = frame_times[mask]
            v_f0 = contour[mask]
            stat = file_results[n]
            lbl = f"{n}: {stat['pred_f0_mean']:.1f} ± {stat['pred_f0_std']:.1f} Hz"
            if has_ground_truth:
                f1_part = f", F1: {stat['voiced_f1']:.1f}%" if stat.get('voiced_f1') is not None else ""
                score_part = f", Score: {stat['composite_score']:.1f}%" if stat.get('composite_score') is not None else ""
                lbl += f" (|ΔM|: {stat['abs_error_mean']:.2f}Hz, |ΔS|: {stat['abs_error_std']:.2f}Hz{score_part}{f1_part})"
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
    parser.add_argument("--method", "-m", type=str, default="champions", choices=["champions", "all", "yin", "acf", "amdf", "baseline"], help="Chế độ thuật toán (mặc định: 'champions' gồm 3 Quán quân tốt nhất)")
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
    elif args.method == "yin":
        detectors = {"YIN Champion": champs["YIN Champion"], "YIN Baseline": bases["YIN Baseline"]}
    elif args.method == "acf":
        detectors = {"ACF Champion": champs["ACF Champion"], "ACF Baseline": bases["ACF Baseline"]}
    elif args.method == "amdf":
        detectors = {"AMDF Champion": champs["AMDF Champion"], "AMDF Baseline": bases["AMDF Baseline"]}
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
        res = process_single_file(
            wav_path=wav_f,
            detectors=detectors,
            output_dir=args.output_dir,
            save_plot=True,
            save_csv=True,
        )
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
                    str(mval.get("ref_f0_mean", "")),
                    str(mval.get("pred_f0_mean", "")),
                    str(mval.get("abs_error_mean", "")),
                    str(mval.get("mape_f0", mval.get("rel_error_mean_pct", ""))),
                    str(mval.get("ref_f0_std", "")),
                    str(mval.get("pred_f0_std", "")),
                    str(mval.get("abs_error_std", "")),
                    str(mval.get("mape_std", mval.get("rel_error_std_pct", ""))),
                    str(mval.get("ref_f0_num", "")),
                    str(mval.get("pred_f0_num", "")),
                    str(mval.get("abs_error_num", "")),
                    str(mval.get("mape_num", "")),
                    str(mval.get("composite_score", "")),
                    str(mval.get("voiced_ratio_pct", "")),
                    str(mval.get("accuracy", "") if mval.get("accuracy") is not None else ""),
                    str(mval.get("voiced_f1", "") if mval.get("voiced_f1") is not None else ""),
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
            print(f"  - Tần số F0 dự đoán trung bình  : {np.mean(means):.2f} Hz ± {np.mean(stds):.2f} Hz (Mean ± Std)")
            print(f"  - Tỉ lệ phân đoạn Voiced TB     : {np.mean(voiced_ratios):.2f}%")

            # Nếu tất cả các file đều có ground truth
            has_err = all("abs_error_mean" in all_summary[f][sys_name] for f in all_summary.keys())
            if has_err:
                ref_means = [all_summary[f][sys_name]["ref_f0_mean"] for f in all_summary.keys()]
                ref_stds = [all_summary[f][sys_name]["ref_f0_std"] for f in all_summary.keys()]
                err_means = [all_summary[f][sys_name]["abs_error_mean"] for f in all_summary.keys()]
                mape_means = [all_summary[f][sys_name].get("mape_f0", all_summary[f][sys_name].get("rel_error_mean_pct", 0.0)) for f in all_summary.keys()]
                err_stds = [all_summary[f][sys_name]["abs_error_std"] for f in all_summary.keys()]
                mape_stds = [all_summary[f][sys_name].get("mape_std", all_summary[f][sys_name].get("rel_error_std_pct", 0.0)) for f in all_summary.keys()]
                
                f1_vals = [all_summary[f][sys_name]["voiced_f1"] for f in all_summary.keys() if all_summary[f][sys_name].get("voiced_f1") is not None]
                acc_vals = [all_summary[f][sys_name]["accuracy"] for f in all_summary.keys() if all_summary[f][sys_name].get("accuracy") is not None]
                
                print(f"  - Chuẩn file .lab trung bình    : {np.mean(ref_means):.2f} Hz ± {np.mean(ref_stds):.2f} Hz (F0mean ± F0std)")
                print(f"  - Sai số tuyệt đối F0mean TB    : {np.mean(err_means):.2f} Hz (MAPE F0mean TB: {np.mean(mape_means):.2f}%)")
                print(f"  - Sai số độ lệch chuẩn F0std TB : {np.mean(err_stds):.2f} Hz (MAPE F0std TB : {np.mean(mape_stds):.2f}%)")

                num_files = [f for f in all_summary.keys() if "ref_f0_num" in all_summary[f][sys_name] and all_summary[f][sys_name]["ref_f0_num"] is not None]
                if len(num_files) > 0:
                    ref_nums = [all_summary[f][sys_name]["ref_f0_num"] for f in num_files]
                    pred_nums = [all_summary[f][sys_name]["pred_f0_num"] for f in num_files]
                    err_nums = [all_summary[f][sys_name]["abs_error_num"] for f in num_files]
                    mape_nums = [all_summary[f][sys_name]["mape_num"] for f in num_files]
                    print(f"  - Số khung Voiced (F0num) TB    : Pred={np.mean(pred_nums):.1f} vs Ref={np.mean(ref_nums):.1f} (|ΔNum|={np.mean(err_nums):.1f} khung | MAPE F0num TB: {np.mean(mape_nums):.2f}%)")
                    composite_score_mean = (np.mean(mape_means) + np.mean(mape_stds) + np.mean(mape_nums)) / 3.0
                    print(f"  >>> ĐIỂM SỐ TỔNG HỢP (FINAL COMPOSITE SCORE - TB 3 MAPE): {composite_score_mean:.2f}% (Độ chính xác tương đối: {100.0 - composite_score_mean:.2f}%) <<<")
                else:
                    composite_score_mean = (np.mean(mape_means) + np.mean(mape_stds)) / 2.0
                    print(f"  >>> ĐIỂM SỐ TỔNG HỢP (FINAL COMPOSITE SCORE - TB 2 MAPE): {composite_score_mean:.2f}% <<<")

                if len(f1_vals) > 0:
                    print(f"  - Voiced F1-Score TB            : {np.mean(f1_vals):.2f}%")
                if len(acc_vals) > 0:
                    print(f"  - V/UV Accuracy TB              : {np.mean(acc_vals):.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()
