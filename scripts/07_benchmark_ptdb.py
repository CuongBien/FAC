#!/usr/bin/env python3
"""Script 07: Đánh giá chuẩn quốc tế (Benchmark) trên tập dữ liệu PTDB-TUG.

Đo đạc các chỉ số chuẩn hóa trong xử lý tiếng nói:
- VDE (Voicing Decision Error, %): Tỉ lệ lỗi quyết định Voiced / Unvoiced.
- GPE (Gross Pitch Error, %): Tỉ lệ lỗi thô khi sai lệch F0 > 20% so với Ground Truth Laryngograph (EGG).
- FFE (F0 Frame Error, %): Tỉ lệ khung lỗi tổng hợp (VDE + GPE).
- FPE_MAE (Fine Pitch Error, Hz): Sai số tuyệt đối trung bình trên các khung chuẩn (< 20%).

So sánh đối đầu giữa 4 hệ thống:
1. Baseline ACF (Không plugins)
2. Enhanced ACF (Toàn bộ plugins: Bandpass, CenterClip, Hysteresis, EnergyExtension, Viterbi)
3. Baseline AMDF (Không plugins)
4. Enhanced AMDF (Toàn bộ plugins)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.pitch_detector import PitchDetector
from src.core.ptdb_loader import (
    find_ptdb_dataset,
    load_ptdb_utterance,
    align_predictions_to_ground_truth,
)
from src.plugins.plugin_detector import PluginPitchDetector
from src.plugins.pre_processing import BandpassFilterPlugin, CenterClippingPlugin
from src.plugins.decision import HysteresisPlugin
from src.plugins.post_processing import EnergyExtensionPlugin, ViterbiTrackingPlugin
from src.analysis.evaluation import compute_pitch_contour_metrics
from src.visualization.plotter import plot_ptdb_benchmark_summary


def load_thresholds() -> Dict[str, Dict[str, float]]:
    """Load optimized thresholds from training reports or use validated defaults."""
    thresholds = {
        "acf": {"raw": 0.4408, "enhanced": 0.3953},
        "amdf": {"raw": 0.4380, "enhanced": 0.5628},
    }
    thresh_file = "outputs/reports/threshold_all_preprocessors.json"
    if os.path.exists(thresh_file):
        try:
            with open(thresh_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "acf" in data:
                    thresholds["acf"]["raw"] = float(data["acf"]["raw"]["t_opt"])
                    thresholds["acf"]["enhanced"] = float(data["acf"]["bp_clip"]["t_opt"])
                if "amdf" in data:
                    thresholds["amdf"]["raw"] = float(data["amdf"]["raw"]["t_opt"])
                    thresholds["amdf"]["enhanced"] = float(data["amdf"]["bp_clip"]["t_opt"])
        except Exception as e:
            print(f"[!] Warning: Could not read {thresh_file}: {e}")
    return thresholds


def build_enhanced_detector(method: str, threshold: float) -> PluginPitchDetector:
    """Build PluginPitchDetector configured with all 5 enhancement plugins."""
    base = PitchDetector(method=method, threshold=threshold)
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)
    clip = CenterClippingPlugin(clipping_ratio=0.40, mode="standard")

    if method == "amdf":
        t_low = threshold * 0.90
        t_high = threshold
    else:
        t_low = 0.35
        t_high = threshold

    hyst = HysteresisPlugin(t_high=t_high, t_low=t_low)
    ext = EnergyExtensionPlugin(threshold_discount=0.20)
    vit = ViterbiTrackingPlugin(f0_min=70.0, f0_max=400.0)

    plugins = [bp, clip, hyst, ext, vit]
    return PluginPitchDetector(base_detector=base, plugins=plugins)


def run_ptdb_benchmark(
    data_dir: str = "data/ptdb_tug",
    tolerance: float = 0.20,
    output_csv: str = "outputs/reports/07_ptdb_benchmark.csv",
    output_json: str = "outputs/reports/07_ptdb_benchmark.json",
    output_fig: str = "outputs/figures/07_ptdb_benchmark.png",
):

    print("=" * 80)
    print("  PTDB-TUG LARYNGOGRAPH BENCHMARK (CHUẨN QUỐC TẾ: VDE, GPE, FFE, FPE)")
    print("=" * 80)

    pairs = find_ptdb_dataset(data_dir)
    if not pairs:
        print(f"[!] Không tìm thấy dữ liệu trong '{data_dir}'.")
        print("[!] Vui lòng chạy lệnh sau để tải dữ liệu mẫu:")
        print("    uv run scripts/download_ptdb_tug.py --speakers F01,M01 --num-utterances 10")
        return

    print(f"[*] Tìm thấy {len(pairs)} câu phát âm (utterances) trong '{data_dir}'.")
    speakers = sorted(list(set(p["speaker"] for p in pairs)))
    print(f"[*] Danh sách người nói ({len(speakers)}): {', '.join(speakers)}")

    thresholds = load_thresholds()
    print(f"[*] Ngưỡng ACF  : Raw={thresholds['acf']['raw']:.4f}, Enhanced={thresholds['acf']['enhanced']:.4f}")
    print(f"[*] Ngưỡng AMDF : Raw={thresholds['amdf']['raw']:.4f}, Enhanced={thresholds['amdf']['enhanced']:.4f}")
    print(f"[*] GPE Tolerance: {tolerance * 100:.0f}%")
    print("-" * 80)

    # Khởi tạo 4 detectors
    systems = {
        "Baseline ACF": PitchDetector(method="acf", threshold=thresholds["acf"]["raw"]),
        "Enhanced ACF": build_enhanced_detector("acf", thresholds["acf"]["enhanced"]),
        "Baseline AMDF": PitchDetector(method="amdf", threshold=thresholds["amdf"]["raw"]),
        "Enhanced AMDF": build_enhanced_detector("amdf", thresholds["amdf"]["enhanced"]),
    }

    results_by_utt = []
    sample_contour_for_plot = None

    for idx, pair in enumerate(pairs, 1):
        utt = load_ptdb_utterance(pair["wav_path"], pair["f0_path"])
        sys.stdout.write(f"\r[*] Đang xử lý câu [{idx}/{len(pairs)}]: {utt.speaker} ({utt.gender}) - {utt.sentence_id} ({utt.duration:.2f}s)...")
        sys.stdout.flush()

        utt_record = {
            "speaker": utt.speaker,
            "gender": utt.gender,
            "sentence_id": utt.sentence_id,
            "duration_sec": round(utt.duration, 3),
            "total_frames": len(utt.gt_times),
            "gt_voiced_frames": int(np.sum(utt.gt_voicing)),
        }

        # Lưu thông tin contour của 1 câu đại diện để vẽ biểu đồ
        if sample_contour_for_plot is None and utt.gender == "male":
            sample_contour_for_plot = {
                "times": utt.gt_times,
                "gt_f0": utt.gt_f0,
                "title": f"Quỹ đạo Pitch {utt.speaker} ({utt.gender.capitalize()}) - Câu {utt.sentence_id.upper()} (Đối sánh Ground Truth EGG)",
            }

        for sys_name, detector in systems.items():
            res = detector.process_signal(utt.signal, utt.sample_rate)
            pred_voicing_raw = (res["labels"] == "v")
            pred_f0_raw = res["f0_contour"]

            aligned_f0, aligned_v = align_predictions_to_ground_truth(
                pred_times=res["frame_times"],
                pred_f0=pred_f0_raw,
                pred_voicing=pred_voicing_raw,
                gt_times=utt.gt_times,
            )

            metrics = compute_pitch_contour_metrics(
                pred_f0=aligned_f0,
                pred_voicing=aligned_v,
                gt_f0=utt.gt_f0,
                gt_voicing=utt.gt_voicing,
                tolerance=tolerance,
            )

            utt_record[f"{sys_name}_vde"] = metrics["vde"]
            utt_record[f"{sys_name}_gpe"] = metrics["gpe"]
            utt_record[f"{sys_name}_ffe"] = metrics["ffe"]
            utt_record[f"{sys_name}_fpe"] = metrics["fpe_mae"]

            # Lưu contour để vẽ
            if sample_contour_for_plot is not None and sample_contour_for_plot.get("gt_f0") is utt.gt_f0:
                if sys_name == "Baseline ACF":
                    sample_contour_for_plot["base_f0"] = aligned_f0
                elif sys_name == "Enhanced ACF":
                    sample_contour_for_plot["enh_f0"] = aligned_f0

        results_by_utt.append(utt_record)

    print("\n" + "=" * 80)
    print("  KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP TRÊN TOÀN BỘ TẬP DỮ LIỆU PTDB-TUG")
    print("=" * 80)

    summary = {}
    for sys_name in systems.keys():
        vde_list = [r[f"{sys_name}_vde"] for r in results_by_utt]
        gpe_list = [r[f"{sys_name}_gpe"] for r in results_by_utt]
        ffe_list = [r[f"{sys_name}_ffe"] for r in results_by_utt]
        fpe_list = [r[f"{sys_name}_fpe"] for r in results_by_utt]

        # Phân rã theo giới tính
        vde_f = [r[f"{sys_name}_vde"] for r in results_by_utt if r["gender"] == "female"]
        gpe_f = [r[f"{sys_name}_gpe"] for r in results_by_utt if r["gender"] == "female"]
        ffe_f = [r[f"{sys_name}_ffe"] for r in results_by_utt if r["gender"] == "female"]

        vde_m = [r[f"{sys_name}_vde"] for r in results_by_utt if r["gender"] == "male"]
        gpe_m = [r[f"{sys_name}_gpe"] for r in results_by_utt if r["gender"] == "male"]
        ffe_m = [r[f"{sys_name}_ffe"] for r in results_by_utt if r["gender"] == "male"]

        summary[sys_name] = {
            "vde": round(float(np.mean(vde_list)), 2),
            "vde_std": round(float(np.std(vde_list)), 2),
            "gpe": round(float(np.mean(gpe_list)), 2),
            "gpe_std": round(float(np.std(gpe_list)), 2),
            "ffe": round(float(np.mean(ffe_list)), 2),
            "ffe_std": round(float(np.std(ffe_list)), 2),
            "fpe_mae": round(float(np.mean(fpe_list)), 2),
            "fpe_std": round(float(np.std(fpe_list)), 2),
            "female": {
                "vde": round(float(np.mean(vde_f)), 2),
                "gpe": round(float(np.mean(gpe_f)), 2),
                "ffe": round(float(np.mean(ffe_f)), 2),
            },
            "male": {
                "vde": round(float(np.mean(vde_m)), 2),
                "gpe": round(float(np.mean(gpe_m)), 2),
                "ffe": round(float(np.mean(ffe_m)), 2),
            },
        }

    # In kết quả dạng liệt kê rõ ràng
    for s_name, s_data in summary.items():
        print(f"\n--- {s_name.upper()} ---")
        print(f"  * Voicing Decision Error (VDE) : {s_data['vde']:.2f}% ± {s_data['vde_std']:.2f}% (Nữ: {s_data['female']['vde']:.2f}%, Nam: {s_data['male']['vde']:.2f}%)")
        print(f"  * Gross Pitch Error (GPE)      : {s_data['gpe']:.2f}% ± {s_data['gpe_std']:.2f}% (Nữ: {s_data['female']['gpe']:.2f}%, Nam: {s_data['male']['gpe']:.2f}%)")
        print(f"  * F0 Frame Error (FFE)         : {s_data['ffe']:.2f}% ± {s_data['ffe_std']:.2f}% (Nữ: {s_data['female']['ffe']:.2f}%, Nam: {s_data['male']['ffe']:.2f}%)")
        print(f"  * Fine Pitch Error (FPE MAE)   : {s_data['fpe_mae']:.2f} Hz ± {s_data['fpe_std']:.2f} Hz")

    # Tính độ cải thiện tương đối (%)
    print("\n" + "=" * 80)
    print("  MỨC ĐỘ CẢI THIỆN NHỜ CÁC PLUGINS (ENHANCED VS BASELINE)")
    print("=" * 80)
    for meth in ["ACF", "AMDF"]:
        base_s = summary[f"Baseline {meth}"]
        enh_s = summary[f"Enhanced {meth}"]

        vde_imp = (base_s["vde"] - enh_s["vde"]) / base_s["vde"] * 100.0 if base_s["vde"] > 0 else 0
        gpe_imp = (base_s["gpe"] - enh_s["gpe"]) / base_s["gpe"] * 100.0 if base_s["gpe"] > 0 else 0
        ffe_imp = (base_s["ffe"] - enh_s["ffe"]) / base_s["ffe"] * 100.0 if base_s["ffe"] > 0 else 0

        print(f"\n* Thuật toán {meth}:")
        print(f"  - Giảm sai số VDE: từ {base_s['vde']:.2f}% -> {enh_s['vde']:.2f}% (Cải thiện {vde_imp:+.1f}%)")
        print(f"  - Giảm sai số GPE: từ {base_s['gpe']:.2f}% -> {enh_s['gpe']:.2f}% (Cải thiện {gpe_imp:+.1f}%)")
        print(f"  - Giảm sai số FFE: từ {base_s['ffe']:.2f}% -> {enh_s['ffe']:.2f}% (Cải thiện {ffe_imp:+.1f}%)")

    # Xuất file CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    import csv
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_by_utt[0].keys()))
        writer.writeheader()
        writer.writerows(results_by_utt)
    print(f"\n[+] Đã lưu báo cáo chi tiết từng câu: {os.path.abspath(output_csv)}")

    # Xuất file JSON tổng hợp
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[+] Đã lưu báo cáo tổng hợp: {os.path.abspath(output_json)}")

    # Vẽ biểu đồ đối sánh
    if output_fig:
        os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
        plot_ptdb_benchmark_summary(
            summary_metrics=summary,
            sample_contour=sample_contour_for_plot,
            save_path=output_fig,
        )
        print(f"[+] Đã lưu biểu đồ đối sánh Benchmark: {os.path.abspath(output_fig)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PTDB-TUG pitch detection benchmark (VDE, GPE, FFE).")
    parser.add_argument("--data-dir", type=str, default="data/ptdb_tug", help="Thư mục chứa PTDB dataset.")
    parser.add_argument("--tolerance", type=float, default=0.20, help="Ngưỡng sai số GPE (default 0.20 = 20%).")
    parser.add_argument("--output-csv", type=str, default="outputs/reports/07_ptdb_benchmark.csv", help="Đường dẫn file CSV.")
    parser.add_argument("--output-json", type=str, default="outputs/reports/07_ptdb_benchmark.json", help="Đường dẫn file JSON.")
    parser.add_argument("--output-fig", type=str, default="outputs/figures/07_ptdb_benchmark.png", help="Đường dẫn ảnh biểu đồ.")
    args = parser.parse_args()

    run_ptdb_benchmark(
        data_dir=args.data_dir,
        tolerance=args.tolerance,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_fig=args.output_fig,
    )


if __name__ == "__main__":
    main()
