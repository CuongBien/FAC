#!/usr/bin/env python3
"""Script 09: Khảo sát độ bền vững kháng nhiễu (Noise Robustness Benchmark) trên tập dữ liệu chuẩn PTDB-TUG.

Đặc điểm nghiên cứu vượt trội:
- Khác với file âm thanh thông thường (khi có nhiễu thì tai người hay mốc thời gian *.lab cũng khó nhận biết chính xác),
  bộ dữ liệu PTDB-TUG sử dụng tín hiệu Laryngograph (EGG) đo trực tiếp rung động thanh quản bằng điện cực.
- Tín hiệu EGG hoàn toàn MIỄN NHIỄM với tiếng ồn trong không khí. Do đó, Ground Truth F0 và Voicing giữ độ chính xác 100%
  ngay cả khi âm thanh Micro bị pha trộn nhiễu cực nặng (0 dB, -5 dB).

Quy trình khảo sát:
- Pha trộn nhiễu trắng ngẫu nhiên AWGN ở các mức SNR: Clean, 20 dB, 10 dB, 5 dB, 0 dB, -5 dB.
- Đo đạc 4 chỉ số chuẩn quốc tế: VDE (%), GPE (%), FFE (%), FPE MAE (Hz) so với Ground Truth EGG.
- Đối kháng 4 hệ thống:
  1. Baseline ACF
  2. Enhanced ACF (Bandpass + CenterClip + Hysteresis + EnergyExtension + Viterbi)
  3. Baseline AMDF
  4. Enhanced AMDF
"""

import argparse
import csv
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

from src.core.audio import add_awgn_noise
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
from src.visualization.plotter import plot_ptdb_noise_robustness_curves


def build_detectors(thresholds: Dict[str, Dict[str, float]]) -> Dict[str, PitchDetector]:
    """Build the 4 benchmark systems with optimal thresholds."""
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)
    clip = CenterClippingPlugin(clipping_ratio=0.40, mode="standard")
    ext = EnergyExtensionPlugin(threshold_discount=0.20)
    vit = ViterbiTrackingPlugin(f0_min=70.0, f0_max=400.0)

    # ACF
    t_acf_raw = thresholds["acf"]["raw"]
    t_acf_enh = thresholds["acf"]["enhanced"]
    hyst_acf = HysteresisPlugin(t_high=t_acf_enh, t_low=max(0.30, t_acf_enh - 0.05))

    acf_base = PitchDetector(method="acf", threshold=t_acf_raw)
    acf_enh = PluginPitchDetector(
        base_detector=PitchDetector(method="acf", threshold=t_acf_enh),
        plugins=[bp, clip, hyst_acf, ext, vit],
    )

    # AMDF
    t_amdf_raw = thresholds["amdf"]["raw"]
    t_amdf_enh = thresholds["amdf"]["enhanced"]
    hyst_amdf = HysteresisPlugin(t_high=t_amdf_enh, t_low=t_amdf_enh * 0.90)

    amdf_base = PitchDetector(method="amdf", threshold=t_amdf_raw)
    amdf_enh = PluginPitchDetector(
        base_detector=PitchDetector(method="amdf", threshold=t_amdf_enh),
        plugins=[bp, clip, hyst_amdf, ext, vit],
    )

    return {
        "Baseline ACF": acf_base,
        "Enhanced ACF": acf_enh,
        "Baseline AMDF": amdf_base,
        "Enhanced AMDF": amdf_enh,
    }


def run_noise_benchmark(
    data_dir: str = "data/ptdb_tug",
    num_per_speaker: int = 5,
    snrs: Optional[List[Optional[float]]] = None,
    output_csv: str = "outputs/reports/09_ptdb_noise_robustness.csv",
    output_json: str = "outputs/reports/09_ptdb_noise_robustness.json",
    output_fig: str = "outputs/figures/09_ptdb_noise_robustness_curves.png",
    seed: int = 42,
):
    print("=" * 80)
    print("  ĐÁNH GIÁ ĐỘ BỀN VỮNG KHÁNG NHIỄU TRÊN PTDB-TUG (LARYNGOGRAPH EGG GROUND TRUTH)")
    print("=" * 80)

    if snrs is None:
        snrs = [None, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0]

    pairs = find_ptdb_dataset(data_dir)
    if not pairs:
        print(f"[!] Không tìm thấy dữ liệu trong '{data_dir}'. Vui lòng tải dữ liệu trước.")
        return

    # Select representative utterances across speakers
    selected_pairs = []
    speakers = sorted(list(set(p["speaker"] for p in pairs)))
    for spk in speakers:
        spk_pairs = [p for p in pairs if p["speaker"] == spk]
        if num_per_speaker > 0:
            spk_pairs = spk_pairs[:num_per_speaker]
        selected_pairs.extend(spk_pairs)

    print(f"[*] Tập kiểm thử: {len(selected_pairs)} câu phát âm từ {len(speakers)} người nói ({', '.join(speakers)})")
    print(f"[*] Các mức SNR khảo sát: {[('Clean' if s is None else f'{s:.0f} dB') for s in snrs]}")
    print(f"[*] Seed tạo nhiễu ngẫu nhiên: {seed}")
    print("-" * 80)

    # Thresholds
    thresholds = {
        "acf": {"raw": 0.4408, "enhanced": 0.3821},
        "amdf": {"raw": 0.4380, "enhanced": 0.5555},
    }
    systems = build_detectors(thresholds)

    snr_summary_list = []
    detailed_rows = []

    for snr_val in snrs:
        snr_label = "Clean" if snr_val is None else f"{snr_val:+.0f} dB"
        print(f"\n[*] Đang đánh giá mức nhiễu SNR = {snr_label}...")

        snr_metrics = {s_name: {"vde": [], "gpe": [], "ffe": [], "fpe": []} for s_name in systems}

        for idx, pair in enumerate(selected_pairs, 1):
            utt = load_ptdb_utterance(pair["wav_path"], pair["f0_path"])

            # Add AWGN noise to speech audio
            noisy_signal = add_awgn_noise(utt.signal, snr_val, seed=seed + idx)

            sys.stdout.write(f"\r  -> Câu [{idx}/{len(selected_pairs)}]: {utt.speaker} - {utt.sentence_id}...")
            sys.stdout.flush()

            for sys_name, detector in systems.items():
                res = detector.process_signal(noisy_signal, utt.sample_rate)
                f0_pred, v_pred = align_predictions_to_ground_truth(
                    res["frame_times"], res["f0_contour"], res["labels"] == "v", utt.gt_times
                )
                m = compute_pitch_contour_metrics(f0_pred, v_pred, utt.gt_f0, utt.gt_voicing, tolerance=0.20)

                snr_metrics[sys_name]["vde"].append(m["vde"])
                snr_metrics[sys_name]["gpe"].append(m["gpe"])
                snr_metrics[sys_name]["ffe"].append(m["ffe"])
                snr_metrics[sys_name]["fpe"].append(m["fpe_mae"])

                detailed_rows.append({
                    "snr": snr_label,
                    "speaker": utt.speaker,
                    "gender": utt.gender,
                    "sentence_id": utt.sentence_id,
                    "system": sys_name,
                    "vde": m["vde"],
                    "gpe": m["gpe"],
                    "ffe": m["ffe"],
                    "fpe_mae": m["fpe_mae"],
                })

        print()
        # Summary for this SNR
        snr_record = {"snr_label": snr_label, "snr_val": snr_val}
        for s_name in systems:
            mean_vde = float(np.mean(snr_metrics[s_name]["vde"]))
            mean_gpe = float(np.mean(snr_metrics[s_name]["gpe"]))
            mean_ffe = float(np.mean(snr_metrics[s_name]["ffe"]))
            mean_fpe = float(np.mean(snr_metrics[s_name]["fpe"]))

            snr_record[s_name] = {
                "vde": round(mean_vde, 2),
                "gpe": round(mean_gpe, 2),
                "ffe": round(mean_ffe, 2),
                "fpe": round(mean_fpe, 2),
            }

            print(f"     {s_name:<28}: GPE = {mean_gpe:5.2f}% | VDE = {mean_vde:5.2f}% | FFE = {mean_ffe:5.2f}% | FPE = {mean_fpe:4.2f} Hz")

        snr_summary_list.append(snr_record)

    print("\n" + "=" * 80)
    print("  TỔNG HỢP SO SÁNH KHÁNG NHIỄU GIỮA CÁC HỆ THỐNG")
    print("=" * 80)

    # Highlight cross-over point and extreme noise survival
    clean_rec = snr_summary_list[0]
    zero_db_rec = [r for r in snr_summary_list if r["snr_label"] == "+0 dB" or r["snr_label"] == "0 dB"][0]
    minus_5_rec = snr_summary_list[-1]

    print("\n1. MÔI TRƯỜNG SẠCH (CLEAN):")
    for s in systems:
        print(f"   * {s:<28}: GPE = {clean_rec[s]['gpe']:5.2f}%, FFE = {clean_rec[s]['ffe']:5.2f}%")

    print("\n2. MÔI TRƯỜNG NHIỄU CỰC NẶNG (0 dB SNR):")
    for s in systems:
        print(f"   * {s:<28}: GPE = {zero_db_rec[s]['gpe']:5.2f}%, FFE = {zero_db_rec[s]['ffe']:5.2f}%")

    print(f"\n3. MỨC ĐỘ THẤT BẠI TẠI -5 dB SNR (Nhiễu mạnh hơn tiếng nói):")
    for s in systems:
        print(f"   * {s:<28}: GPE = {minus_5_rec[s]['gpe']:5.2f}%, FFE = {minus_5_rec[s]['ffe']:5.2f}%")

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"\n[+] Đã lưu báo cáo chi tiết theo mức nhiễu: {os.path.abspath(output_csv)}")

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(snr_summary_list, f, indent=2, ensure_ascii=False)
    print(f"[+] Đã lưu dữ liệu tổng hợp đường cong SNR: {os.path.abspath(output_json)}")

    # Save Plot
    if output_fig:
        os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
        plot_ptdb_noise_robustness_curves(
            snr_results=snr_summary_list,
            save_path=output_fig,
        )
        print(f"[+] Đã lưu biểu đồ đường cong kháng nhiễu PTDB: {os.path.abspath(output_fig)}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PTDB-TUG noise robustness benchmark.")
    parser.add_argument("--data-dir", type=str, default="data/ptdb_tug", help="Thư mục chứa PTDB dataset.")
    parser.add_argument("--num-per-speaker", type=int, default=5, help="Số câu mỗi người nói (default 5, total 20).")
    parser.add_argument("--output-csv", type=str, default="outputs/reports/09_ptdb_noise_robustness.csv", help="File CSV.")
    parser.add_argument("--output-json", type=str, default="outputs/reports/09_ptdb_noise_robustness.json", help="File JSON.")
    parser.add_argument("--output-fig", type=str, default="outputs/figures/09_ptdb_noise_robustness_curves.png", help="File ảnh.")
    args = parser.parse_args()

    run_noise_benchmark(
        data_dir=args.data_dir,
        num_per_speaker=args.num_per_speaker,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_fig=args.output_fig,
    )


if __name__ == "__main__":
    main()
