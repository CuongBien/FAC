"""Script 06: Khảo sát và Kiểm chứng Thực nghiệm Độ Bền Vững Kháng Nhiễu (Noise Robustness).

Đối sánh toàn diện giữa thuật toán ACF và AMDF trên dải tỷ số Tín hiệu trên Nhiễu (SNR):
SNR = [Clean (Không nhiễu), 25 dB, 20 dB, 15 dB, 10 dB, 5 dB, 0 dB, -5 dB]

Mục tiêu khoa học:
1. Kiểm chứng thực nghiệm luận điểm: AMDF nhạy hơn trên tín hiệu sạch, nhưng ACF bền bỉ và chính xác hơn ở môi trường nhiễu nặng (SNR <= 5 dB).
2. Minh họa cơ chế toán học: Tương quan ACF tự khử nhiễu trắng ngẫu nhiên, trong khi hiệu số AMDF bị sàn nhiễu nâng cao làm lấp phẳng đáy cực tiểu.
3. Đánh giá khả năng bảo vệ của các cải tiến (Bandpass Filter, Center Clipping, Viterbi Tracking) trong môi trường nhiễu.
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

from src.core.audio import load_wav, add_awgn_noise
from src.core.pitch_detector import PitchDetector
from src.plugins import (
    PluginPitchDetector,
    BandpassFilterPlugin,
    CenterClippingPlugin,
    EnergyExtensionPlugin,
    ViterbiTrackingPlugin,
)
from src.analysis.evaluation import evaluate_against_ground_truth
from src.visualization.plotter import (
    plot_noise_robustness_curves,
    plot_noise_mechanism_demo,
)


def run_noise_robustness_benchmark(
    test_dir: str = "TinHieuKiemThu",
    out_dir_fig: str = "outputs/figures",
    out_dir_rep: str = "outputs/reports",
    seed: int = 42,
):
    """Run comprehensive noise robustness stress test for ACF vs AMDF."""
    os.makedirs(out_dir_fig, exist_ok=True)
    os.makedirs(out_dir_rep, exist_ok=True)

    files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]
    snr_levels = [None, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0]

    # Load optimized Gaussian thresholds from config
    thresh_json = "configs/trained_thresholds.json"
    t_acf_raw = 0.4408
    t_amdf_raw = 0.4380
    t_acf_bp_clip = 0.3953
    t_amdf_clip = 0.6488

    if os.path.exists(thresh_json):
        try:
            with open(thresh_json, "r", encoding="utf-8") as f:
                t_all = json.load(f)
                if "acf" in t_all:
                    t_acf_raw = float(t_all["acf"]["raw"]["t_opt"])
                    t_acf_bp_clip = float(t_all["acf"]["bp_clip"]["t_opt"])
                if "amdf" in t_all:
                    t_amdf_raw = float(t_all["amdf"]["raw"]["t_opt"])
                    t_amdf_clip = float(t_all["amdf"]["clip"]["t_opt"])
        except Exception:
            pass

    # Initialize 4 representative detector configurations
    detectors = {
        "acf_base": PitchDetector(method="acf", threshold=t_acf_raw),
        "amdf_base": PitchDetector(method="amdf", threshold=t_amdf_raw),
        "acf_enh": PluginPitchDetector(
            base_detector=PitchDetector(method="acf", threshold=t_acf_bp_clip),
            plugins=[
                BandpassFilterPlugin(70.0, 900.0),
                CenterClippingPlugin(0.40, mode="standard"),
                EnergyExtensionPlugin(0.20),
                ViterbiTrackingPlugin(3.0, 2.0),
            ],
        ),
        "amdf_enh": PluginPitchDetector(
            base_detector=PitchDetector(method="amdf", threshold=t_amdf_clip),
            plugins=[
                CenterClippingPlugin(0.40, mode="standard"),
                ViterbiTrackingPlugin(3.0, 2.0),
            ],
        ),
    }

    print("=" * 125)
    print("     KHẢO SÁT ĐỘ BỀN VỮNG KHÁNG NHIỄU (NOISE ROBUSTNESS BENCHMARK): ACF VS. AMDF TRÊN NHIỀU MỨC SNR")
    print("=" * 125)
    header = (
        f"{'SNR (dB)':<10} | "
        f"{'ACF Base Err':<14} | {'AMDF Base Err':<14} | "
        f"{'ACF Enh Err':<14} | {'AMDF Enh Err':<14} | "
        f"{'ACF Base F1':<12} | {'AMDF Base F1':<12} | "
        f"{'Winner':<10}"
    )
    print(header)
    print("-" * 125)

    results_list = []

    for snr in snr_levels:
        snr_label = "Clean" if snr is None else f"{int(snr):+d} dB"
        snr_val = snr if snr is not None else float("inf")

        metrics_collector = {k: {"errs": [], "f1s": [], "accs": []} for k in detectors}

        for f_id in files:
            wav_path = os.path.join(test_dir, f"{f_id}.wav")
            lab_path = os.path.join(test_dir, f"{f_id}.lab")

            sr, clean_sig, _ = load_wav(wav_path)
            # Inject AWGN noise for this file at target SNR
            noisy_sig = add_awgn_noise(clean_sig, snr, seed=seed)

            for key, det in detectors.items():
                res = det.process_signal(noisy_sig, sr)
                ev = evaluate_against_ground_truth(res, lab_path)

                metrics_collector[key]["errs"].append(ev["abs_error_mean"])
                metrics_collector[key]["f1s"].append(ev["voiced_f1"])
                metrics_collector[key]["accs"].append(ev["classification_accuracy"])

        # Aggregate averages for this SNR
        snr_summary = {
            "snr_db": snr_val,
            "snr_label": snr_label,
        }

        for key in detectors:
            avg_err = float(np.mean(metrics_collector[key]["errs"]))
            avg_f1 = float(np.mean(metrics_collector[key]["f1s"]))
            avg_acc = float(np.mean(metrics_collector[key]["accs"]))
            snr_summary[key] = {
                "average_error_hz": round(avg_err, 2),
                "average_voiced_f1_pct": round(avg_f1, 2),
                "average_accuracy_pct": round(avg_acc, 2),
            }

        # Determine winner for this SNR
        err_acf_b = snr_summary["acf_base"]["average_error_hz"]
        err_amdf_b = snr_summary["amdf_base"]["average_error_hz"]
        winner = "AMDF" if err_amdf_b < err_acf_b else "ACF"

        row_str = (
            f"{snr_label:<10} | "
            f"{err_acf_b:6.2f} Hz       | {err_amdf_b:6.2f} Hz       | "
            f"{snr_summary['acf_enh']['average_error_hz']:6.2f} Hz       | {snr_summary['amdf_enh']['average_error_hz']:6.2f} Hz       | "
            f"{snr_summary['acf_base']['average_voiced_f1_pct']:5.2f}%       | {snr_summary['amdf_base']['average_voiced_f1_pct']:5.2f}%       | "
            f"{winner:<10}"
        )
        print(row_str)
        results_list.append(snr_summary)

    print("=" * 125)

    # 1. Plot degradation curves
    curves_fig_path = os.path.join(out_dir_fig, "06_noise_robustness_curves.png")
    plot_noise_robustness_curves(results_list, save_path=curves_fig_path)
    print(f"[*] Đã xuất biểu đồ đường cong suy giảm: {curves_fig_path}")

    # 2. Plot mechanism demo on a representative voiced frame
    rep_wav = os.path.join(test_dir, "studio_F2.wav")
    rep_sr, rep_sig, _ = load_wav(rep_wav)
    # Extract representative voiced frame around 1.5s (vowel [a], F0 ~ 200 Hz)
    frame_len = int(0.025 * rep_sr)
    start_idx = int(1.50 * rep_sr)
    clean_frame = rep_sig[start_idx : start_idx + frame_len]
    noisy_frame_0db = add_awgn_noise(clean_frame, snr_db=0.0, seed=seed)

    demo_fig_path = os.path.join(out_dir_fig, "06_noise_mechanism_frame_demo.png")
    plot_noise_mechanism_demo(
        clean_frame=clean_frame,
        noisy_frame=noisy_frame_0db,
        sample_rate=rep_sr,
        f0_ref=200.0,
        snr_db=0.0,
        save_path=demo_fig_path,
    )
    print(f"[*] Đã xuất biểu đồ cơ chế kháng nhiễu (0 dB Frame Demo): {demo_fig_path}")

    # 3. Save JSON report
    json_path = os.path.join(out_dir_rep, "noise_robustness_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"[*] Đã lưu báo cáo JSON: {json_path}")

    # 4. Save CSV report
    csv_path = os.path.join(out_dir_rep, "noise_robustness_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("SNR_Label,SNR_dB,ACF_Base_Err,AMDF_Base_Err,ACF_Enh_Err,AMDF_Enh_Err,ACF_Base_F1,AMDF_Base_F1,ACF_Enh_F1,AMDF_Enh_F1\n")
        for r in results_list:
            f.write(
                f"{r['snr_label']},{r['snr_db']},"
                f"{r['acf_base']['average_error_hz']},{r['amdf_base']['average_error_hz']},"
                f"{r['acf_enh']['average_error_hz']},{r['amdf_enh']['average_error_hz']},"
                f"{r['acf_base']['average_voiced_f1_pct']},{r['amdf_base']['average_voiced_f1_pct']},"
                f"{r['acf_enh']['average_voiced_f1_pct']},{r['amdf_enh']['average_voiced_f1_pct']}\n"
            )
    print(f"[*] Đã lưu báo cáo CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate noise robustness of ACF vs AMDF.")
    parser.add_argument("--test_dir", type=str, default="TinHieuKiemThu", help="Test audio directory")
    parser.add_argument("--out_fig", type=str, default="outputs/figures", help="Output figures directory")
    parser.add_argument("--out_rep", type=str, default="outputs/reports", help="Output reports directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for AWGN noise")

    args = parser.parse_args()
    run_noise_robustness_benchmark(
        test_dir=args.test_dir,
        out_dir_fig=args.out_fig,
        out_dir_rep=args.out_rep,
        seed=args.seed,
    )
