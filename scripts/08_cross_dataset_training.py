#!/usr/bin/env python3
"""Script 08: Thử nghiệm Cross-Dataset Training & Testing (Huấn luyện ngưỡng tối ưu trên PTDB-TUG).

Quy trình thực nghiệm:
1. Huấn luyện (Training):
   - Sử dụng 40 câu phát âm từ người nói F01 & M01 của PTDB-TUG (~30.000 khung hình).
   - Nhãn Ground Truth lấy trực tiếp từ máy đo thanh quản Laryngograph (EGG).
   - Trích xuất phân bố Gauss N(mean, std) và tìm ngưỡng tối ưu T_PTDB cho cả ACF và AMDF.
   - So sánh trực tiếp T_PTDB với T_VN (huấn luyện từ 4 file TinHieuHuanLuyen tiếng Việt).

2. Đánh giá kiểm thử chéo (Cross-Dataset Evaluation):
   - Tập kiểm thử 1 (PTDB Unseen Speakers): 40 câu phát âm của người nói F02 & M02.
     Đo đạc: VDE (%), GPE (%), FFE (%), FPE MAE (Hz).
   - Tập kiểm thử 2 (Vietnamese Test Set): 4 file TinHieuKiemThu (phone & studio, Nam & Nữ).
     Đo đạc: V/UV Accuracy (%), Voiced F1 (%), F0 error (Hz).

3. Đánh giá tính tổng quát hóa (Generalization):
   - Ngưỡng train trên EGG chuẩn quốc tế có giúp giảm thêm lỗi VDE hay không?
   - Ngưỡng train từ tiếng Anh EGG có áp dụng tốt cho tiếng Việt hay không?
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
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

from src.core.audio import frame_signal
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
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
from src.analysis.threshold import find_gaussian_threshold
from src.analysis.evaluation import compute_pitch_contour_metrics, evaluate_against_ground_truth


def extract_ptdb_distributions(
    train_pairs: List[Dict[str, str]],
    method: str = "acf",
    preprocessors: Optional[List] = None,
) -> Dict:
    """Extract voiced and unvoiced peak/valley distributions across PTDB training utterances."""
    values_v = []
    values_u = []

    for p in train_pairs:
        utt = load_ptdb_utterance(p["wav_path"], p["f0_path"])
        sig = utt.signal
        sr = utt.sample_rate

        # Preprocessing plugins if specified
        if preprocessors:
            for plugin in preprocessors:
                if hasattr(plugin, "pre_process_signal"):
                    sig = plugin.pre_process_signal(sig, sr)

        frames, frame_times = frame_signal(sig, sr, frame_duration_ms=25.0, hop_duration_ms=10.0)

        # Map ground-truth voicing onto frame centers
        idx = np.clip(np.searchsorted(utt.gt_times, frame_times), 0, len(utt.gt_times) - 1)
        voicing = utt.gt_voicing[idx]

        for frame, is_v in zip(frames, voicing):
            if preprocessors:
                for plugin in preprocessors:
                    if hasattr(plugin, "pre_process_frame"):
                        frame = plugin.pre_process_frame(frame, sr)

            if method == "acf":
                _, peak, _ = find_f0_acf(frame, sr, mode="normalized")
                if is_v:
                    values_v.append(peak)
                else:
                    values_u.append(peak)
            else:
                _, dip, _ = find_f0_amdf(frame, sr, mode="normalized")
                if is_v:
                    values_v.append(dip)
                else:
                    values_u.append(dip)

    arr_v = np.array(values_v, dtype=np.float64)
    arr_u = np.array(values_u, dtype=np.float64)

    mean_v, std_v = float(np.mean(arr_v)), float(np.std(arr_v))
    mean_u, std_u = float(np.mean(arr_u)), float(np.std(arr_u))
    t_opt = find_gaussian_threshold(mean_v, std_v, mean_u, std_u)

    return {
        "values_v": arr_v,
        "values_u": arr_u,
        "mean_v": round(mean_v, 4),
        "std_v": round(std_v, 4),
        "mean_u": round(mean_u, 4),
        "std_u": round(std_u, 4),
        "t_opt": round(t_opt, 4),
        "n_v": len(arr_v),
        "n_u": len(arr_u),
    }


def build_system(method: str, threshold: float, enhanced: bool = False) -> PitchDetector:
    """Build Baseline or Enhanced detector with given threshold."""
    base = PitchDetector(method=method, threshold=threshold)
    if not enhanced:
        return base

    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)
    clip = CenterClippingPlugin(clipping_ratio=0.40, mode="standard")
    if method == "amdf":
        t_low = threshold * 0.90
        t_high = threshold
    else:
        t_low = max(0.30, threshold - 0.05)
        t_high = threshold

    hyst = HysteresisPlugin(t_high=t_high, t_low=t_low)
    ext = EnergyExtensionPlugin(threshold_discount=0.20)
    vit = ViterbiTrackingPlugin(f0_min=70.0, f0_max=400.0)

    return PluginPitchDetector(base_detector=base, plugins=[bp, clip, hyst, ext, vit])


def plot_distributions_ptdb(
    dist_acf_raw: Dict,
    dist_acf_enh: Dict,
    dist_amdf_raw: Dict,
    dist_amdf_enh: Dict,
    save_path: str = "outputs/figures/08_threshold_distribution_ptdb.png",
):
    """Plot Gaussian distributions and optimal thresholds for PTDB-TUG training set."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = [
        (axes[0, 0], dist_acf_raw, "1. ACF Raw (Không Tiền Xử Lý)", "#1f77b4", "R(τ) Cực Đại"),
        (axes[0, 1], dist_acf_enh, "2. ACF Enhanced (Bandpass + Center Clip)", "#2ca02c", "R(τ) Cực Đại"),
        (axes[1, 0], dist_amdf_raw, "3. AMDF Raw (Không Tiền Xử Lý)", "#d62728", "D(τ) Cực Tiểu"),
        (axes[1, 1], dist_amdf_enh, "4. AMDF Enhanced (Bandpass + Center Clip)", "#9467bd", "D(τ) Cực Tiểu"),
    ]

    for ax, d, title, col, xlabel in configs:
        x_min = min(np.min(d["values_v"]), np.min(d["values_u"]))
        x_max = max(np.max(d["values_v"]), np.max(d["values_u"]))
        x = np.linspace(x_min, x_max, 500)

        # Gauss curves
        p_v = (1.0 / (np.sqrt(2 * np.pi) * d["std_v"])) * np.exp(-0.5 * ((x - d["mean_v"]) / d["std_v"]) ** 2)
        p_u = (1.0 / (np.sqrt(2 * np.pi) * d["std_u"])) * np.exp(-0.5 * ((x - d["mean_u"]) / d["std_u"]) ** 2)

        ax.hist(d["values_v"], bins=60, density=True, alpha=0.35, color="green", label=f"Voiced (N={d['n_v']})")
        ax.hist(d["values_u"], bins=60, density=True, alpha=0.35, color="red", label=f"Unvoiced (N={d['n_u']})")

        ax.plot(x, p_v, color="darkgreen", linewidth=2.0, label=f"N_V(μ={d['mean_v']:.3f}, σ={d['std_v']:.3f})")
        ax.plot(x, p_u, color="darkred", linewidth=2.0, label=f"N_U(μ={d['mean_u']:.3f}, σ={d['std_u']:.3f})")

        ax.axvline(d["t_opt"], color="blue", linestyle="--", linewidth=2.2, label=f"T_opt = {d['t_opt']:.4f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mật độ xác suất", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("PHÂN BỐ THỐNG KÊ GAUSS & NGƯỠNG TỐI ƯU HUẤN LUYỆN TRÊN PTDB-TUG (EGG GROUND TRUTH)", fontsize=13, fontweight="bold", y=0.98)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_cross_dataset_experiment():
    print("=" * 80)
    print("  THỬ NGHIỆM CROSS-DATASET: HUẤN LUYỆN NGƯỠNG TRÊN PTDB-TUG & ĐÁNH GIÁ CHÉO")
    print("=" * 80)

    # 1. Phân hoạch tập dữ liệu PTDB-TUG
    pairs = find_ptdb_dataset("data/ptdb_tug")
    train_pairs = [p for p in pairs if p["speaker"] in ("F01", "M01")]
    test_pairs = [p for p in pairs if p["speaker"] in ("F02", "M02")]

    print(f"[*] Tập Huấn luyện PTDB-TUG : {len(train_pairs)} câu (Người nói: F01, M01)")
    print(f"[*] Tập Kiểm thử PTDB-TUG   : {len(test_pairs)} câu (Người nói chưa từng thấy: F02, M02)")
    print("-" * 80)

    # 2. Huấn luyện trích xuất phân bố Gauss trên PTDB-TUG
    print("[*] Đang huấn luyện trích xuất phân bố trên tập PTDB (F01, M01)...")
    bp = BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)
    clip = CenterClippingPlugin(clipping_ratio=0.40, mode="standard")

    print("  -> Trích xuất ACF Raw...")
    dist_acf_raw = extract_ptdb_distributions(train_pairs, method="acf", preprocessors=None)
    print("  -> Trích xuất ACF Enhanced (Bandpass + Clip)...")
    dist_acf_enh = extract_ptdb_distributions(train_pairs, method="acf", preprocessors=[bp, clip])

    print("  -> Trích xuất AMDF Raw...")
    dist_amdf_raw = extract_ptdb_distributions(train_pairs, method="amdf", preprocessors=None)
    print("  -> Trích xuất AMDF Enhanced (Bandpass + Clip)...")
    dist_amdf_enh = extract_ptdb_distributions(train_pairs, method="amdf", preprocessors=[bp, clip])

    # Ngưỡng huấn luyện từ tập tiếng Việt (TinHieuHuanLuyen)
    thresh_vn = {
        "acf": {"raw": 0.4408, "enhanced": 0.3953},
        "amdf": {"raw": 0.4380, "enhanced": 0.5628},
    }

    thresh_ptdb = {
        "acf": {"raw": dist_acf_raw["t_opt"], "enhanced": dist_acf_enh["t_opt"]},
        "amdf": {"raw": dist_amdf_raw["t_opt"], "enhanced": dist_amdf_enh["t_opt"]},
    }

    print("\n" + "=" * 80)
    print("  BẢNG ĐỐI SÁNH NGƯỠNG TỐI ƯU: TIẾNG VIỆT (VN) VS. QUỐC TẾ (PTDB-TUG)")
    print("=" * 80)
    print(f"* ACF Raw        : T_VN = {thresh_vn['acf']['raw']:.4f}  |  T_PTDB = {thresh_ptdb['acf']['raw']:.4f}  (Độ lệch: {thresh_ptdb['acf']['raw'] - thresh_vn['acf']['raw']:+.4f})")
    print(f"* ACF Enhanced   : T_VN = {thresh_vn['acf']['enhanced']:.4f}  |  T_PTDB = {thresh_ptdb['acf']['enhanced']:.4f}  (Độ lệch: {thresh_ptdb['acf']['enhanced'] - thresh_vn['acf']['enhanced']:+.4f})")
    print(f"* AMDF Raw       : T_VN = {thresh_vn['amdf']['raw']:.4f}  |  T_PTDB = {thresh_ptdb['amdf']['raw']:.4f}  (Độ lệch: {thresh_ptdb['amdf']['raw'] - thresh_vn['amdf']['raw']:+.4f})")
    print(f"* AMDF Enhanced  : T_VN = {thresh_vn['amdf']['enhanced']:.4f}  |  T_PTDB = {thresh_ptdb['amdf']['enhanced']:.4f}  (Độ lệch: {thresh_ptdb['amdf']['enhanced'] - thresh_vn['amdf']['enhanced']:+.4f})")

    # Lưu biểu đồ phân bố PTDB
    plot_distributions_ptdb(dist_acf_raw, dist_acf_enh, dist_amdf_raw, dist_amdf_enh)
    print("\n[+] Đã lưu biểu đồ phân bố Gauss PTDB: outputs/figures/08_threshold_distribution_ptdb.png")

    # Lưu file JSON ngưỡng
    with open("outputs/reports/threshold_ptdb.json", "w", encoding="utf-8") as f:
        json.dump({
            "thresholds": thresh_ptdb,
            "distributions": {
                "acf_raw": {k: v for k, v in dist_acf_raw.items() if not isinstance(v, np.ndarray)},
                "acf_enh": {k: v for k, v in dist_acf_enh.items() if not isinstance(v, np.ndarray)},
                "amdf_raw": {k: v for k, v in dist_amdf_raw.items() if not isinstance(v, np.ndarray)},
                "amdf_enh": {k: v for k, v in dist_amdf_enh.items() if not isinstance(v, np.ndarray)},
            }
        }, f, indent=2, ensure_ascii=False)
    print("[+] Đã lưu cấu hình ngưỡng PTDB: outputs/reports/threshold_ptdb.json")

    # 3. ĐÁNH GIÁ CHÉO TẬP 1: PTDB-TUG Unseen Speakers (F02, M02 - 40 câu)
    print("\n" + "=" * 80)
    print("  KIỂM THỬ CHÉO 1: TRÊN TẬP PTDB-TUG UNSEEN SPEAKERS (F02, M02 - 40 CÂU)")
    print("=" * 80)

    ptdb_eval_results = []
    models_to_test = [
        ("ACF Baseline", "acf", False),
        ("ACF Enhanced", "acf", True),
        ("AMDF Baseline", "amdf", False),
        ("AMDF Enhanced", "amdf", True),
    ]

    for model_name, meth, enh in models_to_test:
        key = "enhanced" if enh else "raw"
        t_vn = thresh_vn[meth][key]
        t_ptdb = thresh_ptdb[meth][key]

        sys_vn = build_system(meth, t_vn, enhanced=enh)
        sys_ptdb = build_system(meth, t_ptdb, enhanced=enh)

        vde_vn_list, gpe_vn_list, ffe_vn_list = [], [], []
        vde_ptdb_list, gpe_ptdb_list, ffe_ptdb_list = [], [], []

        for p in test_pairs:
            utt = load_ptdb_utterance(p["wav_path"], p["f0_path"])

            # Model VN
            res_vn = sys_vn.process_signal(utt.signal, utt.sample_rate)
            f0_vn, v_vn = align_predictions_to_ground_truth(res_vn["frame_times"], res_vn["f0_contour"], res_vn["labels"] == "v", utt.gt_times)
            m_vn = compute_pitch_contour_metrics(f0_vn, v_vn, utt.gt_f0, utt.gt_voicing)
            vde_vn_list.append(m_vn["vde"])
            gpe_vn_list.append(m_vn["gpe"])
            ffe_vn_list.append(m_vn["ffe"])

            # Model PTDB
            res_ptdb = sys_ptdb.process_signal(utt.signal, utt.sample_rate)
            f0_ptdb, v_ptdb = align_predictions_to_ground_truth(res_ptdb["frame_times"], res_ptdb["f0_contour"], res_ptdb["labels"] == "v", utt.gt_times)
            m_ptdb = compute_pitch_contour_metrics(f0_ptdb, v_ptdb, utt.gt_f0, utt.gt_voicing)
            vde_ptdb_list.append(m_ptdb["vde"])
            gpe_ptdb_list.append(m_ptdb["gpe"])
            ffe_ptdb_list.append(m_ptdb["ffe"])

        mean_vde_vn, mean_gpe_vn, mean_ffe_vn = np.mean(vde_vn_list), np.mean(gpe_vn_list), np.mean(ffe_vn_list)
        mean_vde_ptdb, mean_gpe_ptdb, mean_ffe_ptdb = np.mean(vde_ptdb_list), np.mean(gpe_ptdb_list), np.mean(ffe_ptdb_list)

        ptdb_eval_results.append({
            "system": model_name,
            "vde_vn": round(mean_vde_vn, 2),
            "vde_ptdb": round(mean_vde_ptdb, 2),
            "vde_diff": round(mean_vde_ptdb - mean_vde_vn, 2),
            "gpe_vn": round(mean_gpe_vn, 2),
            "gpe_ptdb": round(mean_gpe_ptdb, 2),
            "ffe_vn": round(mean_ffe_vn, 2),
            "ffe_ptdb": round(mean_ffe_ptdb, 2),
            "ffe_diff": round(mean_ffe_ptdb - mean_ffe_vn, 2),
        })

        print(f"\n* {model_name}:")
        print(f"  - Model Train VN   (T={t_vn:.4f})   : VDE = {mean_vde_vn:.2f}% | GPE = {mean_gpe_vn:.2f}% | FFE = {mean_ffe_vn:.2f}%")
        print(f"  - Model Train PTDB (T={t_ptdb:.4f}) : VDE = {mean_vde_ptdb:.2f}% | GPE = {mean_gpe_ptdb:.2f}% | FFE = {mean_ffe_ptdb:.2f}%")
        print(f"  ==> Độ lệch FFE: {mean_ffe_ptdb - mean_ffe_vn:+.2f}%")

    # 4. ĐÁNH GIÁ CHÉO TẬP 2: Vietnamese Test Set (TinHieuKiemThu/ - 4 files)
    print("\n" + "=" * 80)
    print("  KIỂM THỬ CHÉO 2: TRÊN TẬP TIẾNG VIỆT TINHIEUKIEMTHU (4 FILES: PHONE & STUDIO)")
    print("=" * 80)

    vn_test_dir = "TinHieuKiemThu"
    vn_test_files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]
    vn_eval_results = []

    for model_name, meth, enh in models_to_test:
        key = "enhanced" if enh else "raw"
        t_vn = thresh_vn[meth][key]
        t_ptdb = thresh_ptdb[meth][key]

        sys_vn = build_system(meth, t_vn, enhanced=enh)
        sys_ptdb = build_system(meth, t_ptdb, enhanced=enh)

        acc_vn_list, f1_vn_list, f0_err_vn_list = [], [], []
        acc_ptdb_list, f1_ptdb_list, f0_err_ptdb_list = [], [], []

        for name in vn_test_files:
            wav_path = os.path.join(vn_test_dir, f"{name}.wav")
            lab_path = os.path.join(vn_test_dir, f"{name}.lab")
            if not os.path.exists(wav_path) or not os.path.exists(lab_path):
                continue

            from src.core.audio import load_wav
            sr, sig, _ = load_wav(wav_path)

            res_vn = sys_vn.process_signal(sig, sr)
            eval_vn = evaluate_against_ground_truth(res_vn, lab_path)
            acc_vn_list.append(eval_vn["classification_accuracy"])
            f1_vn_list.append(eval_vn["voiced_f1"])
            f0_err_vn_list.append(eval_vn["abs_error_mean"])

            res_ptdb = sys_ptdb.process_signal(sig, sr)
            eval_ptdb = evaluate_against_ground_truth(res_ptdb, lab_path)
            acc_ptdb_list.append(eval_ptdb["classification_accuracy"])
            f1_ptdb_list.append(eval_ptdb["voiced_f1"])
            f0_err_ptdb_list.append(eval_ptdb["abs_error_mean"])

        m_acc_vn, m_f1_vn, m_f0_vn = np.mean(acc_vn_list), np.mean(f1_vn_list), np.mean(f0_err_vn_list)
        m_acc_ptdb, m_f1_ptdb, m_f0_ptdb = np.mean(acc_ptdb_list), np.mean(f1_ptdb_list), np.mean(f0_err_ptdb_list)

        vn_eval_results.append({
            "system": model_name,
            "acc_vn": round(m_acc_vn, 2),
            "acc_ptdb": round(m_acc_ptdb, 2),
            "acc_diff": round(m_acc_ptdb - m_acc_vn, 2),
            "f1_vn": round(m_f1_vn, 2),
            "f1_ptdb": round(m_f1_ptdb, 2),
            "f0_err_vn": round(m_f0_vn, 2),
            "f0_err_ptdb": round(m_f0_ptdb, 2),
        })

        print(f"\n* {model_name}:")
        print(f"  - Model Train VN   (T={t_vn:.4f})   : Accuracy = {m_acc_vn:.2f}% | F1 = {m_f1_vn:.2f}% | Sai số F0 = {m_f0_vn:.2f} Hz")
        print(f"  - Model Train PTDB (T={t_ptdb:.4f}) : Accuracy = {m_acc_ptdb:.2f}% | F1 = {m_f1_ptdb:.2f}% | Sai số F0 = {m_f0_ptdb:.2f} Hz")
        print(f"  ==> Độ lệch Accuracy: {m_acc_ptdb - m_acc_vn:+.2f}%, F1: {m_f1_ptdb - m_f1_vn:+.2f}%")

    # 5. Lưu báo cáo CSV
    out_csv = "outputs/reports/08_cross_dataset_comparison.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["=== 1. KIEM THU CHÉO TRÊN PTDB-TUG UNSEEN SPEAKERS (F02, M02) ==="])
        writer.writerow(["System", "VDE_VN (%)", "VDE_PTDB (%)", "VDE_Diff (%)", "GPE_VN (%)", "GPE_PTDB (%)", "FFE_VN (%)", "FFE_PTDB (%)", "FFE_Diff (%)"])
        for r in ptdb_eval_results:
            writer.writerow([r["system"], r["vde_vn"], r["vde_ptdb"], r["vde_diff"], r["gpe_vn"], r["gpe_ptdb"], r["ffe_vn"], r["ffe_ptdb"], r["ffe_diff"]])
        writer.writerow([])
        writer.writerow(["=== 2. KIEM THU CHÉO TRÊN TIẾNG VIỆT TINHIEUKIEMTHU (4 FILES) ==="])
        writer.writerow(["System", "Acc_VN (%)", "Acc_PTDB (%)", "Acc_Diff (%)", "F1_VN (%)", "F1_PTDB (%)", "F0_Err_VN (Hz)", "F0_Err_PTDB (Hz)"])
        for r in vn_eval_results:
            writer.writerow([r["system"], r["acc_vn"], r["acc_ptdb"], r["acc_diff"], r["f1_vn"], r["f1_ptdb"], r["f0_err_vn"], r["f0_err_ptdb"]])

    print(f"\n[+] Đã lưu bảng báo cáo đối sánh chéo: {os.path.abspath(out_csv)}")
    print("=" * 80)


if __name__ == "__main__":
    run_cross_dataset_experiment()
