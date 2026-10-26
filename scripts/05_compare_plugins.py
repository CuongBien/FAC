"""Script 05: Đánh giá toàn diện các tổ hợp Plugins cho từng thuật toán (ACF, AMDF, Cepstrum).

Nhiệm vụ:
1. Chạy tất cả 16 tổ hợp plugin (Bandpass, Center Clipping, Energy Extension, Viterbi).
2. Lọc và trích xuất đúng các nhóm tiêu chuẩn:
   - Baseline (Gốc)
   - Nhóm 1: Top 3 Tối ưu MAPE F0
   - Nhóm 2: Top 3 Tối ưu MAPE Std
   - Nhóm 3: Top 3 Tối ưu MAPE Num
   - Nhóm 4: Top 3 Toàn diện nhất (Best Final Composite Score)
3. Xuất bảng dữ liệu 8 cột tiêu chuẩn (hiển thị đầy đủ Δ và MAPE).
4. Lưu tệp CSV/JSON và vẽ biểu đồ 4 panel trực quan gọn gàng.
"""
import argparse
import glob
import itertools
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

from src.core.pitch_detector import PitchDetector
from src.plugins import (
    PluginPitchDetector,
    BandpassFilterPlugin,
    CenterClippingPlugin,
    HysteresisPlugin,
    EnergyExtensionPlugin,
    ViterbiTrackingPlugin,
)
from src.analysis.evaluation import evaluate_against_ground_truth


def get_threshold(method: str, combo: Tuple[str, ...]) -> float:
    """Trả về ngưỡng tối ưu Gauss/thực nghiệm cho từng thuật toán và tổ hợp plugin."""
    m = method.lower()
    has_bp = "BP" in combo
    has_clip = "Clip" in combo

    if m == "acf":
        if has_bp and has_clip:
            return 0.3953
        elif has_bp:
            return 0.4892
        elif has_clip:
            return 0.3352
        else:
            return 0.4408
    elif m == "amdf":
        if has_bp and has_clip:
            return 0.5628
        elif has_clip:
            return 0.6488
        elif has_bp:
            return 0.3734
        else:
            return 0.4380
    elif m == "cepstrum":
        if has_bp:
            return 0.0500
        else:
            return 0.0600
    return 0.40


def plot_clean_ranking(summary_records: List[Dict], save_path: str, title: str):
    """Vẽ biểu đồ đối sánh 4 panel: ΔF0, ΔStd, ΔNum, và Final Score."""
    # Lấy các cấu hình độc nhất để vẽ biểu đồ không bị trùng lặp
    seen = set()
    unique_records = []
    for r in summary_records:
        if r["config_name"] not in seen:
            seen.add(r["config_name"])
            unique_records.append(r)

    names = [r["config_name"] for r in unique_records]
    scores = [r["final_composite_score_pct"] for r in unique_records]
    maes = [r["average_error_hz"] for r in unique_records]
    stds = [r["average_std_error"] for r in unique_records]
    nums = [r["average_num_error"] for r in unique_records]

    n_items = len(names)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, max(5, n_items * 0.45)))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    y_pos = np.arange(n_items)

    # Panel 1: ΔF0 (MAE F0)
    min_mae = min(maes)
    colors1 = ["#2ca02c" if m == min_mae else "#1f77b4" for m in maes]
    bars1 = ax1.barh(y_pos, maes, color=colors1, alpha=0.85, edgecolor="gray")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("ΔF0 (Hz) [Càng nhỏ càng tốt]", fontsize=10, fontweight="bold")
    ax1.set_title("1. Sai số F0 (ΔF0 Hz)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)
    for b in bars1:
        w = b.get_width()
        ax1.annotate(f" {w:.2f} Hz", xy=(w, b.get_y() + b.get_height() / 2),
                     va="center", fontsize=8, fontweight="bold")

    # Panel 2: ΔStd (MAE Std)
    min_std = min(stds)
    colors2 = ["#2ca02c" if s == min_std else "#9467bd" for s in stds]
    bars2 = ax2.barh(y_pos, stds, color=colors2, alpha=0.85, edgecolor="gray")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("ΔStd (Hz) [Càng nhỏ càng tốt]", fontsize=10, fontweight="bold")
    ax2.set_title("2. Sai số Độ lệch chuẩn (ΔStd)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
    for b in bars2:
        w = b.get_width()
        ax2.annotate(f" {w:.2f}", xy=(w, b.get_y() + b.get_height() / 2),
                     va="center", fontsize=8, fontweight="bold")

    # Panel 3: ΔNum (Lệch số khung)
    min_num = min(nums)
    colors3 = ["#2ca02c" if n == min_num else "#ff7f0e" for n in nums]
    bars3 = ax3.barh(y_pos, nums, color=colors3, alpha=0.85, edgecolor="gray")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])
    ax3.invert_yaxis()
    ax3.set_xlabel("ΔNum (Khung) [Càng nhỏ càng tốt]", fontsize=10, fontweight="bold")
    ax3.set_title("3. Lệch số khung (ΔNum)", fontsize=11, fontweight="bold")
    ax3.grid(True, axis="x", linestyle="--", alpha=0.5)
    for b in bars3:
        w = b.get_width()
        ax3.annotate(f" {w:.1f}", xy=(w, b.get_y() + b.get_height() / 2),
                     va="center", fontsize=8, fontweight="bold")

    # Panel 4: Final Score (%)
    min_score = min(scores)
    colors4 = ["#2ca02c" if s == min_score else "#d62728" for s in scores]
    bars4 = ax4.barh(y_pos, scores, color=colors4, alpha=0.85, edgecolor="gray")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([])
    ax4.invert_yaxis()
    ax4.set_xlabel("Final Score (%) [Càng nhỏ càng tốt]", fontsize=10, fontweight="bold")
    ax4.set_title("4. Điểm tổng hợp Final Score (%)", fontsize=11, fontweight="bold")
    ax4.grid(True, axis="x", linestyle="--", alpha=0.5)
    for b in bars4:
        w = b.get_width()
        ax4.annotate(f" {w:.2f}%", xy=(w, b.get_y() + b.get_height() / 2),
                     va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Đã lưu biểu đồ: {save_path}")
    plt.close()


def run_evaluation(
    test_dir: str = "TinHieuHuanLuyen",
    method: str = "acf",
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_ranking_fig: Optional[str] = None,
):
    method = method.lower()
    wav_paths = sorted(glob.glob(os.path.join(test_dir, "*.wav")) + glob.glob(os.path.join(test_dir, "*.WAV")))
    files = [os.path.splitext(os.path.basename(p))[0] for p in wav_paths]
    if not files:
        files = ["phone_F1", "phone_M1", "studio_F1", "studio_M1"] if "huanluyen" in test_dir.lower() else ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]

    dataset_name = "TẬP HUẤN LUYỆN (TinHieuHuanLuyen)" if "huanluyen" in test_dir.lower() else f"TẬP DỮ LIỆU ({test_dir})"

    if output_json is None:
        output_json = f"outputs/reports/compare_plugins_{method}.json" if method != "acf" else "outputs/reports/compare_plugins.json"
    if output_csv is None:
        output_csv = f"outputs/reports/compare_plugins_{method}.csv" if method != "acf" else "outputs/reports/compare_plugins.csv"
    if output_ranking_fig is None:
        output_ranking_fig = f"outputs/figures/05_all_combinations_ranking_{method}.png" if method != "acf" else "outputs/figures/05_all_combinations_ranking.png"

    plugin_keys = ["BP", "Clip", "Energy", "Viterbi"]
    plugin_factories = {
        "BP": lambda: BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
        "Clip": lambda: CenterClippingPlugin(clipping_ratio=0.40),
        "Energy": lambda: EnergyExtensionPlugin(threshold_discount=0.20),
        "Viterbi": lambda: ViterbiTrackingPlugin(),
    }

    all_combos = []
    for r in range(len(plugin_keys) + 1):
        for combo in itertools.combinations(plugin_keys, r):
            all_combos.append(combo)

    print(f"\n[+] Đang chạy toàn bộ {len(all_combos)} tổ hợp cho thuật toán: {method.upper()}...")
    all_results = []

    for combo in all_combos:
        thresh = get_threshold(method, combo)
        base_det = PitchDetector(method=method, threshold=thresh)
        plugins = [plugin_factories[k]() for k in combo]

        detector = PluginPitchDetector(base_detector=base_det, plugins=plugins) if plugins else base_det

        errs, mape_f0s = [], []
        std_errs, mape_stds = [], []
        num_errs, mape_nums, scores = [], [], []

        for f_id in files:
            wav_path = os.path.join(test_dir, f"{f_id}.wav")
            lab_path = os.path.join(test_dir, f"{f_id}.lab")

            res = detector.process_file(wav_path)
            ev = evaluate_against_ground_truth(res, lab_path)

            errs.append(ev["abs_error_mean"])
            mape_f0s.append(ev["mape_f0"])
            std_errs.append(ev.get("abs_error_std", 0.0))
            mape_stds.append(ev.get("mape_std", 0.0))
            num_errs.append(ev.get("abs_error_num", 0))
            mape_nums.append(ev.get("mape_num", 0.0))
            scores.append(ev.get("final_score", ev.get("composite_score", 0.0)))

        name = "Baseline" if not combo else " + ".join(combo)
        all_results.append({
            "config_name": name,
            "combo": list(combo),
            "threshold": round(float(thresh), 4),
            "average_error_hz": round(float(np.mean(errs)), 2),
            "average_mape_f0_pct": round(float(np.mean(mape_f0s)), 2),
            "average_std_error": round(float(np.mean(std_errs)), 2),
            "average_mape_std_pct": round(float(np.mean(mape_stds)), 2),
            "average_num_error": round(float(np.mean(num_errs)), 2),
            "average_mape_num_pct": round(float(np.mean(mape_nums)), 2),
            "final_composite_score_pct": round(float(np.mean(scores)), 2),
        })

    # 1. Baseline
    baseline = next(r for r in all_results if len(r["combo"]) == 0)

    # 2. Nhóm 1: Top 3 Tối ưu MAPE F0
    top3_f0 = sorted(all_results, key=lambda x: (x["average_mape_f0_pct"], x["average_error_hz"]))[:3]

    # 3. Nhóm 2: Top 3 Tối ưu MAPE Std
    top3_std = sorted(all_results, key=lambda x: (x["average_mape_std_pct"], x["average_std_error"]))[:3]

    # 4. Nhóm 3: Top 3 Tối ưu MAPE Num
    top3_num = sorted(all_results, key=lambda x: (x["average_mape_num_pct"], x["average_num_error"]))[:3]

    # 5. Nhóm 4: Top 3 Toàn diện nhất (Best Final Composite Score)
    top3_overall = sorted(all_results, key=lambda x: x["final_composite_score_pct"])[:3]

    # Tập hợp các dòng báo cáo chuẩn
    reported_records = []
    
    # Thêm Baseline
    b_copy = dict(baseline)
    b_copy["group_label"] = "Baseline (Gốc)"
    reported_records.append(b_copy)

    # Thêm Nhóm 1
    for rank, item in enumerate(top3_f0, 1):
        c = dict(item)
        c["group_label"] = f"Nhóm 1 (Top {rank} F0)"
        reported_records.append(c)

    # Thêm Nhóm 2
    for rank, item in enumerate(top3_std, 1):
        c = dict(item)
        c["group_label"] = f"Nhóm 2 (Top {rank} Std)"
        reported_records.append(c)

    # Thêm Nhóm 3
    for rank, item in enumerate(top3_num, 1):
        c = dict(item)
        c["group_label"] = f"Nhóm 3 (Top {rank} Num)"
        reported_records.append(c)

    # Thêm Nhóm 4
    for rank, item in enumerate(top3_overall, 1):
        c = dict(item)
        c["group_label"] = f"Nhóm 4 (Top {rank} Toàn diện)"
        reported_records.append(c)

    # In bảng ra console
    print("\n" + "=" * 135)
    print(f"       BÁO CÁO PHÂN NHÓM TỐI ƯU ({dataset_name}) - THUẬT TOÁN: {method.upper()}")
    print("=" * 135)
    print(f"{'Phân nhóm':<24} | {'Cấu hình Plugin':<30} | {'ΔF0 (Hz)':<10} | {'MAPE F0':<9} | {'ΔStd (Hz)':<10} | {'MAPE Std':<9} | {'ΔNum (Khung)':<13} | {'MAPE Num':<9} | {'FINAL SCORE':<11}")
    print("-" * 135)

    last_group_prefix = ""
    for r in reported_records:
        g_prefix = r["group_label"].split()[0]
        if last_group_prefix and g_prefix != last_group_prefix:
            print("-" * 135)
        last_group_prefix = g_prefix

        print(
            f"{r['group_label']:<24} | "
            f"{r['config_name']:<30} | "
            f"{r['average_error_hz']:5.2f} Hz  | "
            f"{r['average_mape_f0_pct']:5.2f}%   | "
            f"{r['average_std_error']:5.2f} Hz  | "
            f"{r['average_mape_std_pct']:5.2f}%   | "
            f"{r['average_num_error']:5.1f} khung  | "
            f"{r['average_mape_num_pct']:5.2f}%   | "
            f"{r['final_composite_score_pct']:5.2f}%"
        )

    print("=" * 135)

    # Lưu JSON & CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(reported_records, f, indent=2, ensure_ascii=False)

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Group,Config,Threshold,MAE_F0_Hz,MAPE_F0_Pct,MAE_Std_Hz,MAPE_Std_Pct,Delta_Num_Frames,MAPE_Num_Pct,Final_Score_Pct\n")
        for r in reported_records:
            f.write(f"\"{r['group_label']}\",\"{r['config_name']}\",{r['threshold']},{r['average_error_hz']},{r['average_mape_f0_pct']},{r['average_std_error']},{r['average_mape_std_pct']},{r['average_num_error']},{r['average_mape_num_pct']},{r['final_composite_score_pct']}\n")

    plot_clean_ranking(reported_records, save_path=output_ranking_fig, title=f"Đối sánh cấu hình {method.upper()} trên {dataset_name} (Baseline & Top Groups)")
    print(f"[+] Báo cáo CSV: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate plugin configurations across top groups.")
    parser.add_argument("--test_dir", type=str, default="TinHieuHuanLuyen", help="Thư mục dữ liệu (mặc định: TinHieuHuanLuyen)")
    parser.add_argument("--method", type=str, default="acf", choices=["acf", "amdf", "cepstrum"], help="Method (acf, amdf, or cepstrum)")
    parser.add_argument("--out_json", type=str, default=None, help="JSON output")
    parser.add_argument("--out_csv", type=str, default=None, help="CSV output")
    parser.add_argument("--out_fig", type=str, default=None, help="Ranking plot")

    args = parser.parse_args()
    run_evaluation(
        test_dir=args.test_dir,
        method=args.method,
        output_json=args.out_json,
        output_csv=args.out_csv,
        output_ranking_fig=args.out_fig,
    )
