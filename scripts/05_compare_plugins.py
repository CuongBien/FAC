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
import csv
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

from src.metrics.aggregation import sweep_metrics
from src import config
from src.core.pitch_detector import PitchDetector
from src.plugins import (
    PluginPitchDetector,
    BandpassFilterPlugin,
    CenterClippingPlugin,
    EnergyExtensionPlugin,
    ViterbiTrackingPlugin,
)
from src.metrics.evaluation import evaluate_against_ground_truth
from src.metrics.numeric import rank_metric
from src.metrics.formatting import format_metric, csv_value
from src.experiments.serialization import write_json
from src.experiments.manifest import source_manifest


def get_threshold(method: str, combo: Tuple[str, ...]) -> float:
    """Use the shared historical threshold for the preprocessing combination."""
    return config.threshold_for_plugins(method, combo)


def plot_clean_ranking(summary_records: List[Dict], save_path: str, title: str):
    """Plot measured metrics only; missing values cannot win or become zero bars."""
    unique = {}
    for record in summary_records:
        unique.setdefault(record["config_name"], record)
    records = list(unique.values())
    names = [record["config_name"] for record in records]
    fields = [
        ("average_error_hz", "Sai số F0 (Hz)", ".2f"),
        ("average_std_error", "Sai số độ lệch chuẩn (Hz)", ".2f"),
        ("average_num_error", "Lệch số khung voiced", ".1f"),
        ("final_composite_score_pct", "Final Score (%)", ".2f"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, max(5, len(names)*.45)))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=.98)
    for ax, (key, label, spec) in zip(axes, fields):
        values = np.asarray([record[key] for record in records], dtype=float)
        finite = values[np.isfinite(values)]
        best = float(np.min(finite)) if len(finite) else None
        colors = ["#2ca02c" if best is not None and value == best else "#1f77b4" for value in values]
        bars = ax.barh(np.arange(len(names)), values, color=colors, alpha=.85, edgecolor="gray")
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("Càng thấp càng tốt")
        ax.grid(True, axis="x", linestyle="--", alpha=.5)
        for bar, value in zip(bars, values):
            x = float(value) if np.isfinite(value) else 0.0
            ax.annotate(" " + format_metric(value, spec),
                        xy=(x, bar.get_y()+bar.get_height()/2), va="center", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, .95])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ranking: {save_path}")


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
        "BP": lambda: BandpassFilterPlugin(low_cutoff=config.BANDPASS_LOW, high_cutoff=config.BANDPASS_HIGH),
        "Clip": lambda: CenterClippingPlugin(clipping_ratio=config.CLIPPING_RATIO),
        "Energy": lambda: EnergyExtensionPlugin(threshold_discount=config.ENERGY_THRESHOLD_DISCOUNT),
        "Viterbi": lambda: ViterbiTrackingPlugin(w_freq=config.VITERBI_W_FREQ, w_octave=config.VITERBI_W_OCTAVE),
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

        per_file = {}

        for f_id in files:
            wav_path = os.path.join(test_dir, f"{f_id}.wav")
            lab_path = os.path.join(test_dir, f"{f_id}.lab")

            res = detector.process_file(wav_path)
            ev = evaluate_against_ground_truth(res, lab_path)
            per_file[f_id] = ev

        name = "Baseline" if not combo else " + ".join(combo)
        all_results.append({
            "config_name": name,
            "combo": list(combo),
            "threshold": round(float(thresh), 4),
            **sweep_metrics(per_file.values()),
            "files": per_file,
        })

    # 1. Baseline
    raw_path = os.path.splitext(output_json)[0] + '_raw.json'
    write_json(raw_path, {'method': method, 'dataset': test_dir,
                          'provenance': source_manifest(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'all_configurations': all_results})
    raw_csv = os.path.splitext(output_csv)[0] + '_per_file.csv'
    os.makedirs(os.path.dirname(os.path.abspath(raw_csv)), exist_ok=True)
    with open(raw_csv, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['config', 'threshold', 'file', 'pred_mean', 'pred_std', 'pred_num', 'mape_mean', 'mape_std', 'mape_num', 'final_score'])
        for item in all_results:
            for file, metrics in item['files'].items():
                writer.writerow([item['config_name'], item['threshold'], file] + [metrics.get(key) for key in
                                  ('pred_f0_mean', 'pred_f0_std', 'pred_f0_num', 'mape_f0', 'mape_std', 'mape_num', 'final_score')])

    baseline = next(r for r in all_results if len(r["combo"]) == 0)

    # 2. Nhóm 1: Top 3 Tối ưu MAPE F0
    top3_f0 = sorted((r for r in all_results if r["average_mape_f0_pct"] is not None), key=lambda x: (rank_metric(x["average_mape_f0_pct"]), rank_metric(x["average_error_hz"])))[:3]

    # 3. Nhóm 2: Top 3 Tối ưu MAPE Std
    top3_std = sorted((r for r in all_results if r["average_mape_std_pct"] is not None), key=lambda x: (rank_metric(x["average_mape_std_pct"]), rank_metric(x["average_std_error"])))[:3]

    # 4. Nhóm 3: Top 3 Tối ưu MAPE Num
    top3_num = sorted((r for r in all_results if r["average_mape_num_pct"] is not None), key=lambda x: (rank_metric(x["average_mape_num_pct"]), rank_metric(x["average_num_error"])))[:3]

    # 5. Nhóm 4: Top 3 Toàn diện nhất (Best Final Composite Score)
    top3_overall = sorted((r for r in all_results if r["final_composite_score_pct"] is not None), key=lambda x: rank_metric(x["final_composite_score_pct"]))[:3]

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
            f"{format_metric(r['average_error_hz'], '5.2f')} Hz  | "
            f"{format_metric(r['average_mape_f0_pct'], '5.2f')}%   | "
            f"{format_metric(r['average_std_error'], '5.2f')} Hz  | "
            f"{format_metric(r['average_mape_std_pct'], '5.2f')}%   | "
            f"{format_metric(r['average_num_error'], '5.1f')} khung  | "
            f"{format_metric(r['average_mape_num_pct'], '5.2f')}%   | "
            f"{format_metric(r['final_composite_score_pct'], '5.2f')}%"
        )

    print("=" * 135)

    # Lưu JSON & CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(reported_records, f, indent=2, ensure_ascii=False)

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Group,Config,Threshold,MAE_F0_Hz,MAPE_F0_Pct,MAE_Std_Hz,MAPE_Std_Pct,Delta_Num_Frames,MAPE_Num_Pct,Final_Score_Pct\n")
        for r in reported_records:
            f.write(f"\"{csv_value(r['group_label'])}\",\"{csv_value(r['config_name'])}\",{csv_value(r['threshold'])},{csv_value(r['average_error_hz'])},{csv_value(r['average_mape_f0_pct'])},{csv_value(r['average_std_error'])},{csv_value(r['average_mape_std_pct'])},{csv_value(r['average_num_error'])},{csv_value(r['average_mape_num_pct'])},{csv_value(r['final_composite_score_pct'])}\n")

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
