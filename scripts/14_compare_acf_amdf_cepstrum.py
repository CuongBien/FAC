"""Script 14: Đối sánh toàn diện 3 thuật toán ước lượng cao độ F0 cho Bài Tập 2 (BT2):
1. ACF (Hàm Tự Tương Quan - Miền thời gian L2)
2. AMDF (Hàm Hiệu Độ Lớn Trung Bình - Miền thời gian L1)
3. Cepstrum (Xử lý đồng cấu Homomorphic Deconvolution - Miền quefrency / phổ log)

Đánh giá cả 3 thuật toán trên 2 tập dữ liệu: TinHieuHuanLuyen và TinHieuKiemThu.
Tính toán chi tiết:
- MAE F0, MAPE F0mean (%)
- Delta Std, MAPE F0std (%)
- Delta Num, MAPE F0num (%)
- FINAL SCORE (%) = (MAPE_mean + MAPE_std + MAPE_num) / 3
Xuất biểu đồ 300 DPI và file báo cáo CSV/JSON.
"""
import argparse
import glob
import json
import os
import sys
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
from src.pipelines import build_baseline, build_champion
from src.analysis.evaluation import (
    evaluate_against_ground_truth, round_metric, mean_metric, format_metric, csv_value,
)


def build_bt2_detectors():
    """Factories shared with the CLI; each call constructs fresh plugin state."""
    detectors = {
        "1. ACF Baseline": lambda: build_baseline("acf"),
        "2. ACF Champion (BP+Energy+Viterbi)": lambda: build_champion("acf"),
        "3. AMDF Baseline": lambda: build_baseline("amdf"),
        "4. AMDF Champion (Energy+Viterbi)": lambda: build_champion("amdf"),
        "5. Cepstrum Baseline": lambda: build_baseline("cepstrum"),
        "6. Cepstrum Champion (BP+Clip+Energy+Viterbi)": lambda: build_champion("cepstrum"),
    }
    champions = {
        "ACF Champion (Time L2)": detectors["2. ACF Champion (BP+Energy+Viterbi)"],
        "AMDF Champion (Time L1)": detectors["4. AMDF Champion (Energy+Viterbi)"],
        "Cepstrum Champion (Quefrency)": detectors["6. Cepstrum Champion (BP+Clip+Energy+Viterbi)"],
    }
    return detectors, champions


def evaluate_suite(configs_dict, test_dir="TinHieuKiemThu"):
    """Evaluate detector factory functions across audio files in test_dir."""
    wav_paths = sorted(list(set(glob.glob(os.path.join(test_dir, "*.wav")) + glob.glob(os.path.join(test_dir, "*.WAV")))))
    files = [os.path.splitext(os.path.basename(p))[0] for p in wav_paths]
    if not files:
        files = ["phone_F2", "phone_M2", "studio_F2", "studio_M2"]
    suite_results = []

    for name, factory in configs_dict.items():
        det = factory()
        errs, std_errs = [], []
        mapes_f0, mapes_std, mapes_num, num_errs = [], [], [], []
        file_metrics = {}

        for f_id in files:
            wav_path = os.path.join(test_dir, f"{f_id}.wav")
            lab_path = os.path.join(test_dir, f"{f_id}.lab")

            res = det.process_file(wav_path)
            ev = evaluate_against_ground_truth(res, lab_path)

            file_metrics[f_id] = ev
            errs.append(ev["abs_error_mean"])
            std_errs.append(ev["abs_error_std"])
            mapes_f0.append(ev["mape_f0"])
            mapes_std.append(ev["mape_std"])
            mapes_num.append(ev.get("mape_num"))
            num_errs.append(ev.get("abs_error_num"))

        mean_mape_f0 = mean_metric(mapes_f0, digits=None)
        mean_mape_std = mean_metric(mapes_std, digits=None)
        mean_mape_num = mean_metric(mapes_num, digits=None)
        final_composite = mean_metric([mean_mape_f0, mean_mape_std, mean_mape_num], digits=None)

        domain = "Quefrency (Homomorphic)" if "Cepstrum" in name else ("Time (L1 Difference)" if "AMDF" in name else "Time (L2 Correlation)")

        rec = {
            "name": name,
            "configuration": {
                "threshold": (det.base_detector if hasattr(det, 'base_detector') else det).threshold,
                "frame_duration_ms": (det.base_detector if hasattr(det, 'base_detector') else det).frame_duration_ms,
                "hop_duration_ms": (det.base_detector if hasattr(det, 'base_detector') else det).hop_duration_ms,
                "plugins": [plugin.name for plugin in getattr(det, 'plugins', [])],
            },
            "domain": domain,
            "dataset": test_dir,
            "average_error_hz": mean_metric(errs, 2),
            "average_std_error": mean_metric(std_errs, 2),
            "average_num_error": mean_metric(num_errs, 2) if num_errs else None,
            "mape_f0_pct": round_metric(mean_mape_f0, 2),
            "mape_std_pct": round_metric(mean_mape_std, 2),
            "mape_num_pct": round_metric(mean_mape_num, 2) if mean_mape_num is not None else None,
            "final_composite_score_pct": round_metric(final_composite, 2),
            "files": {
                f: {
                    "abs_error_mean": round_metric(file_metrics[f]["abs_error_mean"], 2),
                    "mape_f0": round_metric(file_metrics[f]["mape_f0"], 2),
                    "abs_error_std": round_metric(file_metrics[f]["abs_error_std"], 2),
                    "mape_std": round_metric(file_metrics[f]["mape_std"], 2),
                    "pred_f0_num": file_metrics[f].get("pred_f0_num"),
                    "ref_f0_num": file_metrics[f].get("ref_f0_num"),
                    "abs_error_num": file_metrics[f].get("abs_error_num"),
                    "mape_num": file_metrics[f].get("mape_num"),
                    "composite_score": file_metrics[f].get("composite_score"),
                    "pred_f0_mean": round_metric(file_metrics[f]["pred_f0_mean"], 2),
                    "ref_f0_mean": round_metric(file_metrics[f]["ref_f0_mean"], 2),
                    "pred_f0_std": round_metric(file_metrics[f]["pred_f0_std"], 2),
                    "ref_f0_std": round_metric(file_metrics[f]["ref_f0_std"], 2),
                }
                for f in files
            },
        }
        suite_results.append(rec)

    return suite_results


def plot_bt2_comparison(records, output_fig_path):
    """Plot comprehensive 4-panel comparison for BT2: ACF vs AMDF vs Cepstrum."""
    os.makedirs(os.path.dirname(os.path.abspath(output_fig_path)), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    files = list(records[0]["files"].keys())
    file_labels = [f.replace("_", " ").upper() for f in files] + ["TRUNG BÌNH"]

    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728"]
    x = np.arange(len(file_labels))
    width = 0.85 / len(records)

    # --- Subplot 1: MAE across files ---
    ax1 = axes[0, 0]
    for i, r in enumerate(records):
        vals = [r["files"][f]["abs_error_mean"] for f in files] + [r["average_error_hz"]]
        offset = (i - (len(records) - 1) / 2) * width
        bars = ax1.bar(x + offset, np.asarray(vals, dtype=float), width, label=r["name"], color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, (h + 0.1 if np.isfinite(h) else 0.1), f"{format_metric(h, '.1f')}", ha="center", va="bottom", fontsize=7.5, rotation=35)

    ax1.set_title("1. Sai số tuyệt đối trung bình F0 (MAE - Hz) - Càng thấp càng tốt", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(file_labels, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Sai số F0 (Hz)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 2: FINAL SCORE / Composite MAPE (%) across files ---
    ax2 = axes[0, 1]
    for i, r in enumerate(records):
        vals = [r["files"][f].get("composite_score", 0.0) for f in files] + [r["final_composite_score_pct"]]
        offset = (i - (len(records) - 1) / 2) * width
        bars = ax2.bar(x + offset, np.asarray(vals, dtype=float), width, label=r["name"], color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, (h + 0.2 if np.isfinite(h) else 0.1), f"{format_metric(h, '.1f')}%", ha="center", va="bottom", fontsize=7.5, rotation=35)

    ax2.set_title("2. Điểm số Tổng hợp (Final Score - %) = TB(3 MAPE) - Càng thấp càng tốt", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(file_labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Final Score (%)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 3: Std Error (Hz) ---
    ax3 = axes[1, 0]
    for i, r in enumerate(records):
        vals = [r["files"][f]["abs_error_std"] for f in files] + [r["average_std_error"]]
        offset = (i - (len(records) - 1) / 2) * width
        bars = ax3.bar(x + offset, np.asarray(vals, dtype=float), width, label=r["name"], color=colors[i % len(colors)], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, (h + 0.1 if np.isfinite(h) else 0.1), f"{format_metric(h, '.1f')}", ha="center", va="bottom", fontsize=7.5, rotation=35)

    ax3.set_title("3. Sai số độ lệch chuẩn (Delta Std - Hz) - Độ ổn định contour", fontsize=11, fontweight="bold", pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(file_labels, fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Độ lệch chuẩn sai số (Hz)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 4: Summary Table inside ax4 ---
    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = []
    headers = ["Thuật toán (BT2)", "Miền xử lý", "MAPE F0", "MAPE Std", "MAPE Num", "FINAL SCORE"]
    for r in records:
        table_data.append([
            r["name"],
            r["domain"],
            f"{format_metric(r['mape_f0_pct'], '.2f')}%",
            f"{format_metric(r['mape_std_pct'], '.2f')}%",
            f"{format_metric(r['mape_num_pct'], '.2f')}%" if r['mape_num_pct'] is not None else "N/A",
            f"{format_metric(r['final_composite_score_pct'], '.2f')}%",
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.32, 0.22, 0.11, 0.11, 0.11, 0.13]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.0)

    # Style header row
    for col in range(len(headers)):
        table[(0, col)].set_facecolor("#2c3e50")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    # Style champion rows
    best_name = min((r for r in records if r["final_composite_score_pct"] is not None), key=lambda r: r["final_composite_score_pct"], default={"name": None})["name"]
    for row in range(1, len(table_data) + 1):
        bg = "#eaf2f8" if (row % 2 == 1) else "#ffffff"
        if table_data[row - 1][0] == best_name:
            bg = "#d4efdf"  # highlight best
        for col in range(len(headers)):
            table[(row, col)].set_facecolor(bg)
            if table_data[row - 1][0] == best_name and col == 0:
                table[(row, col)].set_text_props(fontweight="bold", color="#196f3d")

    ax4.set_title("4. Bảng Tổng Hợp So Sánh 3 Thuật Toán (BT2)", fontsize=11, fontweight="bold", pad=20)

    plt.suptitle("BÀI TẬP 2: ĐỐI SÁNH 3 THUẬT TOÁN ĐO CAO ĐỘ PITCH (ACF vs AMDF vs CEPSTRUM)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Đã lưu biểu đồ đối sánh BT2: {output_fig_path}")


def main():
    parser = argparse.ArgumentParser(description="BT2: Đối sánh 3 thuật toán ACF vs AMDF vs Cepstrum.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        help="Chọn tập dữ liệu: kiemthu, huanluyen, both, hoặc đường dẫn tới thư mục chứa file .wav/.lab",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "champions", "baseline"],
        help="Chế độ đối sánh: all (cả baseline và enhanced), champions (chỉ các bản quán quân), baseline (chỉ các bản gốc)",
    )
    args = parser.parse_args()

    all_detectors, champions_only = build_bt2_detectors()

    if args.mode == "champions":
        configs_to_run = champions_only
    elif args.mode == "baseline":
        configs_to_run = {
            "ACF Baseline": all_detectors["1. ACF Baseline"],
            "AMDF Baseline": all_detectors["3. AMDF Baseline"],
            "Cepstrum Baseline": all_detectors["5. Cepstrum Baseline"],
        }
    else:
        configs_to_run = all_detectors

    datasets = []
    if args.dataset == "huanluyen":
        datasets.append(("TinHieuHuanLuyen", "TẬP HUẤN LUYỆN (TinHieuHuanLuyen)"))
    elif args.dataset == "kiemthu":
        datasets.append(("TinHieuKiemThu", "TẬP KIỂM THỬ (TinHieuKiemThu)"))
    elif args.dataset == "both":
        datasets.append(("TinHieuHuanLuyen", "TẬP HUẤN LUYỆN (TinHieuHuanLuyen)"))
        datasets.append(("TinHieuKiemThu", "TẬP KIỂM THỬ (TinHieuKiemThu)"))
    elif os.path.isdir(args.dataset):
        datasets.append((args.dataset, f"TẬP DỮ LIỆU TÙY CHỌN ({args.dataset})"))
    else:
        print(f"[!] Thư mục hoặc tùy chọn không hợp lệ: {args.dataset}")
        sys.exit(1)

    for d_dir, d_title in datasets:
        print("\n" + "=" * 165)
        print(f"  {d_title} - SO SÁNH 3 THUẬT TOÁN (ACF vs AMDF vs CEPSTRUM - BÀI TẬP 2)")
        print("=" * 165)

        suite_results = evaluate_suite(configs_to_run, test_dir=d_dir)

        # Print table header
        print(f"{'Thuật toán (BT2)':<34} | {'Miền xử lý':<24} | {'MAE F0':<9} | {'MAPE F0':<9} | {'ΔStd':<8} | {'MAPE Std':<9} | {'ΔNum':<7} | {'MAPE Num':<9} | {'FINAL SCORE':<12}")
        print("-" * 165)
        for r in suite_results:
            num_err_str = f"{format_metric(r['average_num_error'], '4.1f')}" if r["average_num_error"] is not None else "  N/A"
            mape_num_str = f"{format_metric(r['mape_num_pct'], '5.2f')}%" if r["mape_num_pct"] is not None else "   N/A"
            final_score_str = f"{format_metric(r['final_composite_score_pct'], '5.2f')}%"

            print(
                f"{r['name']:<34} | "
                f"{r['domain']:<24} | "
                f"{format_metric(r['average_error_hz'], '5.2f')} Hz | "
                f"{format_metric(r['mape_f0_pct'], '5.2f')}%    | "
                f"{format_metric(r['average_std_error'], '5.2f')}Hz | "
                f"{format_metric(r['mape_std_pct'], '5.2f')}%    | "
                f"{num_err_str:<7} | "
                f"{mape_num_str:<9} | "
                f"{final_score_str:<12}"
            )

        # Print breakdown per file
        print("\n  Chi tiết từng file:")
        for r in suite_results:
            print(f"  * {r['name']}:")
            for fname, fmetrics in r["files"].items():
                num_info = ""
                if fmetrics.get("ref_f0_num") is not None:
                    num_info = f" | Num: Pred={fmetrics['pred_f0_num']} vs Ref={fmetrics['ref_f0_num']} (Δ={fmetrics['abs_error_num']}, MAPE={format_metric(fmetrics['mape_num'], '.1f')}%)"
                comp_info = ""
                if fmetrics.get("composite_score") is not None:
                    comp_info = f" | Score={format_metric(fmetrics['composite_score'], '.2f')}%"
                print(f"    - {fname:<12}: F0 Pred={format_metric(fmetrics['pred_f0_mean'], '.1f')} vs Ref={format_metric(fmetrics['ref_f0_mean'], '.1f')}Hz (Δ={format_metric(fmetrics['abs_error_mean'], '.2f')}Hz, MAPE={format_metric(fmetrics['mape_f0'], '.2f')}%) | Std: Pred={format_metric(fmetrics['pred_f0_std'], '.1f')} vs Ref={format_metric(fmetrics['ref_f0_std'], '.1f')}Hz (Δ={format_metric(fmetrics['abs_error_std'], '.2f')}Hz, MAPE={format_metric(fmetrics['mape_std'], '.2f')}%){num_info}{comp_info}")

        # Save reports
        prefix_out = "huanluyen_" if d_dir == "TinHieuHuanLuyen" else ""
        os.makedirs("outputs/reports", exist_ok=True)
        with open(f"outputs/reports/{prefix_out}compare_acf_amdf_cepstrum.json", "w", encoding="utf-8") as f:
            json.dump(suite_results, f, ensure_ascii=False, indent=2)

        with open(f"outputs/reports/{prefix_out}compare_acf_amdf_cepstrum.csv", "w", encoding="utf-8") as f:
            f.write("name,domain,dataset,average_error_hz,mape_f0_pct,average_std_error,mape_std_pct,average_num_error,mape_num_pct,final_composite_score_pct\n")
            for r in suite_results:
                f.write(f'"{csv_value(r["name"])}","{csv_value(r["domain"])}","{csv_value(d_dir)}",{csv_value(r["average_error_hz"])},{csv_value(r["mape_f0_pct"])},{csv_value(r["average_std_error"])},{csv_value(r["mape_std_pct"])},{csv_value(r.get("average_num_error",""))},{csv_value(r.get("mape_num_pct",""))},{csv_value(r["final_composite_score_pct"])}\n')

        # Generate plot for testing dataset
        if d_dir == "TinHieuKiemThu":
            plot_bt2_comparison(suite_results, "outputs/figures/14_compare_acf_amdf_cepstrum.png")

    print("\n[HOÀN TẤT] Báo cáo và biểu đồ đối sánh BT2 (ACF vs AMDF vs Cepstrum) đã được lưu trữ thành công!")


if __name__ == "__main__":
    main()
