"""Script 16: Benchmark 4 giải pháp học không giám sát nâng cao vượt giới hạn Histogram:
1. 1D GMM-EM với ràng buộc tiên nghiệm 50/50 (Liên tục, không chia bin)
2. Đa đặc trưng không gian 2D (Periodicity + Zero Crossing Rate ZCR)
3. HMM hai trạng thái: học Gaussian emission, A/pi cố định, reset qua silence
4. Ngưỡng động thích nghi theo ngữ cảnh cửa sổ trượt (Context-Adaptive Dynamic Thresholding)

Đánh giá toàn diện trên cả 3 thuật toán Bài Tập 2: ACF, AMDF, Cepstrum.
Xuất bảng chuẩn 8 cột, báo cáo JSON/CSV và 4 hình vẽ trực quan hóa 300 DPI.
"""
import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.core.audio import load_wav
from src.pipeline.features import extract_raw_features
from src.core.processing import smooth_voiced_segments
from src.analysis.unsupervised_threshold import compute_median_threshold, compute_otsu_threshold
from src.analysis.advanced_unsupervised import (
    compute_gmm_em_1d,
    cluster_multifeature_2d,
    train_and_decode_2state_hmm,
    compute_sliding_window_adaptive_threshold,
)
from src.analysis.evaluation import (
    evaluate_against_ground_truth, round_metric, mean_metric, format_metric, csv_value,
)


def extract_features_for_file(
    wav_path: str,
    method: str,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    ste_silence_ratio: float = 0.008,
) -> Dict[str, Any]:
    """Extract full frame features: raw signal, frames, STE, periodicity metrics, candidate f0, and ZCR."""
    sr, signal, duration = load_wav(wav_path)
    return extract_raw_features(signal, sr, method, frame_duration_ms, hop_duration_ms,
                                f0_min, f0_max, ste_silence_ratio,
                                wav_path=wav_path, duration=duration)


def evaluate_custom_prediction(
    file_data: Dict[str, Any],
    pred_labels: np.ndarray,
    candidate_f0: np.ndarray,
    median_size: int = 3,
) -> Dict[str, Any]:
    """Build final pitch contour from labels and candidate f0, apply post-filter, and evaluate against ground truth."""
    num_frames = len(pred_labels)
    f0_raw = np.zeros(num_frames, dtype=np.float64)

    for i in range(num_frames):
        if pred_labels[i] == "v":
            f0_raw[i] = candidate_f0[i]
        else:
            f0_raw[i] = 0.0

    f0_contour = smooth_voiced_segments(f0_raw, pred_labels, median_size)

    voiced_f0 = f0_contour[f0_contour > 0]
    f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    num_voiced = int(len(voiced_f0))

    pred_res = {
        "frame_times": file_data["frame_times"],
        "labels": pred_labels,
        "f0_contour": f0_contour,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "num_voiced": num_voiced,
    }

    lab_path = file_data["wav_path"].rsplit(".", 1)[0] + ".lab"
    return evaluate_against_ground_truth(pred_res, lab_path)


def aggregate_pipeline_metrics(eval_list: List[Dict]) -> Dict[str, float]:
    """Average 8-column metrics across test files."""
    return {
        "diff_f0": mean_metric([e["abs_error_mean"] for e in eval_list], 2),
        "mape_f0": mean_metric([e["mape_f0"] for e in eval_list], 2),
        "diff_std": mean_metric([e["abs_error_std"] for e in eval_list], 2),
        "mape_std": mean_metric([e["mape_std"] for e in eval_list], 2),
        "diff_num": mean_metric([e["abs_error_num"] for e in eval_list], 1),
        "mape_num": mean_metric([e["mape_num"] for e in eval_list], 2),
        "final_score": mean_metric([e["final_score"] for e in eval_list], 2),
    }


def run_advanced_benchmark(output_dir="outputs"):
    train_dir = "TinHieuHuanLuyen"
    test_dir = "TinHieuKiemThu"

    train_wavs = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    test_wavs = sorted(glob.glob(os.path.join(test_dir, "*.wav")))

    methods = ["acf", "amdf", "cepstrum"]
    summary_all = {}
    gmm_models = {}
    all_train_features = {}

    print("=" * 135)
    print("      CHƯƠNG TRÌNH THỰC NGHIỆM 4 GIẢI PHÁP HỌC KHÔNG GIÁM SÁT NÂNG CAO VƯỢT GIỚI HẠN HISTOGRAM")
    print("      (1. GMM-EM 1D 50/50 | 2. Đa đặc trưng 2D ZCR | 3. HMM A/pi cố định | 4. Ngưỡng động median/STE)")
    print("=" * 135)

    for method in methods:
        print(f"\n{'#' * 45} THUẬT TOÁN: {method.upper()} {'#' * 45}")

        # 1. Trích xuất đặc trưng tập Huấn luyện
        train_p_all = []
        train_z_all = []
        for tw in train_wavs:
            fdata = extract_features_for_file(tw, method=method)
            train_p_all.extend(fdata["periodicity_vals"])
            train_z_all.extend(fdata["zcr_vals"])

        train_p_arr = np.array(train_p_all, dtype=np.float64)
        train_z_arr = np.array(train_z_all, dtype=np.float64)
        all_train_features[method] = {"p": train_p_arr, "z": train_z_arr}

        # Tính các ngưỡng toàn cục từ tập Huấn luyện
        t_median = compute_median_threshold(train_p_arr)
        t_otsu = compute_otsu_threshold(train_p_arr)
        gmm_fit = compute_gmm_em_1d(train_p_arr, prior=0.5)
        t_gmm = gmm_fit["threshold"]
        gmm_models[method] = gmm_fit

        t_sup = config.baseline_threshold(method)

        print(f"1. Khảo sát Ngưỡng Toàn Cục:")
        print(f"   * GMM-EM 1D (50/50 Continuous) : T = {format_metric(t_gmm, '.4f')} (mu0={format_metric(gmm_fit['mu0'], '.3f')}, mu1={format_metric(gmm_fit['mu1'], '.3f')})")
        print(f"   * Median 50% (Phân vị thứ 50)   : T = {format_metric(t_median, '.4f')}")
        print(f"   * Otsu (Cực đại phương sai)    : T = {format_metric(t_otsu, '.4f')}")
        print(f"   * Chuẩn đối sánh Có Giám sát   : T = {format_metric(t_sup, '.4f')}")

        # 2. Chuẩn bị đặc trưng cho 4 file tập Kiểm thử
        test_file_data = [extract_features_for_file(tw, method=method) for tw in test_wavs]

        # 3. Thực thi và đánh giá 7 chiến lược trên tập Kiểm thử:
        strategies_eval = {}

        # --- A. Supervised Baseline ---
        eval_sup = []
        for fd in test_file_data:
            n_frames = len(fd["periodicity_vals"])
            labels = np.full(n_frames, "sil", dtype=object)
            for i in range(n_frames):
                if not fd["is_active"][i]:
                    labels[i] = "sil"
                elif method == "amdf":
                    labels[i] = "v" if fd["periodicity_vals"][i] <= t_sup else "uv"
                else:
                    labels[i] = "v" if fd["periodicity_vals"][i] >= t_sup else "uv"
            eval_sup.append(evaluate_custom_prediction(fd, labels, fd["candidate_f0"]))
        strategies_eval["Supervised (Bayes Lab)"] = aggregate_pipeline_metrics(eval_sup)

        # --- B. Median 50% (Histogram Champion) ---
        eval_med = []
        for fd in test_file_data:
            n_frames = len(fd["periodicity_vals"])
            labels = np.full(n_frames, "sil", dtype=object)
            for i in range(n_frames):
                if not fd["is_active"][i]:
                    labels[i] = "sil"
                elif method == "amdf":
                    labels[i] = "v" if fd["periodicity_vals"][i] <= t_median else "uv"
                else:
                    labels[i] = "v" if fd["periodicity_vals"][i] >= t_median else "uv"
            eval_med.append(evaluate_custom_prediction(fd, labels, fd["candidate_f0"]))
        strategies_eval["Median 50% (Histogram)"] = aggregate_pipeline_metrics(eval_med)

        # --- C. Otsu (Histogram) ---
        eval_otsu = []
        for fd in test_file_data:
            n_frames = len(fd["periodicity_vals"])
            labels = np.full(n_frames, "sil", dtype=object)
            for i in range(n_frames):
                if not fd["is_active"][i]:
                    labels[i] = "sil"
                elif method == "amdf":
                    labels[i] = "v" if fd["periodicity_vals"][i] <= t_otsu else "uv"
                else:
                    labels[i] = "v" if fd["periodicity_vals"][i] >= t_otsu else "uv"
            eval_otsu.append(evaluate_custom_prediction(fd, labels, fd["candidate_f0"]))
        strategies_eval["Otsu (Histogram)"] = aggregate_pipeline_metrics(eval_otsu)

        # --- 1. GMM-EM 1D (50/50 Continuous) ---
        eval_gmm = []
        for fd in test_file_data:
            n_frames = len(fd["periodicity_vals"])
            labels = np.full(n_frames, "sil", dtype=object)
            for i in range(n_frames):
                if not fd["is_active"][i]:
                    labels[i] = "sil"
                elif method == "amdf":
                    labels[i] = "v" if fd["periodicity_vals"][i] <= t_gmm else "uv"
                else:
                    labels[i] = "v" if fd["periodicity_vals"][i] >= t_gmm else "uv"
            eval_gmm.append(evaluate_custom_prediction(fd, labels, fd["candidate_f0"]))
        strategies_eval["1. GMM-EM 1D (50/50 Liên tục)"] = aggregate_pipeline_metrics(eval_gmm)

        # --- 2. Multi-Feature 2D (Periodicity + ZCR) ---
        eval_mf2d = []
        for fd in test_file_data:
            n_frames = len(fd["periodicity_vals"])
            active_idx = np.where(fd["is_active"])[0]
            labels = np.full(n_frames, "sil", dtype=object)
            if len(active_idx):
                sub_p = fd["periodicity_vals"][active_idx]
                sub_z = fd["zcr_vals"][active_idx]
                sub_labels, _ = cluster_multifeature_2d(sub_p, sub_z, method=method)
                for idx_sub, orig_idx in enumerate(active_idx):
                    labels[orig_idx] = sub_labels[idx_sub]
            eval_mf2d.append(evaluate_custom_prediction(fd, labels, fd["candidate_f0"]))
        strategies_eval["2. Đa đặc trưng 2D (Periodicity+ZCR)"] = aggregate_pipeline_metrics(eval_mf2d)

        # --- 3. HMM: shared emissions per file, fixed A/pi, independent active segments ---
        eval_hmm = []
        for fd in test_file_data:
            labels, _ = train_and_decode_2state_hmm(
                fd["periodicity_vals"], method=method, prior=0.5,
                active_mask=fd["is_active"],
            )
            eval_hmm.append(evaluate_custom_prediction(fd, labels, fd["candidate_f0"]))
        strategies_eval["3. 2-State HMM (Viterbi Decoding)"] = aggregate_pipeline_metrics(eval_hmm)

        # --- 4. Context-Adaptive Dynamic Thresholding ---
        eval_dyn = []
        for fd in test_file_data:
            n_frames = len(fd["periodicity_vals"])
            t_dynamic = compute_sliding_window_adaptive_threshold(
                fd["periodicity_vals"], fd["ste"], global_threshold=t_gmm, window_frames=50, alpha=0.30
            )
            labels = np.full(n_frames, "sil", dtype=object)
            for i in range(n_frames):
                if not fd["is_active"][i]:
                    labels[i] = "sil"
                elif method == "amdf":
                    labels[i] = "v" if fd["periodicity_vals"][i] <= t_dynamic[i] else "uv"
                else:
                    labels[i] = "v" if fd["periodicity_vals"][i] >= t_dynamic[i] else "uv"
            eval_dyn.append(evaluate_custom_prediction(fd, labels, fd["candidate_f0"]))
        strategies_eval["4. Ngưỡng Động Thích Nghi Ngữ Cảnh"] = aggregate_pipeline_metrics(eval_dyn)

        # 4. In bảng đối sánh 8 cột
        print("\n2. Bảng Đối Sánh 8 Cột Toàn Chu Trình Pitch Trên TinHieuKiemThu:")
        print(f"   {'Chiến lược phân ngưỡng':<36} | {'ΔF0 (Hz)':<9} | {'MAPE F0 (%)':<12} | {'ΔStd (Hz)':<10} | {'MAPE Std (%)':<13} | {'ΔNum':<7} | {'MAPE Num (%)':<13} | {'FINAL SCORE (%)':<15}")
        print("   " + "-" * 135)
        for s_name, res in strategies_eval.items():
            print(
                f"   {s_name:<36} | "
                f"{format_metric(res['diff_f0'], '<9.2f')} | "
                f"{format_metric(res['mape_f0'], '<12.2f')} | "
                f"{format_metric(res['diff_std'], '<10.2f')} | "
                f"{format_metric(res['mape_std'], '<13.2f')} | "
                f"{format_metric(res['diff_num'], '<7.1f')} | "
                f"{format_metric(res['mape_num'], '<13.2f')} | "
                f"{format_metric(res['final_score'], '<15.2f')}"
            )

        summary_all[method] = {
            "thresholds": {
                "t_median": round_metric(t_median, 4),
                "t_otsu": round_metric(t_otsu, 4),
                "t_gmm": round_metric(t_gmm, 4),
                "t_supervised": round_metric(t_sup, 4),
                "supervised_source": config.THRESHOLD_SOURCE,
            },
            "gmm_model": dict(gmm_fit),
            "protocol": {
                "global_threshold_fit": "train_all_frames_including_silence",
                "clustering_fit": "each_test_file_active_frames",
                "hmm_fit": "shared_emissions_per_test_file_active_segments",
                "hmm_A_pi": "fixed; reset pi at every active segment",
                "dynamic_window_frames": 50,
                "dynamic_window_offsets": [-25, 24],
                "dynamic_alpha": 0.30,
            },
            "pipeline_eval": strategies_eval,
        }

    # ==============================================================================
    # XUẤT 4 HÌNH VẼ ĐỒ THỊ CHẤT LƯỢNG CAO 300 DPI
    # ==============================================================================
    figures_dir = os.path.join(output_dir, "figures")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # --- Hình 1: Mật độ xác suất liên tục GMM-EM 1D và Giao điểm Bayes ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, method in enumerate(methods):
        ax = axes[idx]
        vals = all_train_features[method]["p"]
        fit = gmm_models[method]
        x_grid = np.linspace(min(vals), max(vals), 500)

        # Plot empirical histogram faintly
        ax.hist(vals, bins=60, density=True, alpha=0.35, color="#7f8c8d", edgecolor="white", label="Empirical Histogram")

        # Plot continuous Gaussian components
        pdf0 = fit["weight0"] * norm.pdf(x_grid, loc=fit["mu0"], scale=fit["std0"])
        pdf1 = fit["weight1"] * norm.pdf(x_grid, loc=fit["mu1"], scale=fit["std1"])
        total_pdf = pdf0 + pdf1

        low_label, high_label = ("V", "UV") if method == "amdf" else ("UV", "V")
        ax.plot(x_grid, pdf0, "--", color="#e74c3c", linewidth=2.0, label=f"{low_label}: N({format_metric(fit['mu0'], '.2f')}, {format_metric(fit['std0'], '.2f')}²)")
        ax.plot(x_grid, pdf1, "--", color="#2ecc71", linewidth=2.0, label=f"{high_label}: N({format_metric(fit['mu1'], '.2f')}, {format_metric(fit['std1'], '.2f')}²)")
        ax.plot(x_grid, total_pdf, "-", color="#2c3e50", linewidth=2.5, label="GMM-EM Hỗn hợp")

        ax.axvline(fit["threshold"], color="#9b59b6", linestyle="-", linewidth=2.5, label=f"GMM T* = {format_metric(fit['threshold'], '.4f')}")
        ax.set_title(f"GMM-EM 1D (50/50 Prior) - {method.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Đặc trưng chu kỳ", fontsize=10)
        ax.set_ylabel("Mật độ xác suất liên tục", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "16_advanced_gmm_em_distributions.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"\nSaved: {fig1_path}")

    # --- Hình 2: Biểu đồ tán xạ 2D (Periodicity vs ZCR) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, method in enumerate(methods):
        ax = axes[idx]
        p_arr = all_train_features[method]["p"]
        z_arr = all_train_features[method]["z"]

        # Run 2D clustering on training samples for visualization
        labels_2d, model_info = cluster_multifeature_2d(p_arr, z_arr, method=method)
        v_mask = labels_2d == "v"

        ax.scatter(p_arr[~v_mask], z_arr[~v_mask], c="#e74c3c", alpha=0.45, s=18, label="Cụm Unvoiced (UV)")
        ax.scatter(p_arr[v_mask], z_arr[v_mask], c="#2ecc71", alpha=0.45, s=18, label="Cụm Voiced (V)")

        # Plot centroids
        cv = model_info["centroid_v"]
        cuv = model_info["centroid_uv"]
        ax.scatter([cuv[0]], [cuv[1]], c="#c0392b", s=140, marker="X", edgecolor="black", label=f"Tâm UV ({format_metric(cuv[0], '.2f')}, {format_metric(cuv[1], '.2f')})")
        ax.scatter([cv[0]], [cv[1]], c="#27ae60", s=140, marker="P", edgecolor="black", label=f"Tâm V ({format_metric(cv[0], '.2f')}, {format_metric(cv[1], '.2f')})")

        ax.set_title(f"Không gian Đa đặc trưng 2D - {method.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Đặc trưng chu kỳ (Periodicity Metric)", fontsize=10)
        ax.set_ylabel("Tỉ lệ qua điểm không (ZCR)", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "16_advanced_multifeature_2d_scatter.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # --- Hình 3: Khử rung giật thời gian bằng 2-State HMM Viterbi Tracking ---
    fig, (ax_sig, ax_dec) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})
    # Select sample file: test_wavs[1] (phone_M2)
    demo_fd = extract_features_for_file(test_wavs[1], method="acf")
    time_sig = np.arange(len(demo_fd["signal"])) / demo_fd["sample_rate"]

    ax_sig.plot(time_sig, demo_fd["signal"], color="#34495e", linewidth=0.7)
    ax_sig.set_title(f"HMM: Học emission, A/pi cố định, reset qua silence - {os.path.basename(demo_fd['wav_path'])}", fontsize=12, fontweight="bold")
    ax_sig.set_ylabel("Biên độ sóng âm", fontsize=10)
    ax_sig.grid(True, linestyle=":", alpha=0.6)

    # Static decisions
    static_labels = np.where(
        demo_fd["is_active"] & (demo_fd["periodicity_vals"] >= gmm_models["acf"]["threshold"]), 1, 0
    )
    # HMM Viterbi decisions
    demo_labels, _ = train_and_decode_2state_hmm(
        demo_fd["periodicity_vals"], method="acf", prior=0.5, active_mask=demo_fd["is_active"]
    )
    hmm_labels = (demo_labels == "v").astype(int)

    ax_dec.step(demo_fd["frame_times"], static_labels + 0.05, where="mid", color="#e74c3c", linewidth=1.5, linestyle="--", label="Ngưỡng GMM tĩnh (cùng silence gate)")
    ax_dec.step(demo_fd["frame_times"], hmm_labels, where="mid", color="#27ae60", linewidth=2.2, label="HMM Viterbi (reset tại silence)")
    ax_dec.set_yticks([0, 1])
    ax_dec.set_yticklabels(["Unvoiced / Silence (0)", "Voiced (1)"], fontsize=10)
    ax_dec.set_xlabel("Thời gian (giây)", fontsize=10)
    ax_dec.set_ylabel("Trạng thái quyết định", fontsize=10)
    ax_dec.set_ylim(-0.2, 1.3)
    ax_dec.legend(loc="upper right", fontsize=9.5)
    ax_dec.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "16_advanced_hmm_temporal_tracking.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # --- Hình 4: Biểu đồ cột tổng hợp xếp hạng Final Score ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for idx, method in enumerate(methods):
        ax = axes[idx]
        eval_dict = summary_all[method]["pipeline_eval"]
        strat_names = list(eval_dict.keys())
        scores = np.asarray([eval_dict[k]["final_score"] for k in strat_names], dtype=float)

        # Shorten names for clean display
        short_names = [
            "Supervised",
            "Median 50%",
            "Otsu",
            "GMM-EM 1D",
            "Đa đặc trưng 2D",
            "2-State HMM",
            "Ngưỡng Động",
        ]

        colors = ["#2c3e50", "#d35400", "#8e44ad", "#2980b9", "#16a085", "#27ae60", "#e67e22"]
        y_pos = np.arange(len(short_names))

        bars = ax.barh(y_pos, scores, color=colors, height=0.62)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("FINAL SCORE (%) [Càng thấp càng tốt]", fontsize=10)
        ax.set_title(f"Xếp Hạng Final Score - {method.upper()}", fontsize=12, fontweight="bold")

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3 if np.isfinite(w) else 0.3, bar.get_y() + bar.get_height() / 2.0, f"{format_metric(w, '.2f')}%", ha="left", va="center", fontsize=9, fontweight="bold")

        finite_scores = scores[np.isfinite(scores)]
        ax.set_xlim(0, max(1.0, float(np.max(finite_scores))*1.25) if len(finite_scores) else 1.0)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "16_advanced_final_ranking_barchart.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # 5. Lưu báo cáo JSON và CSV
    json_path = os.path.join(reports_dir, "16_advanced_unsupervised_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Đã lưu báo cáo JSON: {json_path}")

    csv_path = os.path.join(reports_dir, "16_advanced_unsupervised_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Algorithm,Strategy,Diff_F0_Hz,MAPE_F0_Pct,Diff_Std_Hz,MAPE_Std_Pct,Diff_Num_Frames,MAPE_Num_Pct,Final_Score_Pct\n")
        for method, m_data in summary_all.items():
            for s_name, res in m_data["pipeline_eval"].items():
                f.write(
                    f"{csv_value(method.upper())},{csv_value(s_name)},"
                    f"{csv_value(res['diff_f0'])},{csv_value(res['mape_f0'])},{csv_value(res['diff_std'])},{csv_value(res['mape_std'])},"
                    f"{csv_value(res['diff_num'])},{csv_value(res['mape_num'])},{csv_value(res['final_score'])}\n"
                )
    print(f"[OK] Đã lưu báo cáo CSV: {csv_path}")
    print("=" * 135)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs", help="Root for figures/ and reports/")
    run_advanced_benchmark(parser.parse_args().output_dir)
