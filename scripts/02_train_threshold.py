"""Script 02: Trích xuất phân bố thống kê của đỉnh cực đại ACF trên tập Huấn luyện (TinHieuHuanLuyen).
Tính (meanV, stdV), (meanU, stdU) và xác định ngưỡng tối ưu T phân tách Voiced / Unvoiced (Gaussian).
"""
import argparse
import json
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.threshold import extract_training_distributions, find_gaussian_threshold
from src.visualization.plotter import plot_distributions_and_threshold


def run_training(
    train_dir: str = "TinHieuHuanLuyen",
    method: str = "acf",
    frame_duration_ms: float = 30.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    output_fig: str = "outputs/figures/02_threshold_distribution.png",
    output_json: str = "outputs/reports/threshold_acf.json",
):
    # 1. Trích xuất phân bố từ tập huấn luyện
    stats = extract_training_distributions(
        training_dir=train_dir,
        method=method,
        frame_duration_ms=frame_duration_ms,
        hop_duration_ms=hop_duration_ms,
        f0_min=f0_min,
        f0_max=f0_max,
    )

    mean_v, std_v = stats["mean_v"], stats["std_v"]
    mean_u, std_u = stats["mean_u"], stats["std_u"]
    n_v, n_u = stats["num_voiced_frames"], stats["num_unvoiced_frames"]

    # 2. Tìm ngưỡng tối ưu theo phân bố chuẩn Gaussian
    threshold = find_gaussian_threshold(mean_v, std_v, mean_u, std_u)

    # 3. In kết quả kỹ thuật gọn gàng
    print(f"Dataset: {train_dir} ({len(stats['files_processed'])} files: {', '.join(stats['files_processed'])})")
    print(f"Params: Method={method.upper()}, FrameLen={frame_duration_ms}ms, HopLen={hop_duration_ms}ms, Range=[{f0_min:.0f}, {f0_max:.0f}]Hz")
    print(f"Voiced   (V) : N={n_v:4d} frames, mean_V={mean_v:.4f}, std_V={std_v:.4f}")
    print(f"Unvoiced (UV): N={n_u:4d} frames, mean_U={mean_u:.4f}, std_U={std_u:.4f}")
    print(f"Optimal Threshold T = {threshold:.4f}")

    # 4. Lưu biểu đồ phân bố
    plot_distributions_and_threshold(
        values_v=stats["values_v"],
        values_u=stats["values_u"],
        mean_v=mean_v,
        std_v=std_v,
        mean_u=mean_u,
        std_u=std_u,
        threshold=threshold,
        method=method,
        save_path=output_fig,
    )
    print(f"Saved figure: {output_fig}")

    # 5. Lưu báo cáo dạng JSON
    report_data = {
        "method": method,
        "frame_duration_ms": frame_duration_ms,
        "hop_duration_ms": hop_duration_ms,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "files_processed": stats["files_processed"],
        "num_voiced_frames": n_v,
        "num_unvoiced_frames": n_u,
        "mean_v": round(mean_v, 4),
        "std_v": round(std_v, 4),
        "mean_u": round(mean_u, 4),
        "std_u": round(std_u, 4),
        "threshold_T": round(threshold, 4),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {output_json}")

    return threshold, report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Voiced/Unvoiced threshold using training set.")
    parser.add_argument("--train_dir", type=str, default="TinHieuHuanLuyen", help="Thư mục tập huấn luyện")
    parser.add_argument("--method", type=str, default="acf", choices=["acf"], help="Thuật toán")
    parser.add_argument("--frame_len", type=float, default=30.0, help="Độ dài khung (ms)")
    parser.add_argument("--hop_len", type=float, default=10.0, help="Độ dịch khung (ms)")
    parser.add_argument("--f0_min", type=float, default=70.0, help="F0 tối thiểu (Hz)")
    parser.add_argument("--f0_max", type=float, default=400.0, help="F0 tối đa (Hz)")
    parser.add_argument("--out_fig", type=str, default="outputs/figures/02_threshold_distribution.png", help="Đường dẫn lưu ảnh")
    parser.add_argument("--out_json", type=str, default="outputs/reports/threshold_acf.json", help="Đường dẫn lưu JSON")

    args = parser.parse_args()
    run_training(
        train_dir=args.train_dir,
        method=args.method,
        frame_duration_ms=args.frame_len,
        hop_duration_ms=args.hop_len,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        output_fig=args.out_fig,
        output_json=args.out_json,
    )
