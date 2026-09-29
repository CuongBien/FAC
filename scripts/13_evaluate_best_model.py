"""Script đánh giá thuật toán tốt nhất (YIN Champion - Viterbi) trên cả 2 tập Train & Test.

In ra bảng chuẩn theo yêu cầu:
- Các cột: File | MAPE F0mean (%) | MAPE F0std (%) | MAPE F0num (%) | Avg MAPE (%)
- 1 dòng tổng kết MAPE trung bình cuối cùng cho mỗi tập.
"""
import os
import sys
import glob
import argparse
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
    ViterbiTrackingPlugin,
    BandpassFilterPlugin,
    CenterClippingPlugin,
    EnergyExtensionPlugin,
)
from src.analysis.evaluation import evaluate_against_ground_truth


def get_detector(model_key: str = "yin"):
    """Khởi tạo thuật toán Quán quân theo chỉ định."""
    models = {
        "yin": (
            "YIN Champion (Viterbi)",
            lambda: PluginPitchDetector(
                base_detector=PitchDetector(method="yin", threshold=0.25),
                plugins=[ViterbiTrackingPlugin()]
            )
        ),
        "yin_bp": (
            "YIN Ultra-Low MAE (Bandpass)",
            lambda: PluginPitchDetector(
                base_detector=PitchDetector(method="yin", threshold=0.25),
                plugins=[BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0)]
            )
        ),
        "amdf": (
            "AMDF Champion (Config 13: Clip + Viterbi)",
            lambda: PluginPitchDetector(
                base_detector=PitchDetector(method="amdf", threshold=0.648843),
                plugins=[CenterClippingPlugin(clipping_ratio=0.40), ViterbiTrackingPlugin()]
            )
        ),
        "acf": (
            "ACF Champion (Config 29: BP + Clip + Energy + Viterbi)",
            lambda: PluginPitchDetector(
                base_detector=PitchDetector(method="acf", threshold=0.395257),
                plugins=[
                    BandpassFilterPlugin(low_cutoff=70.0, high_cutoff=900.0),
                    CenterClippingPlugin(clipping_ratio=0.40),
                    EnergyExtensionPlugin(threshold_discount=0.20),
                    ViterbiTrackingPlugin(),
                ]
            )
        ),
    }
    name, factory = models.get(model_key.lower(), models["yin"])
    return name, factory()


def evaluate_dataset(detector, dataset_dir, title, model_name):
    """Chạy đánh giá trên một tập dữ liệu và in bảng kết quả chuẩn."""
    wav_paths = sorted(list(set(glob.glob(os.path.join(dataset_dir, "*.wav")) + glob.glob(os.path.join(dataset_dir, "*.WAV")))))
    if not wav_paths:
        print(f"[!] Không tìm thấy file .wav nào trong: {dataset_dir}")
        return None

    rows = []
    mapes_mean = []
    mapes_std = []
    mapes_num = []
    file_avg_mapes = []

    for wav_p in wav_paths:
        base_name = os.path.splitext(os.path.basename(wav_p))[0]
        lab_p = os.path.splitext(wav_p)[0] + ".lab"

        res = detector.process_file(wav_p)
        if not os.path.exists(lab_p):
            print(f"[!] Bỏ qua {base_name} vì không tìm thấy file nhãn .lab")
            continue

        ev = evaluate_against_ground_truth(res, lab_p)

        m_mean = ev["mape_f0"]
        m_std = ev["mape_std"]
        m_num = ev.get("mape_num", 0.0)

        # Avg MAPE cho từng file
        if ev.get("mape_num") is not None:
            avg_m = (m_mean + m_std + m_num) / 3.0
            mapes_num.append(m_num)
        else:
            avg_m = (m_mean + m_std) / 2.0

        mapes_mean.append(m_mean)
        mapes_std.append(m_std)
        file_avg_mapes.append(avg_m)

        rows.append({
            "file": base_name,
            "mape_mean": m_mean,
            "mape_std": m_std,
            "mape_num": m_num if ev.get("mape_num") is not None else None,
            "avg_mape": avg_m,
        })

    # In bảng định dạng chuẩn
    print("\n" + "=" * 80)
    print(f"  {title.upper()} - THUẬT TOÁN: {model_name.upper()}")
    print("=" * 80)
    print(f"{'File':<18} | {'MAPE F0mean (%)':<16} | {'MAPE F0std (%)':<15} | {'MAPE F0num (%)':<15} | {'Avg MAPE (%)':<13}")
    print("-" * 80)

    for r in rows:
        num_str = f"{r['mape_num']:6.2f}%" if r['mape_num'] is not None else "     N/A"
        print(f"{r['file']:<18} | {r['mape_mean']:13.2f}%  | {r['mape_std']:12.2f}%  | {num_str:<15} | {r['avg_mape']:10.2f}%")

    print("-" * 80)
    tb_mean = float(np.mean(mapes_mean))
    tb_std = float(np.mean(mapes_std))
    tb_num = float(np.mean(mapes_num)) if mapes_num else 0.0
    final_score = (tb_mean + tb_std + tb_num) / 3.0 if mapes_num else (tb_mean + tb_std) / 2.0

    num_tb_str = f"{tb_num:6.2f}%" if mapes_num else "     N/A"
    print(f"{'TỔNG KẾT (TRUNG BÌNH)':<18} | {tb_mean:13.2f}%  | {tb_std:12.2f}%  | {num_tb_str:<15} | {final_score:10.2f}%")
    print("=" * 80)
    print(f"  >>> ĐIỂM SỐ CUỐI CÙNG (FINAL SCORE - LỖI TB): {final_score:.2f}% (ĐỘ CHÍNH XÁC: {100.0 - final_score:.2f}%) <<<\n")

    return {
        "rows": rows,
        "tb_mean": tb_mean,
        "tb_std": tb_std,
        "tb_num": tb_num,
        "final_score": final_score
    }


def main():
    parser = argparse.ArgumentParser(description="Chạy thuật toán và in bảng kết quả MAPE.")
    parser.add_argument(
        "--model",
        type=str,
        default="yin",
        choices=["yin", "yin_bp", "amdf", "acf"],
        help="Lựa chọn thuật toán: yin (YIN Viterbi), yin_bp (YIN Bandpass), amdf (AMDF Config 13), acf (ACF Config 29)"
    )
    parser.add_argument("--test-dir", type=str, default=None, help="Đường dẫn thư mục test tùy chọn (nếu có)")
    args = parser.parse_args()

    model_name, detector = get_detector(args.model)

    if args.test_dir:
        evaluate_dataset(detector, args.test_dir, f"Tập Dữ Liệu: {args.test_dir}", model_name)
    else:
        # Chạy mặc định 2 bảng: Train và Test
        evaluate_dataset(detector, "TinHieuHuanLuyen", "Bảng 1: Tập Huấn Luyện (TinHieuHuanLuyen)", model_name)
        evaluate_dataset(detector, "TinHieuKiemThu", "Bảng 2: Tập Kiểm Thử (TinHieuKiemThu)", model_name)


if __name__ == "__main__":
    main()
