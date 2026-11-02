# FAC — Ước lượng F0 và phân loại Voiced/Unvoiced

Dự án môn Xử lý tín hiệu tiếng nói, hiện tập trung vào **ACF, AMDF và Cepstrum**, pipeline plugin ba chặng và các phương pháp tìm ngưỡng không giám sát.

## Bắt đầu

Chạy các lệnh từ thư mục gốc dự án:

```powershell
uv sync
uv run python predict.py -i TinHieuKiemThu -m champions -o outputs/predictions/current_test
```

Nếu đã có môi trường `.venv`, có thể chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe predict.py -i TinHieuKiemThu -m champions -o outputs/predictions/current_test
```

Xem [hướng dẫn kiểm thử](docs/TESTING.md) để chọn thuật toán, xử lý dữ liệu mới và chạy benchmark.

[Kế hoạch refactor](docs/REFACTOR_PLAN.md) chia việc chuẩn hóa thí nghiệm và quản lý báo cáo thành sáu đợt. **Đã hoàn tất đợt 1–4**: có spec/store/registry, pipeline contract, runner với journal/resume và API metric v1/đối sánh có metadata. 189 test đạt; điểm số cũ được giữ nguyên. Xem [quản lý thí nghiệm](docs/EXPERIMENTS.md), [pipeline](docs/PIPELINE.md), [runner](docs/RUNNER.md) và [metric/đối sánh](docs/METRICS.md).

## Dữ liệu và chỉ số

- Train: `TinHieuHuanLuyen/` — phone_F1, phone_M1, studio_F1, studio_M1.
- Test: `TinHieuKiemThu/` — phone_F2, phone_M2, studio_F2, studio_M2.
- Phone dùng 16 kHz; studio dùng 44,1 kHz. Code sử dụng sample rate thực tế của WAV.
- Baseline dùng khung 25 ms, Champion dùng 30 ms; hop 10 ms, tìm F0 trong 70–400 Hz.
- LAB hiện hành chỉ chứa `F0mean`, `F0std`, `F0num`; không có timestamps hoặc contour F0 tham chiếu từng khung.

Final Score là trung bình sai số phần trăm của **mean F0, std F0 và số khung voiced**, sau đó lấy trung bình giữa các file. Đây là chỉ số sai lệch thống kê toàn file; không phải Accuracy/F1 phân loại từng khung. Yêu cầu gốc được giữ tại [HuongDan.txt](HuongDan.txt).

## Kết quả và trạng thái kiểm chứng

Đã hoàn tất bốn đợt sửa và [rà soát main ngày 01/11/2026](docs/MAIN_REVIEW_FIXES.md): sửa Otsu, xử lý LAB thiếu thống kê và kiểm chứng artifact. Đợt 4 khảo sát 222 cấu hình chỉ trên train, đóng băng trước test; baseline vẫn là tham số lịch sử. Các Champion hiện dùng khung 30 ms, hop 10 ms.

Đã hoàn tất [benchmark tám phương pháp không giám sát với Champion](docs/CHAMPION_UNSUPERVISED_RESULTS.md): 288 kết quả từng file, mô hình đóng băng trước test. Sau [ba sửa lỗi biên cuối](docs/FINAL_EDGE_CASE_FIXES.md), **120/120 test đạt** và cả 288 kết quả vẫn khớp chính xác. Cepstrum ngưỡng động đạt 16,480% test; 2D fit từng input đạt 10,375% nhưng thuộc nhóm thích nghi riêng. Các preset bên dưới được giữ nguyên.

<!-- STAGE4_README_START -->
| Champion | Ngưỡng | Train Final Score (%) | Test Final Score (%) |
|---|---:|---:|---:|
| ACF | 0.45 | 2.215 | 2.385 |
| AMDF | 0.5 | 2.078 | 2.755 |
| CEPSTRUM | 0.1 | 11.390 | 22.942 |
<!-- STAGE4_README_END -->

**ACF/AMDF cải thiện trên test; Cepstrum tăng sai số từ 21.203% lên 22.942% dù giảm trên train.** Không chọn lại tham số dựa vào test. Điểm thấp hơn tốt hơn; không phải Accuracy/F1 từng khung.

Đã truy ra mốc Cepstrum lịch sử 4,43%/7,62% là pipeline lai: quyết định bằng ACF, hậu xử lý/Viterbi bằng Cepstrum. Không dùng mốc đó làm chất lượng Cepstrum đúng thuật toán.

[Báo cáo đợt 4](docs/STAGE4_RESULTS.md) có raw result, 17 giao thức không giám sát, replay nhân quả và giới hạn tổng quát hóa. [Chương V](docs/05_unsupervised_thresholding_analysis.md) có bảng tích hợp Champion mới và phân biệt với các mốc lịch sử chưa đủ bằng chứng. [Đợt 1](docs/STAGE1_FIXES.md), [đợt 2](docs/STAGE2_FIXES.md), [đợt 3](docs/STAGE3_FIXES.md) giữ các mốc trước sửa. Các artifact lịch sử được bảo toàn.

LAB hiện hành chỉ có thống kê toàn file nên chưa thể fit lại Gaussian V/UV có giám sát hoặc đánh giá Accuracy/F1. MAPE không xác định dùng N/A; Final Score yêu cầu đủ ba thành phần. [Vấn đề còn lại](docs/KNOWN_ISSUES.md).

## Cấu trúc

```text
FAC/
├── configs/                 # Spec JSON có phiên bản cho hạ tầng thí nghiệm mới
├── TinHieuHuanLuyen/         # WAV/LAB train
├── TinHieuKiemThu/           # WAV/LAB test
├── src/
│   ├── core/                # Audio, framing, LAB, ACF, AMDF, Cepstrum, detector
│   ├── pipeline/            # Feature contract, engine, extractor và cache theo run
│   ├── plugins/
│   │   ├── pre_processing/  # Bandpass, center clipping
│   │   ├── decision/        # Hysteresis
│   │   └── post_processing/ # Energy extension, Viterbi
│   ├── analysis/            # Thresholds và evaluation
│   ├── experiments/         # Spec, manifest, store, registry và runner/lifecycle
│   ├── metrics/             # Metric v1, profile tổng hợp, metadata và đối sánh
│   └── visualization/       # Plotter dùng chung
├── scripts/                 # Demo và benchmark
├── docs/                    # Năm chương báo cáo, hướng dẫn, vấn đề cần sửa
├── outputs/
│   ├── figures/             # Hình minh họa và benchmark được giữ
│   ├── reports/             # JSON/CSV benchmark và ngưỡng đã lưu
│   └── predictions/         # Kết quả CLI; tách train/test theo thư mục
├── archive/                 # Snapshot và artifact cũ, có manifest SHA-256
├── predict.py               # CLI dự đoán
├── HuongDan.txt             # Yêu cầu môn học
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Tài liệu

1. [ACF: lý thuyết và thực nghiệm](docs/01_acf_theory_and_experiments.md).
2. [AMDF: lý thuyết và thực nghiệm](docs/02_amdf_theory_and_experiments.md).
3. [Cepstrum: lý thuyết và thực nghiệm](docs/03_cepstrum_theory_and_experiments.md).
4. [Benchmark trên tập kiểm thử](docs/04_test_set_benchmark_and_analysis.md).
5. [Tìm ngưỡng không giám sát](docs/05_unsupervised_thresholding_analysis.md).
6. [Hướng dẫn kiểm thử](docs/TESTING.md), [sửa đợt 1](docs/STAGE1_FIXES.md), [sửa đợt 2](docs/STAGE2_FIXES.md), [sửa đợt 3](docs/STAGE3_FIXES.md), [vấn đề cần sửa](docs/KNOWN_ISSUES.md), [nhật ký dọn dẹp](docs/CLEANUP.md).
7. [Danh mục kết quả](outputs/README.md), [bản lưu lịch sử](archive/2026-10-28-pre-cleanup/README.md).

YIN, benchmark nhiễu AWGN và PTDB-TUG thuộc phiên bản trước, được lưu trong ZIP lịch sử; không phải tính năng của CLI hiện tại.
