# FAC - Speech Processing Project (F0 Estimation & V/UV Classification)

Dự án nghiên cứu và hiện thực các thuật toán ước lượng tần số cơ bản \(F_0\) (Pitch) và phân loại Hữu thanh (Voiced) / Vô thanh (Unvoiced) trên miền thời gian sử dụng **Hàm tự tương quan (ACF)** và **Hàm hiệu độ lớn trung bình (AMDF)**.

---

## 1. Cấu trúc thư mục (Directory Structure)

```
D:\SP\
├── TinHieuHuanLuyen/            # Dữ liệu âm thanh và nhãn ground truth huấn luyện (*.wav, *.lab)
├── TinHieuKiemThu/              # Dữ liệu âm thanh và nhãn ground truth kiểm thử (*.wav, *.lab)
├── HuongDan.txt                 # Hướng dẫn chi tiết yêu cầu đồ án
│
├── src/                         # Mã nguồn chính theo kiến trúc module
│   ├── core/                    # Xử lý tín hiệu tiếng nói cốt lõi
│   │   ├── audio.py             # Đọc wav, chuẩn hóa, phân khung (framing), năng lượng ngắn hạn (STE)
│   │   ├── lab_parser.py        # Đọc và bóc tách dữ liệu mốc thời gian, nhãn từ file *.lab
│   │   ├── acf.py               # Thuật toán Autocorrelation Function (ACF) & tìm cực đại
│   │   ├── amdf.py              # Thuật toán Average Magnitude Difference Function (AMDF) & tìm cực tiểu
│   │   └── pitch_detector.py    # Pipeline ước lượng F0 hoàn chỉnh & gán nhãn V/UV/Sil
│   │
│   ├── analysis/                # Phân tích thống kê và đánh giá sai số
│   │   ├── threshold.py         # Trích xuất phân bố (mean, std), tính ngưỡng T tối ưu (Gaussian/Histogram)
│   │   └── evaluation.py        # Tính F0mean, F0std, độ lệch tuyệt đối & tương đối so với Ground Truth
│   │
│   └── visualization/           # Các hàm trực quan hóa và vẽ biểu đồ Matplotlib
│       └── plotter.py           # Vẽ waveform + F0 contour, phân bố V/UV, phân tích khung đại diện
│
├── scripts/                     # Các kịch bản thực thi theo đúng từng yêu cầu đề bài
│   ├── 01_demo_frames.py        # [Yêu cầu 4.1] Minh họa 1 khung Voiced vs 1 khung Unvoiced
│   ├── 02_train_threshold.py    # [Yêu cầu 4.2] Tìm ngưỡng V/UV tối ưu từ tập huấn luyện
│   ├── 03_compare_params.py     # [Yêu cầu 4.2] Khảo sát ảnh hưởng tham số (20ms vs 30ms, ngưỡng T)
│   └── 04_run_testing.py        # [Yêu cầu 4.3] Chạy 4 file test, xuất 4 hình và bảng so sánh F0mean/std
│
├── outputs/                     # Kết quả đầu ra
│   ├── figures/                 # Chứa các biểu đồ hình ảnh (.png)
│   └── reports/                 # Bảng số liệu báo cáo (.csv, .json)
│
├── pyproject.toml               # Cấu hình dự án & thư viện
├── requirements.txt             # Danh sách thư viện Python
└── .gitignore                   # Loại trừ file tạm, cache, venv
```

---

## 2. Thiết lập môi trường (Environment Setup)

Dự án sử dụng Python (>= 3.9) với các thư viện xử lý khoa học: `numpy`, `scipy`, `matplotlib`.

Sử dụng công cụ `uv` cực nhanh đã có sẵn trên máy:
```powershell
# Tạo môi trường ảo và cài thư viện
uv venv
uv pip install -r requirements.txt
```
Hoặc chạy trực tiếp bất kỳ script nào:
```powershell
uv run python scripts/04_run_testing.py
```
