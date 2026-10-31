# HƯỚNG DẪN KIỂM THỬ TRÊN DỮ LIỆU CỦA THẦY (PITCH TRACKING TESTING CHEATSHEET)

Tài liệu hướng dẫn nhanh dành cho buổi báo cáo / kiểm thử trực tiếp trên dữ liệu âm thanh của thầy.

---

## 1. CÔNG CỤ CHÍNH: `predict.py`

File [`predict.py`](predict.py) là công cụ CLI đa năng được thiết kế riêng để kiểm thử ngay lập tức trên dữ liệu mới:
- **Tự động kích hoạt 3 hệ thống Quán quân tối ưu nhất**:
  - **YIN Champion**: Thuật toán YIN kết hợp Viterbi Trellis Tracking (Sai số $F_0$ thấp nhất: **1.27 Hz**).
  - **ACF Champion (Config 29)**: Bandpass Filter + Center Clipping + Energy Extension + Viterbi (Sai số $F_0$: **1.86 Hz**, kháng nhiễu cực đoan tốt nhất).
  - **AMDF Champion (Config 13)**: Center Clipping + Viterbi Tracking (Sai số $F_0$: **1.50 Hz**).
- **Tự động đối soát nhãn chuẩn `.lab`**: Nếu trong thư mục có file `.lab` cùng tên với file `.wav`, hệ thống tự động tính toán toàn diện:
  - Sai số trung bình **$F_0\text{mean}$**: Sai số tuyệt đối ($|\Delta\text{Mean}|$) và lệch tương đối ($\%$).
  - Sai số độ lệch chuẩn **$F_0\text{std}$**: Sai số tuyệt đối ($|\Delta\text{Std}|$) và lệch tương đối ($\%$).
  - Độ chính xác phân loại Hữu thanh / Vô thanh: **V/UV Accuracy** và **Voiced F1-Score**.
- **Chế độ Blind Prediction (Dự đoán mù)**: Nếu thầy chỉ đưa file `.wav` không kèm `.lab`, hệ thống vẫn tự động trích xuất $F_0\text{mean} \pm F_0\text{std}$, tỉ lệ Hữu thanh, vẽ đồ thị và xuất file CSV chi tiết từng khung.

---

## 2. CÁC LỆNH CHẠY NHANH THƯỜNG DÙNG

### Kịch bản 1: Thầy đưa 1 file âm thanh bất kỳ
```powershell
# Chạy cả 3 thuật toán Quán quân trên file của thầy:
uv run python predict.py --input "duong/dan/toi/file_am_thanh.wav"

# Hoặc chỉ chạy riêng thuật toán Quán quân YIN:
uv run python predict.py --input "file_am_thanh.wav" --method yin
```

### Kịch bản 2: Thầy đưa một thư mục chứa nhiều file âm thanh
```powershell
# Chạy tự động toàn bộ file trong thư mục:
uv run python predict.py --input "duong/dan/toi/thu_muc_cua_thay/"
```

### Kịch bản 3: Chạy kiểm thử trên tập dữ liệu mẫu `TinHieuKiemThu/`
```powershell
uv run python predict.py --input TinHieuKiemThu/
```

### Kịch bản 4: Chạy script kiểm thử truyền thống `04_run_testing.py`
```powershell
# Chạy ACF Champion (Config 29):
uv run python scripts/04_run_testing.py --method acf --plugins all

# Chạy AMDF Champion (Config 13):
uv run python scripts/04_run_testing.py --method amdf --plugins all
```

---

## 3. KẾT QUẢ ĐẦU RA (OUTPUTS)

Tất cả kết quả được tự động lưu trong thư mục `outputs/predictions/`:
1. **File thống kê tổng hợp cả $F_0\text{mean}$ và $F_0\text{std}$**:
   - `outputs/predictions/evaluation_summary_metrics.csv`
   - Chứa đầy đủ: `File`, `Method`, `Ref_F0mean`, `Pred_F0mean`, `Abs_Err_F0mean`, `Ref_F0std`, `Pred_F0std`, `Abs_Err_F0std`, `V_UV_Accuracy`, `Voiced_F1`.
2. **File bảng số liệu chi tiết theo từng khung (Frame-by-frame)**:
   - `outputs/predictions/<tên_file>_pitch_prediction.csv`
   - Cột thời gian (giây) và giá trị $F_0$ (Hz) của từng thuật toán.
3. **Ảnh biểu đồ trực quan hóa cao độ**:
   - `outputs/predictions/<tên_file>_pitch_contour.png`
   - Hiển thị dạng sóng, vùng hữu thanh chuẩn (Ground Truth), đường dải chuẩn $F_0\text{mean} \pm F_0\text{std}$ và quỹ đạo $F_0$ của các thuật toán.

---

## 4. GIẢI THÍCH CHỈ SỐ VỚI THẦY

- **$F_0\text{mean}$ (Hz)**: Tần số cơ bản trung bình của các khung hữu thanh. Phản ánh độ cao âm thanh đặc trưng của người nói (Nam $\approx 100 - 160\text{ Hz}$, Nữ $\approx 180 - 250\text{ Hz}$).
- **$F_0\text{std}$ (Hz)**: Độ lệch chuẩn của tần số cơ bản qua các khung hữu thanh. Phản ánh mức độ biến thiên ngữ điệu (intonation / pitch dynamic range) của câu nói.
- **Voiced F1-Score (%)**: Đánh giá độ tin cậy phân loại nguyên âm / phụ âm hữu thanh, cân bằng giữa Precision và Recall.
- **Viterbi Trellis Tracking**: Kỹ thuật quy hoạch động nắn đường cong $F_0$ toàn cục, loại trừ hiện tượng nhảy quãng tám (Octave Doubling / Halving) và làm mượt $F_0\text{std}$ về sát nhãn chuẩn.
