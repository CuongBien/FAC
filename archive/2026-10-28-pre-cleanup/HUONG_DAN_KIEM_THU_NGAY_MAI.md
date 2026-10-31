# HƯỚNG DẪN THỰC CHIẾN KIỂM THỬ TRÊN DỮ LIỆU CỦA THẦY (TESTING GUIDE)

> **Tài liệu bỏ túi cho buổi bảo vệ / thực nghiệm:** Hướng dẫn từng bước chạy các thuật toán Pitch Tracking tốt nhất của đề tài trên dữ liệu âm thanh mới do Thầy cung cấp.

---

## 1. TỔNG QUAN 3 THUẬT TOÁN QUÁN QUÂN (CHAMPIONS)

Khi Thầy yêu cầu chạy thuật toán tốt nhất, hệ thống đã cấu hình sẵn 3 Quán quân tối ưu nhất:

* **1. YIN Champion (Vô địch độ chính xác cao độ & Âm thanh sạch):**
  * **Cấu hình:** `YIN Baseline` kết hợp `Viterbi Tracking` ($T = 0.25$).
  * **Hiệu năng:** Sai số F0 trung bình **$1.27\text{ Hz}$**, Voiced F1-Score **$92.22\%$**.
  * **Ưu điểm vượt trội:** Loại bỏ hiện tượng Formant Ripple nhờ hàm CMNDF $d'(\tau)$ và nội suy parabol dưới mức mẫu rời rạc. Vô địch tuyệt đối ở môi trường sạch và nhiễu nhẹ ($\text{SNR} \ge 10\text{ dB}$).

* **2. ACF Champion (Vô địch Kháng nhiễu cực đoan):**
  * **Cấu hình:** Cấu hình 29 (`Bandpass Filter 70–900Hz` + `Center Clipping 40%` + `Energy Extension` + `Viterbi Tracking`, $T = 0.3953$).
  * **Hiệu năng:** Sai số F0 trung bình **$1.86\text{ Hz}$**, Voiced F1-Score **$91.39\%$** (cải thiện $41.3\%$ so với ACF Baseline $3.17\text{ Hz}$).
  * **Ưu điểm vượt trội:** Nhờ tính chất trực giao của hàm tương quan, nhiễu trắng tự triệt tiêu lẫn nhau ($R_{ww}(\tau) = 0$ với $\tau > 0$). Giữ vững sai số xuất sắc **$2.75\text{ Hz}$** tại $0\text{ dB}$ và **$2.17\text{ Hz}$** tại $-5\text{ dB}$.

* **3. AMDF Champion (Vô địch Tốc độ tính toán):**
  * **Cấu hình:** Cấu hình 13 (`Center Clipping 40%` + `Viterbi Tracking`, $T = 0.6488$).
  * **Hiệu năng:** Sai số F0 trung bình **$1.50\text{ Hz}$**, Voiced F1-Score **$91.42\%$**.
  * **Ưu điểm vượt trội:** Chỉ dùng phép trừ và trị tuyệt đối, hoàn toàn không cần phép nhân; tối ưu cho hệ thống nhúng / DSP thời gian thực.

---

## 2. CHEAT SHEET CÂU LỆNH CHẠY NHANH (COPY-PASTE DÙNG NGAY)

Hệ thống đã trang bị sẵn công cụ dòng lệnh chuyên dụng `predict.py` ở thư mục gốc.

### Trường hợp 1: Thầy đưa 1 file âm thanh cụ thể
Thầy mở 1 file (ví dụ `cau_noi.wav`), em mở terminal gõ:

```powershell
# Chạy đồng thời cả 3 Quán quân để so sánh kết quả
uv run python predict.py --input "duong_dan/file_cua_thay.wav"
```

* **Kết quả hiển thị ngay trên Terminal:**
  * Tần số $F_0$ trung bình (Hz), độ lệch chuẩn (Std).
  * Tỉ lệ khung Hữu thanh (Voiced ratio %).
  * Nếu thư mục có sẵn file nhãn `.lab` cùng tên, hệ thống tự động đo: Sai số tuyệt đối $|ΔF_0|$, Điểm Voiced F1-Score, V/UV Accuracy.
* **Kết quả tự động xuất ra thư mục `outputs/predictions/`:**
  * File ảnh đồ thị trực quan: `outputs/predictions/<tên_file>_pitch_contour.png`
  * Bảng số liệu F0 từng khung: `outputs/predictions/<tên_file>_pitch_prediction.csv`

---

### Trường hợp 2: Thầy đưa cả một thư mục chứa nhiều file .wav
Thầy copy cho một thư mục chứa nhiều câu âm thanh (ví dụ thư mục `D:\DataThay\`):

```powershell
uv run python predict.py --input "D:\DataThay\"
```

* Hệ thống sẽ tự động duyệt toàn bộ các file `.wav` trong thư mục, chạy phân tích từng file và in ra **Bảng tổng kết trung bình toàn bộ tập kiểm thử**.

---

### Trường hợp 3: Thầy yêu cầu chỉ chạy riêng 1 thuật toán cụ thể
Nếu Thầy bảo: *"Chỉ chạy thuật toán YIN cho thầy xem"* hoặc *"Chỉ chạy ACF"*:

```powershell
# Chỉ chạy YIN (Champion vs Baseline)
uv run python predict.py --input "file_cua_thay.wav" --method yin

# Chỉ chạy ACF (Champion vs Baseline)
uv run python predict.py --input "file_cua_thay.wav" --method acf

# Chỉ chạy AMDF (Champion vs Baseline)
uv run python predict.py --input "file_cua_thay.wav" --method amdf
```

---

### Trường hợp 4: Thầy muốn đối sánh 6 hệ thống (Baseline vs Champion)

```powershell
uv run python predict.py --input "file_cua_thay.wav" --method all
```

Lệnh này sẽ chạy toàn bộ: ACF Base, ACF Champ, AMDF Base, AMDF Champ, YIN Base, YIN Champ trên cùng một biểu đồ đối sánh.

---

## 3. CÁCH ĐỌC KẾT QUẢ XUẤT RA ĐỂ TRẢ LỜI THẦY

Khi lệnh chạy xong, mở file ảnh trong thư mục `outputs/predictions/`:

* **Panel 1 (Dạng sóng):** Cho Thầy thấy vùng nào biên độ lớn, có dao động tuần hoàn rõ rệt (nguyên âm / phụ âm hữu thanh) và vùng nào là khoảng lặng / tiếng ồn gió.
* **Panel 2 (Quỹ đạo cao độ F0):**
  * Đường màu xanh lá (`YIN Champion`): Quỹ đạo mượt mà nhất, bám sát các biến thiên ngữ điệu tinh vi.
  * Đường nét đứt màu xanh dương (`ACF Champion`): Rất ổn định, không bị đứt đoạn hay nhảy quãng tám.
  * Đường nét chấm gạch màu đỏ (`AMDF Champion`): Đáy sâu, bắt điểm bắt đầu và kết thúc từ rất nhạy.
* **File CSV (`<tên>_pitch_prediction.csv`):**
  * Chứa chi tiết từng khung thời gian ($10\text{ ms}$): Cột `Time_Sec`, `YIN_Champion_F0_Hz`, `ACF_Champion_F0_Hz`, `AMDF_Champion_F0_Hz`.
  * Khung vô thanh (Unvoiced) được gán giá trị chính xác bằng `0.00 Hz`.

---

## 4. BỘ CÂU HỎI "HỎI XOÁY ĐÁP XOAY" CỦA THẦY & CÂU TRẢ LỜI ĂN ĐIỂM

* **Câu 1: "Làm sao thuật toán của em phân biệt được Hữu thanh (Voiced) và Vô thanh (Unvoiced)?"**
  * *Trả lời:* "Dạ thưa Thầy, hệ thống kết hợp 2 tiêu chí vật lý:
    1. Tiêu chí Năng lượng ngắn hạn (Short-Time Energy) để loại bỏ khoảng lặng/tạp âm nền.
    2. Độ cao đỉnh cực đại của hàm chuẩn hóa (ACF/CMNDF). Đặc biệt, hệ thống dùng **Ngưỡng trễ kép Hysteresis (Schmitt Trigger)** với $T_{\text{high}}$ và $T_{\text{low}}$ để mô phỏng quán tính dao động sinh học của dây thanh, chống hiện tượng rung lật quyết định (decision chattering)."

* **Câu 2: "Tại sao pitch đôi khi bị nhảy vọt gấp đôi hoặc tụt một nửa (quãng tám) và em xử lý thế nào?"**
  * *Trả lời:* "Dạ thưa Thầy, đó là lỗi Octave Doubling / Halving:
    1. Nhảy gấp đôi xảy ra khi cộng hưởng Formant tạo ra đỉnh phụ cạnh tranh với đỉnh chu kỳ $T_0$. Em giải quyết bằng **Center Clipping** (cắt $40\%$ biên độ giữa) để đập phẳng dao động formant.
    2. Em bổ sung **Viterbi Tracking** sử dụng quy hoạch động toàn cục để phạt nặng chi phí khi có bước nhảy tần số bất thường giữa các khung liên tiếp ($|\log(f_t / f_{t-1})|$), triệt tiêu hoàn toàn các bước nhảy quãng tám đột ngột."

* **Câu 3: "Nếu file âm thanh của thầy bị dính tiếng ồn đường phố hoặc tạp âm micro thì thuật toán nào tốt nhất?"**
  * *Trả lời:* "Dạ thưa Thầy, **ACF Champion** là thuật toán kháng nhiễu mạnh nhất. Về mặt toán học, hàm tương quan của nhiễu trắng độc lập tự triệt tiêu về 0 ở mọi độ trễ $\tau > 0$ ($R_{ww}(\tau) = \sigma^2 \delta[\tau]$), năng lượng nhiễu chỉ dồn về đỉnh gốc $\tau = 0$, giúp đỉnh chu kỳ $T_0$ vẫn nhô cao bền bỉ ngay cả ở mức nhiễu cực đoan $0\text{ dB}$ và $-5\text{ dB}$."

* **Câu 4: "Giữa 3 thuật toán ACF, AMDF, YIN, em khuyến nghị sử dụng thuật toán nào trong thực tế?"**
  * *Trả lời:* "Dạ thưa Thầy:
    1. Trong môi trường sạch hoặc giao tiếp văn phòng ($\text{SNR} \ge 10\text{ dB}$): Khuyến nghị dùng **YIN** vì đạt độ chính xác cao độ cao nhất ($1.27\text{ Hz}$).
    2. Trong môi trường công nghiệp, đường phố nhiều tạp âm ($\text{SNR} \le 5\text{ dB}$): Khuyến nghị dùng **ACF Enhanced** vì tính kháng nhiễu vượt trội.
    3. Trong thiết bị nhúng vi điều khiển / chip DSP pin yếu: Khuyến nghị dùng **AMDF Enhanced** vì không tốn tài nguyên phép nhân."

---

## 5. XỬ LÝ SỰ CỐ TỨC THÌ (TROUBLESHOOTING)

* **Sự cố 1: File âm thanh của Thầy có định dạng lạ hoặc tần số lấy mẫu khác 16kHz / 44.1kHz?**
  * *Giải pháp:* Hàm `load_wav` trong `src/core/audio.py` tự động chuyển đổi stereo thành mono và tự động thích ứng với mọi tần số lấy mẫu (8kHz, 16kHz, 22.05kHz, 44.1kHz, 48kHz). Em chỉ cần truyền đường dẫn file vào là chạy được ngay.
* **Sự cố 2: Thầy không có file nhãn `.lab`?**
  * *Giải pháp:* `predict.py` tự động nhận diện nếu không có file `.lab`, hệ thống chuyển sang chế độ dự đoán mù (Blind Prediction), vẫn xuất đầy đủ F0mean, F0std, biểu đồ và CSV mà không báo lỗi.
* **Sự cố 3: Máy tính trên lớp không có internet?**
  * *Giải pháp:* Toàn bộ môi trường ảo `.venv` và thư viện đã được cài đặt cục bộ qua `uv`. Không cần kết nối internet vẫn chạy mượt mà 100%.
