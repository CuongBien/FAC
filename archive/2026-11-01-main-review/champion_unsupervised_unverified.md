# Bảng cũ chưa đủ bằng chứng tái lập

Nguồn: commit `5a9273f`, mục IV của chương V. Giữ để truy xuất lịch sử. Các nhãn Baseline, kết luận 50/50 và khẳng định kỷ lục bên dưới đã bị rút khỏi báo cáo hiện hành. Một số số liệu chưa có script/mô hình/kết quả từng file tương ứng. Không dùng bản lưu này làm kết luận đã kiểm chứng.

## IV. BẢNG TỔNG HỢP TOÀN DIỆN CẢ 8 PHƯƠNG PHÁP HỌC KHÔNG GIÁM SÁT KHI TÍCH HỢP VÀO CHUỖI QUÁN QUÂN (CHAMPION PIPELINE)

Để giải đáp triệt để câu hỏi: *"Các phương pháp học không giám sát có thể nâng cấp thêm cho chuỗi Quán quân hoàn chỉnh (đã có bộ tiền xử lý Bandpass/Clipping và hậu xử lý Energy/Viterbi) hay không?"*, nhóm đã thực hiện tích hợp trực tiếp **toàn bộ 8 phương pháp** (4 phương pháp từ Histogram/Phân cụm 1D và 4 phương pháp Nâng cao) vào cấu hình Quán quân chính thức trên tập Kiểm thử (`TinHieuKiemThu`).

Mọi phương pháp đều được đánh giá qua bảng chuẩn 8 cột thống nhất của đề tài:
* $\Delta F_0$: Sai số tuyệt đối trung bình tần số cơ bản (Hz).
* $\text{MAPE } F_0$: Sai số tương đối trung bình tần số cơ bản (%).
* $\Delta\text{Std}$: Sai số tuyệt đối độ lệch chuẩn tần số cơ bản (Hz).
* $\text{MAPE Std}$: Sai số tương đối độ lệch chuẩn tần số cơ bản (%).
* $\Delta\text{Num}$: Sai số tuyệt đối số lượng khung hữu thanh (Khung).
* $\text{MAPE Num}$: Sai số tương đối số lượng khung hữu thanh (%).
* $\text{FINAL SCORE}$: Điểm lỗi tổng hợp $= (\text{MAPE } F_0 + \text{MAPE Std} + \text{MAPE Num}) / 3$ (Càng thấp càng tối ưu).

---

### 1. Thuật toán ACF (Chuỗi Quán quân: Bandpass 70–900 Hz + Energy + Viterbi)
*Khung phân tích 30 ms, bước nhảy 10 ms. Mốc đối sánh Quán quân quét lưới Stage 4: $T = 0.4500$, Final Score = **2.39%**.*

| STT | Phương pháp học không giám sát | Cơ chế / Ngưỡng $T$ | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | **FINAL SCORE (%)** | Đánh giá so với Quán quân gốc |
|:---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---|
| **1** | **Ngưỡng Động Thích Nghi Ngữ Cảnh** | **Adaptive STE** | **1.61** | **1.00** | **0.94** | **2.72** | **5.5** | **2.80** | **2.17%** | 🔥 **Vượt trội nhất (Kỷ lục mới)!** |
| **2** | **Otsu (Cực đại phương sai)** | **$T = 0.4559$** | **1.87** | **1.17** | **0.87** | **2.40** | **6.0** | **3.24** | **2.27%** | 🔥 **Tối ưu hơn Quán quân gốc** |
| **3** | **K-Means (Phân cụm 1D)** | **$T = 0.4599$** | **1.89** | **1.19** | **0.81** | **1.96** | **6.8** | **3.76** | **2.30%** | 🔥 **Tối ưu hơn Quán quân gốc** |
| — | *Quán quân Quét lưới (Stage 4)* | *$T = 0.4500$* | *1.80* | *1.12* | *0.90* | *2.60* | *6.0* | *3.43* | ***2.39%*** | *Mốc chuẩn Quán quân* |
| — | *Baseline lịch sử (BT2)* | *$T = 0.4408$* | *1.80* | *1.12* | *0.93* | *2.70* | *6.2* | *3.65* | ***2.49%*** | *Chuẩn tham số cũ* |
| 4 | Median (Tiên nghiệm 50/50) | $T = 0.4293$ | 1.81 | 1.12 | 0.97 | 2.86 | 6.5 | 3.94 | **2.64%** | Khá |
| 5 | 1D GMM-EM (50/50 Liên tục) | $T = 0.4294$ | 1.81 | 1.12 | 0.97 | 2.86 | 6.5 | 3.94 | **2.64%** | Khá (trùng điểm phân vị) |
| 6 | 2-State HMM (Baum-Welch + Viterbi) | Viterbi Decoding | 2.97 | 1.86 | 5.16 | 12.86 | 16.0 | 10.20 | **8.30%** | Trễ biên chuyển tiếp |
| 7 | Đa đặc trưng 2D (Periodicity + ZCR) | Phân cụm 2D | 3.89 | 2.42 | 4.65 | 10.67 | 25.8 | 14.80 | **9.30%** | Nhiễu do đặc trưng ZCR |
| 8 | Đáy Thung Lũng (Valley Minimum) | $T = 0.2568$ | 3.10 | 2.02 | 5.44 | 33.16 | 14.0 | 10.34 | **15.17%** | Ngưỡng quá thấp, lọt tạp âm |

---

### 2. Thuật toán AMDF (Chuỗi Quán quân: Energy + Viterbi)
*Khung phân tích 30 ms, bước nhảy 10 ms. Mốc đối sánh Quán quân quét lưới Stage 4: $T = 0.5000$, Final Score = **2.76%**.*

| STT | Phương pháp học không giám sát | Cơ chế / Ngưỡng $T$ | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | **FINAL SCORE (%)** | Đánh giá so với Quán quân gốc |
|:---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---|
| — | ***Quán quân Quét lưới (Stage 4)*** | ***$T = 0.5000$*** | ***0.56*** | ***0.33*** | ***1.30*** | ***3.76*** | ***7.2*** | ***4.18*** | ***2.76%*** | 🏆 **Tối ưu nhất toàn diện** |
| **1** | **1D GMM-EM (50/50 Liên tục)** | **$T = 0.4239$** | **0.75** | **0.47** | **1.21** | **2.88** | **9.8** | **5.28** | **2.88%** | 🔥 **Sát nút Quán quân gốc** |
| **2** | **Ngưỡng Động Thích Nghi Ngữ Cảnh** | **Adaptive STE** | **0.53** | **0.31** | **1.37** | **3.98** | **8.0** | **4.43** | **2.91%** | 🔥 **Tương đương Quán quân gốc** |
| — | *Baseline lịch sử (BT2)* | *$T = 0.4380$* | *0.76* | *0.48* | *1.28* | *3.41* | *9.5* | *5.06* | ***2.98%*** | *Khá* |
| 3 | Median (Tiên nghiệm 50/50) | $T = 0.4565$ | 0.70 | 0.43 | 1.28 | 3.41 | 9.5 | 5.15 | **3.00%** | Khá |
| 4 | Đa đặc trưng 2D (Periodicity + ZCR) | Phân cụm 2D | 0.49 | 0.29 | 1.37 | 4.08 | 9.2 | 5.16 | **3.17%** | Khá |
| 5 | Đáy Thung Lũng (Valley Minimum) | $T = 0.4033$ | 0.57 | 0.34 | 1.09 | 2.69 | 12.5 | 7.03 | **3.35%** | Trung bình |
| 6 | K-Means (Phân cụm 1D) | $T = 0.4006$ | 0.59 | 0.36 | 0.97 | 2.30 | 13.5 | 7.80 | **3.49%** | Trung bình |
| 7 | Otsu (Cực đại phương sai) | $T = 0.3950$ | 0.67 | 0.42 | 0.93 | 2.25 | 14.8 | 8.50 | **3.72%** | Trung bình |
| 8 | 2-State HMM (Baum-Welch + Viterbi) | Viterbi Decoding | 2.67 | 1.73 | 1.40 | 5.41 | 21.5 | 14.00 | **7.04%** | Mất khung biên âm thanh |

---

### 3. Thuật toán Cepstrum (Chuỗi Quán quân: Bandpass + Center Clipping 40% + Energy + Viterbi)
*Khung phân tích 30 ms, bước nhảy 10 ms. Mốc đối sánh Quán quân quét lưới Stage 4: $T = 0.1000$, Final Score = **22.94%**.*

| STT | Phương pháp học không giám sát | Cơ chế / Ngưỡng $T$ | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | **FINAL SCORE (%)** | Đánh giá so với Quán quân gốc |
|:---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---|
| **1** | **2-State HMM (Baum-Welch + Viterbi)** | **Viterbi Decoding** | **5.88** | **3.71** | **0.93** | **2.50** | **53.0** | **36.73** | **14.31%** | 🔥 **Vượt trội Quán quân gốc!** |
| **2** | **1D GMM-EM (fit trên đặc trưng BP+Clip)** | **$T = 0.1261$** | **3.57** | **2.35** | **3.26** | **13.75** | **37.8** | **27.10** | **14.40%** | 🔥 **Vượt trội Quán quân gốc!** |
| **3** | **Đa đặc trưng 2D (Periodicity + ZCR)** | **Phân cụm 2D** | **5.72** | **3.50** | **6.22** | **25.59** | **29.0** | **17.85** | **15.65%** | 🔥 **Vượt trội Quán quân gốc!** |
| **4** | **Ngưỡng Động Thích Nghi Ngữ Cảnh** | **Adaptive STE** | **3.52** | **2.38** | **6.09** | **33.22** | **18.8** | **13.04** | **16.21%** | 🔥 **Vượt trội Quán quân gốc!** |
| 5 | Median (Tiên nghiệm 50/50) | $T = 0.0779$ | 4.00 | 2.79 | 9.35 | 50.98 | 10.2 | 8.09 | **20.62%** | Tốt hơn Quán quân gốc |
| — | *Baseline lịch sử (BT2)* | *$T = 0.0600$* | *4.06* | *2.87* | *9.86* | *55.58* | *13.2* | *9.68* | ***22.71%*** | *Tương đương Quán quân* |
| — | *Quán quân Quét lưới (Stage 4)* | *$T = 0.1000$* | *6.12* | *4.13* | *9.62* | *51.62* | *18.8* | *13.09* | ***22.94%*** | *Mốc chuẩn Quán quân* |
| 6 | 1D GMM-EM (fit trên Cepstrum thô) | $T = 0.1024$ | 5.37 | 3.74 | 9.62 | 51.48 | 20.2 | 14.21 | **23.14%** | Tương đương Quán quân |
| 7 | Otsu (Cực đại phương sai) | $T = 0.1903$ | 8.90 | 5.71 | 1.91 | 9.52 | 82.2 | 58.13 | **24.46%** | Ngưỡng quá cao |
| 8 | K-Means (Phân cụm 1D) | $T = 0.1945$ | 9.35 | 5.95 | 1.99 | 9.86 | 83.8 | 59.17 | **25.00%** | Ngưỡng quá cao |
| 9 | Đáy Thung Lũng (Valley Minimum) | $T = 0.2556$ | 13.19 | 8.07 | 6.23 | 21.51 | 107.2 | 74.97 | **34.85%** | Lệch nặng do phân phối lệch |

---

### 4. Kết Luận Khoa Học Thực Nghiệm:
1. **Sức mạnh cộng hưởng giữa Tiền/Hậu xử lý và Học không giám sát:**
   * Các kỹ thuật học không giám sát không chỉ hoạt động tốt ở mô hình độc lập mà khi kết hợp vào chuỗi Quán quân hoàn chỉnh còn phát huy sức mạnh vượt bậc.
   * Trên **ACF**, giải pháp **Ngưỡng Động Adaptive STE (2.17%)**, **Otsu (2.27%)** và **K-Means (2.30%)** đã chính thức vượt qua Quán quân quét lưới thủ công (**2.39%**).
   * Trên **AMDF**, Quán quân quét lưới ($T=0.5000$, **2.76%**) giữ vị trí số một, theo sát là **1D GMM-EM (2.88%)** và **Ngưỡng Động Adaptive STE (2.91%)**.
   * Trên **Cepstrum**, sự phối hợp giữa bộ lọc dải thông + xén biên trung tâm với mô hình xác suất nâng cao như **2-State HMM (14.31%)** và **1D GMM-EM (14.40%)** đã giảm tới hơn $8.5\%$ điểm lỗi so với cấu hình Quán quân lưới cố định (**22.94%**).
2. **Khẳng định tính đúng đắn của giả thiết tiên nghiệm 50/50:**
   * Các phương pháp bám sát giả thiết $P(V) \approx 50\%$ (Median, Otsu, GMM-EM 50/50) đều nằm trong nhóm dẫn đầu, chứng minh giả thiết mà Giảng viên đặt ra là hoàn toàn chính xác và có giá trị thực tiễn rất cao.

---
