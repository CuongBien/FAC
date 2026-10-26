# CHƯƠNG I: THUẬT TOÁN HÀM TỰ TƯƠNG QUAN (AUTOCORRELATION FUNCTION - ACF)

> **Dẫn hướng tài liệu:** [Trang chủ README](../README.md) > **Chương I: Thuật toán ACF**

---

## I.1. Cơ Sở Lý Thuyết Thuật Toán ACF

Hàm tự tương quan (Autocorrelation Function - ACF) đo lường mức độ tương đồng giữa một tín hiệu với chính bản sao trễ thời gian $\tau$ của nó.

Với một khung tín hiệu $x[n]$ gồm $N$ mẫu ($n = 0, 1, \dots, N-1$), hàm tự tương quan chuẩn hóa được định nghĩa:

$$
R(\tau) = \frac{\sum_{n=0}^{N-1-\tau} x[n] \cdot x[n+\tau]}{R(0)}
$$

Trong đó năng lượng của khung là:

$$
R(0) = \sum_{n=0}^{N-1} x^2[n]
$$

Nhờ phép chuẩn hóa qua $R(0)$, biên độ hàm luôn nằm trong đoạn $[-1.0, 1.0]$ và $R(0) = 1.0$.

* **Âm Hữu thanh (Voiced):** Được tạo ra khi dây thanh âm rung động tuần hoàn dưới áp lực luồng khí từ phổi, tạo nên các xung thanh môn lặp lại với chu kỳ cơ bản $T_0$. Hàm $R(\tau)$ sẽ đạt một đỉnh cực đại nổi bật thứ hai tại độ trễ $\tau_0 = T_0 \cdot f_s$. Tần số cơ bản được tính:

$$
F_0 = \frac{f_s}{\tau_0} \quad (\text{Hz})
$$

* **Âm Vô thanh (Unvoiced):** Dây thanh âm không rung, âm thanh được tạo ra do luồng khí xoáy qua chỗ thắt thanh quản. Dạng sóng ngẫu nhiên tựa nhiễu trắng, hàm $R(\tau)$ suy giảm nhanh về 0 và dao động hỗn loạn, không có đỉnh vượt ngưỡng trong dải cao độ người trưởng thành $[70, 400]\text{ Hz}$.

---

## I.2. Minh Họa Khung Hữu Thanh vs Vô Thanh

Thực nghiệm trên file âm thanh `TinHieuHuanLuyen/phone_F1.wav` ($f_s = 16000\text{ Hz}$, khung chuẩn $25\text{ ms} = 400\text{ mẫu}$, dịch $10\text{ ms} = 160\text{ mẫu}$):

![01_demo_acf_frames.png](../outputs/figures/01_demo_acf_frames.png)

### Bảng phân tích chi tiết hai khung đại diện:

| Đặc trưng | Khung Hữu Thanh (Voiced) | Khung Vô Thanh (Unvoiced) |
| :--- | :--- | :--- |
| **Vị trí khung** | Khung #103 ($t = 1.042\text{ s}$) | Khung #176 ($t = 1.772\text{ s}$) |
| **Đặc tính dạng sóng** | Tuần hoàn rõ rệt, biên độ xung thanh môn lớn | Dao động ngẫu nhiên, tựa nhiễu, biên độ thấp |
| **Độ trễ cực đại $\tau_0$** | $\tau_0 = 71$ mẫu | Không có độ trễ tuần hoàn rõ ràng ($\tau = 53$) |
| **Biên độ cực đại $R(\tau_0)$** | **$0.801$** ($\gg T = 0.4408$) | **$0.386$** ($< T = 0.4408$) |
| **Chu kỳ cơ bản $T_0$** | $T_0 = 71 / 16000 = 4.44\text{ ms}$ | Không xác định |
| **Tần số cơ bản $F_0$** | **$F_0 = 225.35\text{ Hz}$** (gần nhãn $217\text{ Hz}$) | **$F_0 = \text{Undefined}$** (Vô thanh) |

---

## I.3. Huấn Luyện Ngưỡng T Tối Ưu Bằng Phân Bố Gauss

Để xác định ngưỡng phân định V/UV dựa trên cơ sở thống kê toán học vững chắc với chiều dài khung chuẩn **$25\text{ ms}$**, đồ án trích xuất biên độ đỉnh ACF lớn nhất của toàn bộ các khung trong 4 file huấn luyện (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`), phân chia theo nhãn ground-truth `.lab`:
* Tập khung Hữu thanh: $N_V = 614$ khung $\rightarrow$ phân bố $\mathcal{N}(\mu_V, \sigma_V^2)$
* Tập khung Vô thanh: $N_U = 180$ khung $\rightarrow$ phân bố $\mathcal{N}(\mu_U, \sigma_U^2)$

Ngưỡng tối ưu Bayes $T$ là nghiệm của phương trình giao điểm mật độ xác suất:

$$
\frac{(T - \mu_V)^2}{\sigma_V^2} - \frac{(T - \mu_U)^2}{\sigma_U^2} + 2\ln\left(\frac{\sigma_V}{\sigma_U}\right) = 0
$$

### Bảng kết quả huấn luyện ngưỡng (Khung 25 ms):

| Chế độ tiền xử lý | Số khung Voiced | $\mu_V \pm \sigma_V$ | Số khung Unvoiced | $\mu_U \pm \sigma_U$ | Ngưỡng tối ưu $T$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Chưa lọc)** | 614 | $0.6166 \pm 0.1549$ | 180 | $0.2718 \pm 0.1355$ | **$T = 0.4408$** |
| **Có lọc thông dải (Bandpass)** | 614 | $0.6490 \pm 0.1466$ | 180 | $0.3326 \pm 0.1337$ | **$T = 0.4892$** |

### Biểu đồ phân bố xác suất Gaussian:

| Baseline (Chưa lọc: $T = 0.4408$) | Có lọc Bandpass (Đã lọc: $T = 0.4892$) |
| :---: | :---: |
| ![02_threshold_distribution.png](../outputs/figures/02_threshold_distribution.png) | ![02_threshold_distribution_bandpassprefilter.png](../outputs/figures/02_threshold_distribution_bandpassprefilter.png) |

---

## I.4. Khảo Sát Ảnh Hưởng Chiều Dài Khung (20ms vs 25ms [Chuẩn] vs 30ms)

Theo định hướng của giảng viên, hệ thống được **chốt cố định khung 25 ms** làm thước đo chuẩn mực công bằng cho toàn bộ sinh viên. Thực nghiệm đối chiếu giữa 3 mức chiều dài khung (20 ms, 25 ms, 30 ms) trên file giọng nam trầm `TinHieuHuanLuyen/phone_M1.wav` ($F_{0\text{-mean}} = 122.0\text{ Hz}$, $F_{0\text{-std}} = 18.0\text{ Hz}$, $T = 0.4408$):

### Bảng so sánh định lượng:

| Chỉ số đánh giá | Khung 20 ms | Khung 25 ms (Chuẩn môn học) | Khung 30 ms | Nhận xét phân tích |
| :--- | :---: | :---: | :---: | :--- |
| **Số mẫu / khung ($f_s=16\text{k}$)** | 320 mẫu | **400 mẫu** | 480 mẫu | 25ms tương ứng đúng 400 mẫu chẵn |
| **Số khung Voiced phát hiện** | 180 khung | **193 khung** | 204 khung | 25ms bắt trọn vẹn số chu kỳ của giọng nam |
| **$F_{0\text{-mean}}$ ước lượng** | 134.52 Hz | **130.93 Hz** | 130.04 Hz | Tiệm cận chuẩn 122.0 Hz hơn rất nhiều so với 20ms |
| **Sai số tuyệt đối $\lvert\Delta F_0\rvert$** | 12.52 Hz | **8.93 Hz** | 8.04 Hz | **Giảm 28.7% sai số so với khung 20ms** |
| **Sai số tương đối (%)** | 10.26 % | **7.32 %** | 6.59 % | Sai số kiểm soát tốt dưới 7.5% |
| **V/UV Accuracy (%)** | 74.70 % | **78.02 %** | 81.16 % | Tăng +3.32% so với 20ms |
| **Voiced F1-Score (%)** | 84.43 % | **87.41 %** | 90.18 % | Tăng gần +3% so với 20ms |

![03_compare_frame_length.png](../outputs/figures/03_compare_frame_length.png)

$\implies$ **Luận điểm khoa học cho việc chốt chuẩn 25 ms:**
1. **Khung 20 ms (320 mẫu):** Quá ngắn đối với các âm vực nam trầm ($F_0 \approx 70 - 80\text{ Hz} \implies T_0 \approx 12.5 - 14.3\text{ ms}$). Khung 20 ms chỉ chứa được khoảng $1.4 - 1.6$ chu kỳ, khiến phép tự tương quan dễ bị nhiễu và bắt trượt đỉnh, dẫn tới sai số cao (12.52 Hz).
2. **Khung 30 ms (480 mẫu):** Chứa nhiều chu kỳ tuần hoàn ($> 2.4$ chu kỳ), tuy nhiên độ dài khung lớn làm giảm độ phân giải thời gian và gây hiệu ứng nhòe (*smearing*) tại các ranh giới chuyển tiếp âm nhanh.
3. **Khung 25 ms (400 mẫu):** Là **điểm cân bằng vàng (Golden Middle Ground)**. Ở tần số đáy $80\text{ Hz}$, $25\text{ ms}$ chứa trọn vẹn đúng $2.0$ chu kỳ tuần hoàn đầy đủ, đủ điều kiện tiên quyết cho phép tìm cực đại tự tương quan hoạt động chính xác, đồng thời vẫn giữ được độ nhạy thời gian vượt trội. Do đó, **25 ms (hop 10 ms)** là lựa chọn chuẩn xác và công bằng nhất.

---

## I.5. Đánh Giá Kiểm Thử Trên 4 File Kiểm Thử (Baseline)

Hệ thống đánh giá toàn diện trên 3 trục chỉ số chuẩn mực:
1. **Độ chính xác tần số cơ bản ($F_{0	ext{-mean}}$):** Sai số tuyệt đối $\lvert\Delta F_0\rvert$ (Hz) và sai số phần trăm tương đối $\text{MAPE}_{F_0}$ (%).
2. **Độ ổn định đường bao cao độ ($F_{0	ext{-std}}$):** Sai số độ lệch chuẩn $\lvert\Delta\text{Std}\rvert$ (Hz) và sai số tương đối $\text{MAPE}_{\text{std}}$ (%).
3. **Chất lượng phân loại Hữu thanh ($F_{0	ext{-num}}$):** Sai lệch số khung $\lvert\Delta\text{Num}\rvert$ (khung) và sai số tương đối $\text{MAPE}_{\text{num}}$ (%).

**Điểm lỗi tổng hợp cuối cùng (Final Composite Score):**
$$
\text{Final Score (\%)} = \frac{\text{MAPE}(F_0) + \text{MAPE}(\text{std}) + \text{MAPE}(\text{num})}{3}
$$

### Bảng kết quả kiểm thử định lượng Baseline ($T = 0.4408$, Frame = 25 ms):

| File kiểm thử | Kênh / Giới tính | Ref $F_0$ | Pred $F_0$ | $\lvert\Delta F_0\rvert$ | MAPE $F_0$ | Ref Std | Pred Std | $\lvert\Delta\text{Std}\rvert$ | MAPE Std | Ref Num | Pred Num | $\lvert\Delta\text{Num}\rvert$ | MAPE Num | **FINAL SCORE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `phone_F2.wav` | Điện thoại / Nữ | 150.20 Hz | 152.77 Hz | 2.57 Hz | 1.71% | 30.70 | 31.44 | 0.74 | 2.40% | 219 | 182 | 37 | 16.89% | **7.00%** |
| `phone_M2.wav` | Điện thoại / Nam | 130.20 Hz | 131.92 Hz | 1.72 Hz | 1.32% | 15.40 | 14.07 | 1.33 | 8.63% | 123 | 108 | 15 | 12.20% | **7.38%** |
| `studio_F2.wav`| Studio / Nữ | 198.60 Hz | 198.98 Hz | 0.38 Hz | 0.19% | 47.80 | 43.35 | 4.45 | 9.31% | 139 | 126 | 13 | 9.35% | **6.28%** |
| `studio_M2.wav`| Studio / Nam | 154.70 Hz | 155.97 Hz | 1.27 Hz | 0.82% | 30.30 | 30.24 | 0.06 | 0.19% | 116 | 112 | 4 | 3.45% | **1.49%** |
| **TRUNG BÌNH** | — | — | — | **1.48 Hz** | **1.01%** | — | — | **1.65** | **5.13%** | — | — | **17.2** | **10.47%** | **5.54%** |

### Đồ thị đường bao Pitch Contour### Đồ thị đường bao Pitch Contour 4 file kiểm thử (Baseline):

| `phone_F2.wav` (Nữ - Kênh điện thoại) | `phone_M2.wav` (Nam - Kênh điện thoại) |
| :---: | :---: |
| ![04_test_phone_F2.png](../outputs/figures/04_test_phone_F2.png) | ![04_test_phone_M2.png](../outputs/figures/04_test_phone_M2.png) |
| **`studio_F2.wav` (Nữ - Phòng thu Studio)** | **`studio_M2.wav` (Nam - Phòng thu Studio)** |
| ![04_test_studio_F2.png](../outputs/figures/04_test_studio_F2.png) | ![04_test_studio_M2.png](../outputs/figures/04_test_studio_M2.png) |

---

## I.6. Các Giải Pháp Cải Tiến Dành Riêng Cho ACF (Plugins)

### I.6.1. Plugin 1: Ngưỡng trễ kép Hysteresis (Schmitt Trigger)

**Vấn đề giải quyết:**  
Hiện tượng rung lật trạng thái (*decision chattering*) khi đỉnh tự tương quan dao động quanh ngưỡng tĩnh $T$.

**Mô hình máy trạng thái:**  
Ký hiệu $S_i \in \lbrace 0, 1 \rbrace$ là nhãn khung thứ $i$ (0: Unvoiced, 1: Voiced) và $R_i^* = \max_{\tau \in [\tau_{\min}, \tau_{\max}]} R_i(\tau)$ là đỉnh cực đại tự tương quan.

Phương trình chuyển trạng thái:

$$
S_i = \begin{cases} 
1, & \text{khi } R_i^* \ge T_{\text{high}} \\
1, & \text{khi } S_{i-1} = 1 \text{ và } R_i^* \ge T_{\text{low}} \\
0, & \text{ngược lại}
\end{cases}
$$

**Thông số áp dụng:**  
* Mô hình Baseline: $T_{\text{high}} = 0.4408, T_{\text{low}} = 0.40$
* Mô hình Bandpass: $T_{\text{high}} = 0.4892, T_{\text{low}} = 0.42$

---

### I.6.2. Plugin 2: Mở rộng vùng hữu thanh theo năng lượng khung biên (STE)

**Vấn đề giải quyết:**  
Khắc phục tình trạng mất 1–2 khung ở mép âm tiết khi thanh môn bắt đầu khép hoặc hạ giọng.

**Công thức toán học:**  
Năng lượng ngắn hạn (Short-Time Energy - STE) của khung thứ $i$:

$$
E_i = \sum_{n=0}^{N-1} x_i^2[n]
$$

Với đoạn hữu thanh liên tục $C_k = [i_{\text{start}}, i_{\text{end}}]$, năng lượng đỉnh trong phân đoạn là:

$$
E_{\max}^{(k)} = \max_{j \in C_k} E_j
$$

Khung biên lân cận ($i_{\text{start}} - 1$ hoặc $i_{\text{end}} + 1$) được mở rộng thành Voiced nếu thỏa mãn đồng thời:

$$
\begin{cases} 
E_{\text{boundary}} \ge \alpha \cdot E_{\max}^{(k)} & (\alpha = 0.20) \\
R_{\text{boundary}}^* \ge (1 - \beta) \cdot T & (\beta = 0.20) 
\end{cases}
$$

---

### I.6.3. Plugin 3: Bộ lọc thông dải Butterworth & Lọc Zero-Phase

**Vấn đề giải quyết:**  
Triệt tiêu trôi DC, tiếng ù gió micro ($< 70\text{ Hz}$) và các sóng hài formant bậc cao ($> 900\text{ Hz}$) gây lỗi Pitch Doubling.

**Hàm truyền biên độ bình phương:**  
Bộ lọc thông dải Butterworth bậc $M = 2$ với dải thông $[f_L, f_H] = [70, 900]\text{ Hz}$:

$$
|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega^2 - \omega_L \omega_H}{\omega (\omega_H - \omega_L)}\right)^{4}}
$$

**Lọc hai chiều không lệch pha (Zero-Phase Filtering):**  
Sử dụng kỹ thuật `filtfilt` để triệt tiêu hoàn toàn độ trễ pha:

$$
Y(e^{j\omega}) = |H(e^{j\omega})|^2 X(e^{j\omega}) \implies \Theta(\omega) = 0
$$

Độ lệch pha bằng 0 tuyệt đối tại mọi tần số, bảo toàn nguyên vẹn vị trí thời gian của các đỉnh xung thanh môn $T_0$.

**Đồng bộ phân bố:**  
Do lọc làm sạch tín hiệu, $\mu_V$ tăng từ $0.6166 \rightarrow 0.6490$, hệ thống tự động đồng bộ ngưỡng tối ưu mới:

$$
T = 0.4892
$$

---

### I.6.4. Plugin 4: Cắt gọt trung tâm (Center Clipping - Sondhi 1968)

**Vấn đề giải quyết:**  
Triệt tiêu hoàn toàn thành phần dao động cộng hưởng của thanh đạo (Formant $F_1, F_2$) gây biến dạng đỉnh tự tương quan và sinh ra các đỉnh phụ lồi lõm (nguyên nhân chính gây lỗi nhảy quãng tám Pitch Doubling).

**Nguyên lý làm phẳng phổ (Spectral Flattening):**  
Với mỗi khung tín hiệu $x[n]$, xác định mức cắt $C_L = \eta \cdot \max_{n} |x[n]|$ (thực nghiệm tối ưu $\eta = 0.40$). Biến đổi cắt gọt Sondhi chuẩn:

$$
y[n] = \begin{cases} 
x[n] - C_L & \text{nếu } x[n] > C_L \\ 
x[n] + C_L & \text{nếu } x[n] < -C_L \\ 
0 & \text{nếu } |x[n]| \le C_L 
\end{cases}
$$

**Đồng bộ phân bố ngưỡng tối ưu Gauss:**  
Khi các mẫu biên độ nhỏ bị gán về $0$, phân bố tương quan thay đổi:
* Nhóm Center Clip độc lập: $\mu_V = 0.5188, \mu_U = 0.1759 \implies T = 0.3352$
* Nhóm kết hợp Bandpass + Center Clip: $\mu_V = 0.5801, \mu_U = 0.2230 \implies T = 0.3953$

---

### I.6.5. Plugin 5: Quy hoạch động Viterbi Tracking (Dynamic Programming Pitch Tracking)

**Vấn đề giải quyết:**  
Các thuật toán dò cực trị cục bộ (ACF, AMDF) chỉ xét từng khung độc lập nên dễ mắc lỗi **nhảy quãng tám (Pitch Doubling / Halving)** hoặc bắt nhầm đỉnh formant do sóng hài phụ khi biên độ của đỉnh sai cao hơn đỉnh thật chỉ $0.02 - 0.05$. Bộ lọc trung vị (Median Filter) chỉ sửa được các lỗi 1 khung đơn lẻ, bất lực nếu lỗi kéo dài 2-3 khung liên tiếp.

**Cơ chế sinh lý học & Không gian trạng thái Trellis:**  
Dây thanh âm là cơ quan cơ học sinh học có quán tính, không thể biến thiên tần số đột ngột trong $10\text{ ms}$. Tại mỗi khung hữu thanh, plugin trích xuất Top-5 cực trị tốt nhất ($K=5$) để xây dựng lưới không gian trạng thái.

**Hàm chi phí Trellis (Cost Function):**

* **Chi phí cục bộ ($C_{\text{local}}$):** Đo lường độ tin cậy của ứng viên tại khung $t$:

$$
C_{\text{local}}(\tau) = 1.0 - R_{\text{norm}}(\tau) \quad (\text{với ACF})
$$

* **Chi phí chuyển tiếp ($C_{\text{trans}}$):** Phạt bước nhảy tần số theo thang Logarithm cơ số 2 (Octave):

$$
C_{\text{trans}}(s_{t-1}, s_t) = w_{\text{freq}} \cdot \left( \log_2(F_{0, t}) - \log_2(F_{0, t-1}) \right)^2
$$

Nếu bước nhảy rơi vào vùng nhảy quãng tám ($[0.8, 1.2]\text{ octave}$), áp dụng mức phạt bổ sung $w_{\text{octave}} = 2.0$.

**Thuật toán Viterbi:** Lan truyền tiến tìm đường đi có tổng chi phí nhỏ nhất và truy vết ngược (Backtracking) để thu được chuỗi cao độ tối ưu toàn cục.

---

## I.7. Khảo Sát & Phân Nhóm Cấu Hình Tối Ưu Cho ACF Trên Tập Huấn Luyện (TinHieuHuanLuyen)

Toàn bộ quá trình thử nghiệm, tinh chỉnh siêu tham số và lựa chọn cấu hình tối ưu được thực hiện nghiêm ngặt trên **Tập Huấn Luyện (TinHieuHuanLuyen)** gồm 4 file chuẩn (phone_F1, phone_M1, studio_F1, studio_M1). Hệ thống quét qua 16 tổ hợp plugin (Bandpass Filter, Center Clipping, Energy Extension, Viterbi Tracking) và phân nhóm thành 4 nhóm mục tiêu:

### Bảng kết quả phân nhóm tối ưu cho thuật toán ACF (Tập Huấn Luyện):

| Phân nhóm chỉ số | Cấu hình Plugin | $\Delta F_0$ (Hz) | MAPE $ (%) | $\Delta\text{Std}$ (Hz) | MAPE Std (%) | $\Delta\text{Num}$ (Khung) | MAPE Num (%) | FINAL SCORE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Gốc)** | Baseline | 2.84 Hz | 1.95% | 4.55 Hz | 25.21% | 19.2 khung | 12.53% | **13.23%** |
| **Nhóm 1 (Top 1 F0)** | Energy + Viterbi | **1.08 Hz** | **0.54%** | 1.54 Hz | 7.63% | 8.5 khung | 6.01% | 4.73% |
| **Nhóm 1 (Top 2 F0)** | BP + Energy + Viterbi | 1.34 Hz | 0.64% | **0.66 Hz** | **2.39%** | **8.2 khung** | **5.83%** | **2.95%** 🏆 |
| **Nhóm 1 (Top 3 F0)** | Clip + Energy | 1.48 Hz | 0.82% | 1.23 Hz | 4.21% | 10.8 khung | 7.23% | 4.09% |
| **Nhóm 2 (Top 1 Std)** | BP + Energy + Viterbi | 1.34 Hz | 0.64% | **0.66 Hz** | **2.39%** | **8.2 khung** | **5.83%** | **2.95%** 🏆 |
| **Nhóm 2 (Top 2 Std)** | Clip + Energy | 1.48 Hz | 0.82% | 1.23 Hz | 4.21% | 10.8 khung | 7.23% | 4.09% |
| **Nhóm 2 (Top 3 Std)** | Clip + Energy + Viterbi | 1.55 Hz | 0.85% | 1.26 Hz | 4.39% | 10.8 khung | 7.23% | 4.16% |
| **Nhóm 3 (Top 1 Num)** | BP + Energy | 2.91 Hz | 1.92% | 4.24 Hz | 23.74% | **8.2 khung** | **5.83%** | 10.50% |
| **Nhóm 3 (Top 2 Num)** | BP + Energy + Viterbi | 1.34 Hz | 0.64% | **0.66 Hz** | **2.39%** | **8.2 khung** | **5.83%** | **2.95%** 🏆 |
| **Nhóm 3 (Top 3 Num)** | Energy | 2.40 Hz | 1.62% | 4.21 Hz | 23.50% | 8.5 khung | 6.01% | 10.38% |
| **Nhóm 4 (Top 1 Toàn diện)** | BP + Energy + Viterbi | **1.34 Hz** | **0.64%** | **0.66 Hz** | **2.39%** | **8.2 khung** | **5.83%** | **2.95%** 🏆 |
| **Nhóm 4 (Top 2 Toàn diện)** | Clip + Energy | 1.48 Hz | 0.82% | 1.23 Hz | 4.21% | 10.8 khung | 7.23% | 4.09% |
| **Nhóm 4 (Top 3 Toàn diện)** | Clip + Energy + Viterbi | 1.55 Hz | 0.85% | 1.26 Hz | 4.39% | 10.8 khung | 7.23% | 4.16% |

### Bảng đối sánh chi tiết 4 file huấn luyện: ACF Baseline vs ACF Top 1 Champion [BP + Energy + Viterbi]:

| File huấn luyện | Cấu hình | Ref $ | Pred $ | $\lvert\Delta F_0\rvert$ | MAPE $ | Ref Std | Pred Std | $\lvert\Delta\text{Std}\rvert$ | MAPE Std | Ref Num | Pred Num | $\lvert\Delta\text{Num}\rvert$ | MAPE Num | **FINAL SCORE** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| phone_F1.wav | Baseline | 215.60 Hz | 217.71 Hz | 2.11 Hz | 0.98% | 20.60 | 20.51 | 0.09 | 0.43% | 148 | 132 | 16 | 10.81% | **4.07%** |
| | **Champion** | 215.60 Hz | 217.12 Hz | **1.52 Hz** | **0.71%** | 20.60 | 20.96 | 0.36 | 1.75% | 148 | 136 | **12** | **8.11%** | **3.52%** |
| phone_M1.wav | Baseline | 123.70 Hz | 130.91 Hz | 7.21 Hz | 5.83% | 16.80 | 32.20 | 15.40 | 91.65% | 232 | 193 | 39 | 16.81% | **38.10%** |
| | **Champion** | 123.70 Hz | 123.62 Hz | **0.08 Hz** | **0.06%** | 16.80 | 16.94 | **0.14** | **0.83%** | 232 | 221 | **11** | **4.74%** | **1.88%** 🏆 |
| studio_F1.wav| Baseline | 229.60 Hz | 231.39 Hz | 1.79 Hz | 0.78% | 36.80 | 35.44 | 1.36 | 3.69% | 127 | 117 | 10 | 7.87% | **4.11%** |
| | **Champion** | 229.60 Hz | 232.98 Hz | 3.38 Hz | 1.47% | 36.80 | 35.76 | **1.04** | **2.83%** | 127 | 123 | **4** | **3.15%** | **2.48%** 🏆 |
| studio_M1.wav| Baseline | 116.90 Hz | 116.64 Hz | 0.26 Hz | 0.22% | 26.40 | 25.06 | 1.34 | 5.07% | 82 | 70 | 12 | 14.63% | **6.64%** |
| | **Champion** | 116.90 Hz | 117.29 Hz | 0.39 Hz | 0.33% | 26.40 | 25.30 | **1.10** | **4.15%** | 82 | 76 | **6** | **7.32%** | **3.93%** |
| **TRUNG BÌNH** | Baseline | — | — | 2.84 Hz | 1.95% | — | — | 4.55 | 25.21% | — | — | 19.2 | 12.53% | **13.23%** |
| | **Champion** | — | — | **1.34 Hz** | **0.64%** | — | — | **0.66** | **2.39%** | — | — | **8.2** | **5.83%** | **2.95%** 🏆 |

### Biểu đồ cột xếp hạng### Biểu đồ cột xếp hạng toàn bộ 32 cấu hình ACF:
![05_all_combinations_ranking.png](../outputs/figures/05_all_combinations_ranking.png)

### Biểu đồ trực quan đối sánh Baseline vs Cấu hình tối ưu sai số (Cấu hình 29):

| `phone_F2.wav` | `phone_M2.wav` |
| :---: | :---: |
| ![05_compare_plugin_phone_F2.png](../outputs/figures/05_compare_plugin_phone_F2.png) | ![05_compare_plugin_phone_M2.png](../outputs/figures/05_compare_plugin_phone_M2.png) |
| **`studio_F2.wav`** | **`studio_M2.wav`** |
| ![05_compare_plugin_studio_F2.png](../outputs/figures/05_compare_plugin_studio_F2.png) | ![05_compare_plugin_studio_M2.png](../outputs/figures/05_compare_plugin_studio_M2.png) |

---

**Dẫn hướng:** [Trang chủ README](../README.md) | [Chương II: Thuật toán AMDF →](02_amdf_theory_and_experiments.md)
