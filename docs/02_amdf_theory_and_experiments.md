# CHƯƠNG II: THUẬT TOÁN HÀM HIỆU ĐỘ LỚN TRUNG BÌNH (AMDF)

> **Cập nhật 30/10/2026:** Bảng/hình gốc trong chương là kết quả lịch sử. Phần bổ sung cuối chương và [báo cáo đợt 4](STAGE4_RESULTS.md) ghi kết quả hiện hành, giao thức chọn cấu hình trên train và nguyên nhân chênh lệch Cepstrum.


> **Trạng thái sau rà soát 28/10/2026:** Các phân bố Gaussian trong chương là kết quả lịch sử. LAB hiện hành chỉ có thống kê toàn file; các ngưỡng đang được dùng như tham số đã lưu. Xem [các vấn đề cần xử lý](KNOWN_ISSUES.md).

> **Dẫn hướng tài liệu:** [Trang chủ README](../README.md) > **Chương II: Thuật toán AMDF**

---

## II.1. Cơ Sở Lý Thuyết Thuật Toán AMDF

Hàm hiệu độ lớn trung bình (Average Magnitude Difference Function - AMDF) là một thuật toán ước lượng cao độ trên miền thời gian có chi phí tính toán thấp. Thay vì tính tích tương quan như ACF, AMDF đo lường hiệu số biên độ tuyệt đối giữa khung tín hiệu và phiên bản dịch trễ của nó:

$$
D(\tau) = \sum_{n=0}^{N-1-\tau} |x[n] - x[n+\tau]|
$$

Để chuẩn hóa biên độ trong dải $[0.0, 1.0]$ và loại bỏ ảnh hưởng biên độ tín hiệu, hàm AMDF chuẩn hóa được định nghĩa:

$$
D_{\text{norm}}(\tau) = \frac{\sum_{n=0}^{N-1-\tau} |x[n] - x[n+\tau]|}{\sum_{n=0}^{N-1-\tau} (|x[n]| + |x[n+\tau]|)}
$$

* **Tại độ trễ $\tau = 0$:** $x[n] - x[n] = 0 \implies D(0) = 0$.
* **Âm Hữu thanh (Voiced):** Tín hiệu có tính chất gần như tuần hoàn ($x[n] \approx x[n + \tau_0]$). Tại độ trễ $\tau_0 = T_0 \cdot f_s$, các mẫu gần như triệt tiêu lẫn nhau, làm xuất hiện một **đáy cực tiểu địa phương rất sâu (Deep Dip / Valley)**:

$$
D_{\text{norm}}(\tau_0) \approx 0 \ll T
$$

Tần số cơ bản $F_0$ được tính từ vị trí của đáy cực tiểu sâu nhất:

$$
F_0 = \frac{f_s}{\tau_0} \quad (\text{Hz})
$$

* **Âm Vô thanh (Unvoiced):** Do dạng sóng ngẫu nhiên, hiệu số $\lvert x[n] - x[n+\tau] \rvert$ luôn có độ lớn đáng kể ở mọi độ trễ $\tau$. Giá trị hàm AMDF luôn duy trì ở mức cao ($> 0.5$) và dao động hỗn loạn, không có đáy nào lặn sâu dưới ngưỡng phân tách $T$.

---

## II.2. Minh Họa Khung Hữu Thanh vs Vô Thanh bằng AMDF

Thực nghiệm trên file âm thanh `TinHieuHuanLuyen/phone_F1.wav` ($f_s = 16000\text{ Hz}$, khung chuẩn $25\text{ ms} = 400\text{ mẫu}$):

![01_demo_amdf_frames.png](../outputs/figures/01_demo_amdf_frames.png)

### Bảng phân tích chi tiết hai khung đại diện (AMDF):

| Đặc trưng | Khung Hữu Thanh (Voiced) | Khung Vô Thanh (Unvoiced) |
| :--- | :--- | :--- |
| **Vị trí khung** | Khung #103 ($t = 1.042\text{ s}$) | Khung #176 ($t = 1.772\text{ s}$) |
| **Đặc tính dạng sóng** | Tuần hoàn rõ rệt, biên độ lớn | Ngẫu nhiên tựa nhiễu trắng, biên độ nhỏ |
| **Độ trễ cực tiểu $\tau_0$** | $\tau_0 = 71$ mẫu | Đáy cục bộ trôi nổi tại $\tau = 52$ |
| **Biên độ cực tiểu $D(\tau_0)$** | **$0.075$** ($\ll T = 0.4380$) | **$0.542$** ($> T = 0.4380$) |
| **Chu kỳ cơ bản $T_0$** | $T_0 = 71 / 16000 = 4.44\text{ ms}$ | Không xác định |
| **Tần số cơ bản $F_0$** | **$F_0 = 225.35\text{ Hz}$** (minh họa khung; LAB hiện hành chỉ có mean toàn file) | **$F_0 = \text{Undefined}$** (Vô thanh) |

---

## II.3. Huấn Luyện Ngưỡng T_AMDF Tối Ưu Bằng Phân Bố Gauss

Đối với AMDF, khung Voiced có biên độ đáy cực tiểu **nhỏ** ($\mu_V < \mu_U$), ngược lại với ACF. Trích xuất trên toàn bộ tập `TinHieuHuanLuyen/`:
* Tập khung Hữu thanh: $N_V = 614$ khung $\rightarrow \mathcal{N}(\mu_V, \sigma_V^2)$
* Tập khung Vô thanh: $N_U = 180$ khung $\rightarrow \mathcal{N}(\mu_U, \sigma_U^2)$

### Bảng kết quả huấn luyện ngưỡng AMDF theo các điều kiện tiền xử lý (Khung 25 ms):

| Chế độ tiền xử lý | Số khung Voiced | $\mu_V \pm \sigma_V$ (Đáy AMDF) | Số khung Unvoiced | $\mu_U \pm \sigma_U$ (Đáy AMDF) | Ngưỡng tối ưu $T$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Chưa xử lý)** | 614 | $0.2382 \pm 0.1486$ | 180 | $0.6121 \pm 0.1137$ | **$T = 0.4380$** |
| **Có lọc Bandpass** | 614 | $0.1872 \pm 0.1428$ | 180 | $0.5455 \pm 0.1207$ | **$T = 0.3734$** |
| **Cắt gọt Center Clip ($\eta=0.4$)** | 614 | $0.4040 \pm 0.2449$ | 180 | $0.8646 \pm 0.1570$ | **$T = 0.6488$** |
| **Bandpass + Center Clip** | 614 | $0.3054 \pm 0.2405$ | 180 | $0.7953 \pm 0.1733$ | **$T = 0.5628$** |

### Biểu đồ phân bố xác suất Gaussian AMDF:

| Baseline AMDF ($T = 0.4380$) | Có lọc Bandpass AMDF ($T = 0.3734$) |
| :---: | :---: |
| ![02_threshold_distribution_amdf.png](../outputs/figures/02_threshold_distribution_amdf.png) | ![02_threshold_distribution_amdf_bandpassprefilter.png](../outputs/figures/02_threshold_distribution_amdf_bandpassprefilter.png) |

---

## II.4. Đánh giá trên tập kiểm thử

Kết quả test được trình bày tập trung tại [Chương IV](04_test_set_benchmark_and_analysis.md). Bảng baseline và các hình từ lần chạy cũ được giữ trong [archive](../archive/2026-10-28-pre-cleanup/README.md).

LAB hiện hành chỉ có F0mean/F0std/F0num nên chưa tính được Accuracy/F1 theo từng khung.

---

## II.5. Các Giải Pháp Cải Tiến Dành Riêng Cho AMDF (Plugins)

Do hàm AMDF dựa trên chuẩn khoảng cách $L_1$ và tìm **cực tiểu đáy (Deep Dip)** thay vì tìm **cực đại đỉnh (Peak)** như ACF, toàn bộ 5 Plugins được tái cấu trúc và thích ứng theo logic đối ngẫu toán học:

### II.5.1. Plugin 1: Ngưỡng trễ kép Schmitt Trigger thích ứng AMDF (Hysteresis)

**Nguyên lý đối ngẫu:**  
Đối với AMDF, khung Voiced có giá trị đáy $D_{\text{norm}}(\tau_0)$ **thấp hơn** ngưỡng phân tách ($D \le T$). Do đó, logic ngưỡng trễ kép được đảo ngược hoàn toàn so với ACF để chống hiện tượng rung lật trạng thái:

* **Ngưỡng kích hoạt vào Voiced ($T_{\text{enter}}$ - Ngưỡng nghiêm ngặt):** Để chuyển từ Unvoiced sang Voiced, đáy AMDF phải đủ sâu, lặn xuống dưới mức ngưỡng thấp: $T_{\text{enter}} = \min(T_{\text{high}}, T_{\text{low}}) = T_{\text{low}}$.
* **Ngưỡng duy trì trạng thái Voiced ($T_{\text{exit}}$ - Ngưỡng nới lỏng):** Khi đã ở trạng thái Voiced, hệ thống cho phép đáy AMDF trôi nông lên tới mức ngưỡng cao hơn trước khi bị ngắt về Unvoiced: $T_{\text{exit}} = \max(T_{\text{high}}, T_{\text{low}}) = T_{\text{high}}$.

**Mô hình máy trạng thái:**  
Với $D_i^* = \min_{\tau \in [\tau_{\min}, \tau_{\max}]} D_{\text{norm}, i}(\tau)$ là độ sâu cực tiểu của khung $i$:

$$
S_i = \begin{cases} 
1, & \text{khi } D_i^* \le T_{\text{enter}} \\
1, & \text{khi } S_{i-1} = 1 \text{ và } D_i^* \le T_{\text{exit}} \text{ và } \lvert F_{0, i} - F_{0, i-1} \rvert \le 40\text{ Hz} \\
0, & \text{ngược lại}
\end{cases}
$$

* Mô hình Baseline AMDF ($T = 0.4380$): $T_{\text{enter}} = 0.40, T_{\text{exit}} = 0.48$.
* Mô hình Bandpass AMDF ($T = 0.3734$): $T_{\text{enter}} = 0.35, T_{\text{exit}} = 0.42$.

---

### II.5.2. Plugin 2: Mở rộng vùng hữu thanh theo năng lượng khung biên (STE Energy Extension)

**Vấn đề giải quyết:**  
Tại ranh giới mở/đóng thanh môn, biên độ rung giảm làm đáy AMDF bị nông hóa ($D_i^* > T$), khiến Baseline AMDF cắt cụt mất các khung biên của âm tiết.

**Cơ chế bù đắp biên cho AMDF:**  
Quét các khung biên lân cận ($i_{\text{start}} - 1$ hoặc $i_{\text{end}} + 1$) của mỗi phân đoạn hữu thanh liên tục. Khung được mở rộng thành Voiced nếu thỏa mãn đồng thời:
1. **Năng lượng còn đủ lớn:** $E_{\text{boundary}} \ge \alpha \cdot E_{\max}^{(k)}$ (với $\alpha = 0.20$ và $\text{STE} \ge 1.5 \cdot \text{STE}_{\text{thresh}}$).
2. **Đáy AMDF còn tính tuần hoàn (Ngưỡng nới lỏng):** Do đáy bị nông hóa, ngưỡng chấp nhận được tăng thêm $\beta = 20\%$:

$$
D_{\text{boundary}}^* \le (1 + \beta) \cdot T \quad (\beta = 0.20)
$$

3. **Tính liên tục cao độ:** Sai lệch pitch giữa khung biên và khung liền kề $\le 40\text{ Hz}$.

---

### II.5.3. Plugin 3: Bộ lọc thông dải Butterworth bậc 2 & Lọc Zero-Phase

**Vấn đề giải quyết:**  
Nhiễu tần số cực thấp ($< 70\text{ Hz}$) làm trôi baseline của tín hiệu khiến phép trừ $\lvert x[n] - x[n+\tau] \rvert$ không triệt tiêu về $0$. Đồng thời các sóng hài bậc cao ($> 900\text{ Hz}$) tạo ra nhiều gợn răng cưa cục bộ lấp đầy đáy AMDF.

**Hiệu ứng trên hàm AMDF:**  
* Bộ lọc dải thông $[70, 900]\text{ Hz}$ hai chiều `filtfilt` ($\Theta(\omega) = 0$) làm sạch dạng sóng, giúp các xung thanh môn lặp lại trùng khít biên độ, làm cho **đáy cực tiểu AMDF lặn sâu hơn đáng kể**:
  * Giá trị đáy trung bình khung Voiced giảm mạnh: $\mu_V = 0.2382 \rightarrow 0.1872$ (đáy sâu hơn $21.4\%$).
  * Giá trị đáy trung bình khung Unvoiced cũng giảm: $\mu_U = 0.6121 \rightarrow 0.5455$.
* **Đồng bộ phân bố ngưỡng tối ưu Gauss:** $T = 0.3734$

---

### II.5.4. Plugin 4: Cắt gọt trung tâm (Center Clipping - Sondhi 1968)

**Vấn đề giải quyết:**  
Cộng hưởng thanh đạo (Formant $F_1, F_2$) tạo ra các đỉnh/đáy dao động phụ tuần hoàn giả bên trong mỗi chu kỳ cơ bản, khiến hàm AMDF xuất hiện các đáy cực tiểu giả nông hơn ở các độ trễ ngắn (nguyên nhân gây nhân đôi pitch $2F_0$).

**Cơ chế làm phẳng phổ (Spectral Flattening):**  
Áp dụng ngưỡng cắt $C_L = 0.40 \cdot \max_n \lvert x[n] \rvert$, gán toàn bộ các mẫu có biên độ nhỏ $\le C_L$ về $0$:
* Khi hai đoạn tín hiệu cùng bằng $0$, hiệu số $\lvert x[n] - x[n+\tau] \rvert = 0$ hoàn toàn, triệt tiêu toàn bộ dao động gợn sóng formant.
* Đáy cực tiểu tại chu kỳ thật $\tau_0$ trở nên siêu sắc nét, trong khi các đáy phụ biến mất.
* **Đồng bộ phân bố ngưỡng tối ưu Gauss:**
  * Nhóm Center Clip độc lập: $\mu_V = 0.4040, \mu_U = 0.8646 \implies T = 0.6488$
  * Nhóm kết hợp Bandpass + Center Clip: $\mu_V = 0.3054, \mu_U = 0.7953 \implies T = 0.5628$

---

### II.5.5. Plugin 5: Quy hoạch động Viterbi Tracking trên lưới cực tiểu AMDF

**Không gian trạng thái Trellis cho AMDF:**  
Tại mỗi khung hữu thanh, plugin quét tìm Top-$K$ ($K=5$) đáy cực tiểu địa phương sâu nhất ($D(\tau) < D(\tau-1)$ và $D(\tau) \le D(\tau+1)$) trong dải cao độ $[70, 400]\text{ Hz}$.

**Hàm chi phí Trellis (Cost Function) thiết kế riêng cho AMDF:**
* **Chi phí cục bộ ($C_{\text{local}}$):** Vì hàm $D_{\text{norm}}(\tau) \in [0, 1]$ biểu thị sai số khác biệt (đáy càng sâu giá trị càng nhỏ, càng đáng tin cậy), hàm chi phí cục bộ được định nghĩa trực tiếp bằng chính giá trị AMDF:

$$
C_{\text{local}}(\tau) = D_{\text{norm}}(\tau) \quad (\text{đối ngẫu với ACF: } 1.0 - R_{\text{norm}})
$$

* **Chi phí chuyển tiếp ($C_{\text{trans}}$):** Phạt bước nhảy tần số giữa hai khung liên tiếp theo thang Octave:

$$
C_{\text{trans}}(s_{t-1}, s_t) = w_{\text{freq}} \cdot \left( \log_2(F_{0, t}) - \log_2(F_{0, t-1}) \right)^2
$$

Nếu bước nhảy rơi vào vùng nhảy quãng tám ($[0.8, 1.2]\text{ octave}$ hoặc $[1.8, 2.2]\text{ octave}$), áp dụng mức phạt bổ sung $w_{\text{octave}} = 2.0$.

* **Truy vết tối ưu toàn cục:** Thuật toán Viterbi tìm đường đi qua các đáy AMDF có tổng chi phí nhỏ nhất trên toàn phân đoạn, loại bỏ hoàn toàn hiện tượng bắt nhầm đáy giả hoặc nhảy quãng tám.

---

## II.6. Khảo Sát & Phân Nhóm Cấu Hình Tối Ưu Cho AMDF Trên Tập Huấn Luyện (TinHieuHuanLuyen)

Toàn bộ quá trình thực nghiệm, khảo sát 16 tổ hợp plugin và lựa chọn các cấu hình tiêu biểu được tiến hành trên **Tập Huấn Luyện (TinHieuHuanLuyen)** gồm 4 file chuẩn (phone_F1, phone_M1, studio_F1, studio_M1). Dưới đây là bảng phân loại thành 4 nhóm tối ưu đại diện cho 3 chỉ số mục tiêu cốt lõi và điểm tổng hợp toàn diện:

### Bảng kết quả phân nhóm tối ưu cho thuật toán AMDF (Tập Huấn Luyện):

| Phân nhóm chỉ số | Cấu hình Plugin | $\Delta F_0$ (Hz) | MAPE $ (%) | $\Delta\text{Std}$ (Hz) | MAPE Std (%) | $\Delta\text{Num}$ (Khung) | MAPE Num (%) | FINAL SCORE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Gốc)** | Baseline | 1.49 Hz | 0.93% | 2.31 Hz | 11.34% | 14.8 khung | 8.77% | **7.01%** |
| **Nhóm 1 (Top 1 F0)** | BP + Energy + Viterbi | **0.86 Hz** | **0.48%** | 1.02 Hz | 3.81% | 11.8 khung | 7.64% | 3.98% |
| **Nhóm 1 (Top 2 F0)** | BP + Viterbi | 1.11 Hz | 0.67% | 1.24 Hz | 4.78% | 17.2 khung | 11.39% | 5.61% |
| **Nhóm 1 (Top 3 F0)** | BP + Energy | 1.11 Hz | 0.68% | 2.16 Hz | 10.53% | 11.8 khung | 7.64% | 6.29% |
| **Nhóm 2 (Top 1 Std)** | Energy + Viterbi | 1.30 Hz | 0.80% | **0.96 Hz** | **3.42%** | **9.2 khung** | **5.64%** | **3.29%** 🏆 |
| **Nhóm 2 (Top 2 Std)** | BP + Energy + Viterbi | **0.86 Hz** | **0.48%** | 1.02 Hz | 3.81% | 11.8 khung | 7.64% | 3.98% |
| **Nhóm 2 (Top 3 Std)** | Viterbi | 1.46 Hz | 0.92% | 1.16 Hz | 4.37% | 14.8 khung | 8.77% | 4.69% |
| **Nhóm 3 (Top 1 Num)** | BP + Clip + Energy | 1.90 Hz | 1.16% | 1.79 Hz | 7.29% | **7.8 khung** | **4.93%** | 4.46% |
| **Nhóm 3 (Top 2 Num)** | BP + Clip + Energy + Viterbi | 1.85 Hz | 1.14% | 1.79 Hz | 7.29% | **7.8 khung** | **4.93%** | 4.45% |
| **Nhóm 3 (Top 3 Num)** | Energy | 1.27 Hz | 0.76% | 2.04 Hz | 10.00% | 9.2 khung | 5.64% | 5.47% |
| **Nhóm 4 (Top 1 Toàn diện)** | Energy + Viterbi | **1.30 Hz** | **0.80%** | **0.96 Hz** | **3.42%** | **9.2 khung** | **5.64%** | **3.29%** 🏆 |
| **Nhóm 4 (Top 2 Toàn diện)** | BP + Energy + Viterbi | 0.86 Hz | 0.48% | 1.02 Hz | 3.81% | 11.8 khung | 7.64% | 3.98% |
| **Nhóm 4 (Top 3 Toàn diện)** | Clip + Energy + Viterbi | 2.01 Hz | 1.22% | 1.37 Hz | 5.05% | 10.8 khung | 6.75% | 4.34% |

### Bảng đối sánh chi tiết 4 file huấn luyện: AMDF Baseline vs AMDF Top 1 Champion [Energy + Viterbi]:

| File huấn luyện | Cấu hình | Ref $ | Pred $ | $\lvert\Delta F_0\rvert$ | MAPE $ | Ref Std | Pred Std | $\lvert\Delta\text{Std}\rvert$ | MAPE Std | Ref Num | Pred Num | $\lvert\Delta\text{Num}\rvert$ | MAPE Num | **FINAL SCORE** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| phone_F1.wav | Baseline | 215.60 Hz | 216.65 Hz | 1.05 Hz | 0.49% | 20.60 | 20.87 | 0.27 | 1.30% | 148 | 134 | 14 | 9.46% | **3.75%** |
| | **Champion** | 215.60 Hz | 215.88 Hz | **0.28 Hz** | **0.13%** | 20.60 | 20.92 | **0.32** | **1.55%** | 148 | 137 | **11** | **7.43%** | **3.04%** 🏆 |
| phone_M1.wav | Baseline | 123.70 Hz | 125.07 Hz | 1.37 Hz | 1.11% | 16.80 | 22.27 | 5.47 | 32.55% | 232 | 202 | 30 | 12.93% | **15.53%** |
| | **Champion** | 123.70 Hz | 122.59 Hz | **1.11 Hz** | **0.90%** | 16.80 | 17.14 | **0.34** | **2.00%** | 232 | 217 | **15** | **6.47%** | **3.12%** 🏆 |
| studio_F1.wav| Baseline | 229.60 Hz | 231.76 Hz | 2.16 Hz | 0.94% | 36.80 | 35.18 | 1.62 | 4.41% | 127 | 114 | 13 | 10.24% | **5.20%** |
| | **Champion** | 229.60 Hz | 232.18 Hz | 2.58 Hz | 1.12% | 36.80 | 35.08 | 1.72 | 4.68% | 127 | 116 | **11** | **8.66%** | **4.82%** 🏆 |
| studio_M1.wav| Baseline | 116.90 Hz | 115.51 Hz | 1.39 Hz | 1.19% | 26.40 | 24.53 | 1.87 | 7.09% | 82 | 80 | 2 | 2.44% | **3.57%** |
| | **Champion** | 116.90 Hz | 115.67 Hz | **1.23 Hz** | **1.05%** | 26.40 | 24.96 | **1.44** | **5.45%** | 82 | 82 | **0** | **0.00%** | **2.17%** 🏆 |
| **TRUNG BÌNH** | Baseline | — | — | 1.49 Hz | 0.93% | — | — | 2.31 | 11.34% | — | — | 14.8 | 8.77% | **7.01%** |
| | **Champion** | — | — | **1.30 Hz** | **0.80%** | — | — | **0.95** | **3.42%** | — | — | **9.2** | **5.64%** | **3.29%** 🏆 |

### Biểu đồ các cấu hình đại diện từ khảo sát 16 tổ hợp AMDF:
![05_all_combinations_ranking_amdf.png](../outputs/figures/05_all_combinations_ranking_amdf.png)

Các hình và bảng đối sánh dùng cấu hình/nhãn cũ đã chuyển vào [archive](../archive/2026-10-28-pre-cleanup/README.md). Đối sánh ACF–AMDF hiện tại xem [Chương IV](04_test_set_benchmark_and_analysis.md).

---

**Dẫn hướng:** [← Chương I: Thuật toán ACF](01_acf_theory_and_experiments.md) | [Trang chủ README](../README.md) | [Chương III: Thuật toán Cepstrum →](03_cepstrum_theory_and_experiments.md)

<!-- STAGE4_CURRENT_START -->
## Kết quả kiểm chứng hiện hành — 30/10/2026

Nguồn: cấu hình đóng băng và `outputs/stage4_validation/evaluation.json`. Final Score là sai số thống kê toàn file; thấp hơn tốt hơn. Accuracy/F1 từng khung không xác định vì thiếu nhãn thời gian.

Preset: Energy + Viterbi, frame/hop 30/10 ms, T=0.5.

| Cấu hình | Train (%) | Test (%) |
|---|---:|---:|
| baseline | 6.760 | 6.357 |
| historical_champion | 3.757 | 4.125 |
| selected_champion | 2.078 | 2.755 |

| File test | Mean dự đoán (Hz) | Std dự đoán (Hz) | Khung voiced | Final Score (%) |
|---|---:|---:|---:|---:|
| phone_F2.wav | 150.07 | 30.86 | 201 | 2.94 |
| phone_M2.wav | 130.07 | 16.23 | 129 | 3.45 |
| studio_F2.wav | 199.65 | 43.92 | 134 | 4.08 |
| studio_M2.wav | 155.62 | 30.62 | 116 | 0.55 |

[Chi tiết và giới hạn kiểm chứng](STAGE4_RESULTS.md).
<!-- STAGE4_CURRENT_END -->
