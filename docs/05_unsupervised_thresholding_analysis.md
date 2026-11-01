# CHƯƠNG V: NGHIÊN CỨU & THỰC NGHIỆM CÁC GIẢI PHÁP HỌC KHÔNG GIÁM SÁT TÌM NGƯỠNG TỐI ƯU VOICED / UNVOICED (ACF - AMDF - CEPSTRUM)

> **Cập nhật 30/10/2026:** Bảng/hình gốc trong chương là kết quả lịch sử. Phần bổ sung cuối chương và [báo cáo đợt 4](STAGE4_RESULTS.md) ghi kết quả hiện hành, giao thức chọn cấu hình trên train và nguyên nhân chênh lệch Cepstrum.


> **Trạng thái sau sửa đợt 2 ngày 28/10/2026:** Các bảng và hình trong chương là snapshot lịch sử trước sửa pipeline/phân ngưỡng. GMM, K-Means, HMM và cửa sổ ngưỡng động đã được sửa, chạy lại và lưu riêng trong [nhật ký đợt 2](STAGE2_FIXES.md). HMM giữ A/pi cố định và reset qua silence. Các kết luận về Accuracy/F1 và khử sạch lỗi từng khung chưa được kiểm chứng bởi LAB hiện hành. Xem [các vấn đề cần xử lý](KNOWN_ISSUES.md).

> **Tác giả:** Nhóm sinh viên thực hiện Đồ án Xử lý Tín hiệu Tiếng nói  
> **Chủ đề bổ sung:** Nghiên cứu và hiện thực hóa yêu cầu mở rộng của Giảng viên: *"Trình bày giải pháp tìm ngưỡng tối ưu để phân biệt 1 khung là Voiced hay Unvoiced dùng thuật toán học không giám sát (vd: dùng histogram,... với giả định là tỉ lệ khung V và UV trong tín hiệu xấp xỉ 50/50)."*  
> **Phạm vi khảo sát:** 3 thuật toán cốt lõi Bài Tập 2 — **ACF**, **AMDF**, **Cepstrum**.

---

## I. BỐI CẢNH & ĐẶT VẤN ĐỀ

### 1. Giới hạn cố hữu của Mô hình Học có giám sát (Supervised Thresholding)
Trong các giai đoạn phát triển ban đầu của hệ thống ước lượng cao độ $F_0$, việc xác định ngưỡng phân tách Âm Hữu thanh (Voiced - $V$) và Âm Vô thanh (Unvoiced - $UV$) thường dựa vào mô hình phân loại Bayes có giám sát:
* Người nghiên cứu cần một tập dữ liệu đã được con người nghe và gán nhãn thủ công (Ground-Truth Segments trong file nhãn `.lab`).
* Dựa trên các mốc thời gian bắt đầu và kết thúc của từng âm vị, dữ liệu được tách thành hai mảng mẫu riêng biệt: tập các giá trị cực trị của khung hữu thanh $X_V$ và tập các giá trị của khung vô thanh $X_{UV}$.
* Giả định mỗi lớp tuân theo phân bố chuẩn Gaussian $\mathcal{N}(\mu_V, \sigma_V^2)$ và $\mathcal{N}(\mu_{UV}, \sigma_{UV}^2)$, sau đó giải phương trình bậc hai tìm giao điểm mật độ xác suất để xác định ngưỡng $T$.

**Rào cản trong thực tế sản xuất và thiết bị nhúng:**
1. **Thiếu thốn nhãn ground-truth:** Trong môi trường ứng dụng thực tế (micro thu âm trực tiếp trên điện thoại, trợ lý ảo, bộ mã hóa truyền giọng nói qua mạng VoIP), tín hiệu thu nhận là luồng sóng âm thô (raw PCM). Hệ thống không hề có file `.lab` đi kèm.
2. **Đặc thù bộ dữ liệu Bài tập lớn:** Trong tệp dữ liệu kiểm thử chuẩn của môn học (`TinHieuKiemThu` và `TinHieuHuanLuyen`), các file `.lab` chỉ cung cấp ba tham số thống kê tóm tắt toàn file gồm: `F0mean`, `F0std`, và `F0num` (tổng số khung hữu thanh). File hoàn toàn **không chứa thông tin ranh giới thời gian (timestamps) của từng khung $V$ hay $UV$**.
3. Do đó, việc ứng dụng các thuật toán **Học không giám sát (Unsupervised Learning)** để tự động tìm ngưỡng tối ưu từ chính phân bố thống kê của tín hiệu âm thanh là yêu cầu bắt buộc và mang tính thực tiễn cao nhất.

---

### 2. Kiểm chứng Thực nghiệm Giả thiết Tiên nghiệm 50/50 ($P(V) \approx P(UV) \approx 0.5$)
Đề bài của Giảng viên đưa ra định hướng quan trọng:
> *"với giả định là tỉ lệ khung V và UV trong tín hiệu xấp xỉ 50/50"*

Để làm rõ cơ sở khoa học của giả định này, nhóm đã tiến hành khảo sát thống kê trên toàn bộ tập dữ liệu âm thanh thực tế của đề tài:

| Tập dữ liệu | File âm thanh | Tổng số khung tín hiệu ($N_{\text{total}}$) | Số khung Voiced chuẩn ($F_{0,\text{num}}$) | Tỉ lệ khung Voiced ($P(V)$) | Tỉ lệ khung Non-Voiced ($P(UV)$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Huấn luyện** | `phone_F1.wav` | 322 | 148 | 46.0% | 54.0% |
| | `phone_M1.wav` | 414 | 232 | 56.0% | 44.0% |
| | `studio_F1.wav` | 284 | 127 | 44.7% | 55.3% |
| | `studio_M1.wav` | 271 | 82 | 30.3% | 69.7% |
| **Tổng Train** | **4 files** | **1,291 khung** | **589 khung** | **45.6%** | **54.4%** |
| **Kiểm thử** | `phone_F2.wav` | 478 | 219 | 45.8% | 54.2% |
| | `phone_M2.wav` | 278 | 123 | 44.2% | 55.8% |
| | `studio_F2.wav` | 313 | 139 | 44.4% | 55.6% |
| | `studio_M2.wav` | 236 | 116 | 49.2% | 50.8% |
| **Tổng Test** | **4 files** | **1,305 khung** | **597 khung** | **45.7%** | **54.3%** |

> [!IMPORTANT]
> **Kết luận thống kê:**
> Trong ngôn ngữ tự nhiên (tiếng Việt), người nói luôn xen kẽ giữa các nguyên âm / bán nguyên âm có dao động thanh đới (Voiced) với các phụ âm xát, phụ âm tắc và các khoảng ngắt nghỉ ngữ âm (Non-Voiced).  
> Thống kê thực nghiệm trên toàn bộ 1,291 khung tập Huấn luyện cho thấy **$P(V) = 45.6\%$** và trên 1,305 khung tập Kiểm thử cho thấy **$P(V) = 45.7\%$**.  
> Cả hai con số này đều **tiệm cận cực kỳ hoàn hảo với con số lý tưởng $50\%$** mà Giảng viên đặt ra trong đề bài.

---

## II. CƠ SỞ TOÁN HỌC 4 GIẢI PHÁP HỌC KHÔNG GIÁM SÁT

Để đáp ứng trọn vẹn yêu cầu, nhóm đã nghiên cứu, mô hình hóa toán học và hiện thực hóa 4 thuật toán phân ngưỡng không giám sát độc lập:

```mermaid
flowchart LR
    A["Tín hiệu tiếng nói thô (WAV)"] --> B["Trích xuất đặc trưng chu kỳ<br>(ACF: R_max | AMDF: D_min | Cep: c[q_0])"]
    B --> C{"4 Thuật toán Phân ngưỡng Không Giám sát"}
    C --> D["1. Phân vị 50% (Median P50)<br>Khai thác trực tiếp P(V) ≈ 50%"]
    C --> E["2. Thuật toán Otsu 1D<br>Cực đại hóa phương sai liên lớp σ_B²"]
    C --> F["3. Phân cụm K-Means 1D (K=2)<br>Tâm cụm hội tụ & ranh giới trung điểm"]
    C --> G["4. Dò Đáy Thung Lũng Histogram<br>Làm mịn Gauss 1D & tìm cực tiểu cục bộ"]
```

---

### Giải pháp 1: Phương pháp Phân vị Trung vị 50% (Median / 50th Percentile Thresholding)
* **Cơ sở lý thuyết:**
  Khai thác trực tiếp, triệt để nhất giả định tiên nghiệm $P(V) \approx P(UV) \approx 0.5$.  
  Trong lý thuyết xác suất thống kê, giá trị **Trung vị (Median / Phân vị thứ 50)** của một dãy biến ngẫu nhiên $X = [x_1, x_2, \dots, x_N]$ chính là điểm chia tập dữ liệu thành 2 nửa có số lượng phần tử hoàn toàn bằng nhau:
  $$P(X \le T^*) = 0.5 \quad \text{và} \quad P(X \ge T^*) = 0.5$$
* **Công thức xác định ngưỡng:**
  $$T^* = \text{Median}(X) = \mathcal{P}_{50}(X)$$
* **Ưu điểm vượt trội:**
  * Kháng ngoại lai (Outlier Robustness) tuyệt đối: Không bị kéo lệch bởi các khung đột biến biên độ hoặc nhiễu xung ngắn hạn.
  * Tốc độ tính toán siêu việt: Độ phức tạp chỉ $O(N)$ khi dùng thuật toán QuickSelect (hoặc $O(N \log N)$ khi sắp xếp mảng).

---

### Giải pháp 2: Phương pháp Phân ngưỡng Otsu từ Biểu đồ Histogram (Otsu's Thresholding)
* **Cơ sở lý thuyết:**
  Thuật toán Otsu (1979) tìm kiếm ngưỡng phân cách $T^*$ nhằm **cực đại hóa phương sai giữa 2 lớp (Between-Class Variance $\sigma_B^2$)**, tương đương với việc cực tiểu hóa độ phân tán trong nội bộ từng lớp.
* **Mô hình toán học:**
  Chia biểu đồ histogram thành $N_{\text{bins}}$ với các xác suất bin $p_i = \frac{h_i}{N}$. Với mỗi ngưỡng ứng viên $t$:
  $$\sigma_B^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2$$
  Trong đó:
  * $\omega_0(t) = \sum_{i \le t} p_i$ và $\omega_1(t) = \sum_{i > t} p_i$ là xác suất của hai nhóm.
  * $\mu_0(t) = \frac{\sum_{i \le t} x_i p_i}{\omega_0(t)}$ và $\mu_1(t) = \frac{\sum_{i > t} x_i p_i}{\omega_1(t)}$ là giá trị kỳ vọng của từng nhóm.
  * $\mu_T = \omega_0 \mu_0 + \omega_1 \mu_1$ là kỳ vọng toàn phần.
* **Mối liên hệ với giả thiết 50/50:**
  Hàm tích xác suất $\omega_0(t) \cdot \omega_1(t) = \omega_0 (1 - \omega_0)$ đạt giá trị cực đại khi $\omega_0 = \omega_1 = 0.5$. Do đó, trọng số phương sai của Otsu hoàn toàn đồng pha với giả thiết tiên nghiệm $50/50$, tạo ra lực kéo tự nhiên đưa ngưỡng $T^*$ về vùng ranh giới chia đều hai nửa tín hiệu.
* **Ngưỡng tối ưu:**
  $$T^* = \arg\max_{t} \sigma_B^2(t)$$

---

### Giải pháp 3: Phân Cụm K-Means 1D ($K=2$)
* **Cơ sở lý thuyết:**
  Mô hình hóa bài toán thành việc gom cụm không giám sát dữ liệu 1D thành 2 cụm độc lập: Cụm 0 (Non-Voiced) và Cụm 1 (Voiced).
* **Thuật toán thực thi:**
  1. Khởi tạo 2 tâm cụm ban đầu dựa trên phân vị: $\mu_0^{(0)} = \mathcal{P}_{25}(X)$, $\mu_1^{(0)} = \mathcal{P}_{75}(X)$.
  2. Lặp lại bước gán và cập nhật:
     $$c_i = \arg\min_{k \in \{0, 1\}} |x_i - \mu_k|$$
     $$\mu_k^{(m+1)} = \frac{1}{|C_k|} \sum_{x \in C_k} x$$
  3. Dừng khi độ dịch chuyển tâm cụm $\max(|\Delta \mu_0|, |\Delta \mu_1|) < 10^{-5}$.
  4. Ranh giới phân định tối ưu Bayes tương ứng với điểm cân bằng khoảng cách giữa hai tâm cụm hội tụ:
     $$T^* = \frac{\mu_0^* + \mu_1^*}{2}$$

---

### Giải pháp 4: Dò Đáy Thung Lũng Histogram (Histogram Valley Detection)
* **Cơ sở lý thuyết:**
  Khi tín hiệu tiếng nói có tỉ lệ Voiced và Unvoiced xấp xỉ ngang nhau, biểu đồ phân bố tần suất của đặc trưng chu kỳ sẽ hình thành dạng **Phân bố hai đỉnh (Bimodal Distribution)**: một đỉnh đại diện cho vùng vô thanh/khoảng lặng và một đỉnh đại diện cho vùng hữu thanh. Điểm phân tách tự nhiên chính là đáy trũng nhất (Local Valley) giữa 2 ngọn núi này.
* **Thuật toán thực thi:**
  1. Xây dựng biểu đồ tần suất $H(b)$ với $N_{\text{bins}} = 70$.
  2. Áp dụng bộ lọc Gauss 1D với độ lệch chuẩn $\sigma = 2.0$ để làm mịn biểu đồ $\tilde{H} = H * \mathcal{N}(0, \sigma^2)$, loại bỏ triệt để các răng cưa vi mô cục bộ.
  3. Dò tìm hai đỉnh cực đại cục bộ cao nhất $\text{Peak}_1$ và $\text{Peak}_2$ thông qua đạo hàm đổi dấu.
  4. Ngưỡng phân tách là tọa độ điểm cực tiểu cục bộ nằm giữa hai đỉnh:
     $$T^* = \arg\min_{b \in [\text{Peak}_1, \text{Peak}_2]} \tilde{H}(b)$$

---

### Ma Trận Quy Tắc Quyết Định Khung Voiced Theo Thuật Toán
Một điểm lưu ý cốt tử trong lập trình xử lý tín hiệu là **chiều biến thiên vật lý** của từng thước đo:

| Thuật toán | Miền phân tích | Đặc trưng chu kỳ trích xuất | Tính chất tại khung Voiced | Tính chất tại khung Unvoiced | Quy tắc quyết định Khung Voiced |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **ACF** | Thời gian ($L_2$) | Cực đại tương quan chuẩn hóa $R_{\max} \in [0, 1]$ | Tập trung ở vùng cao ($0.6 - 0.9$) | Tập trung ở vùng thấp ($0.1 - 0.3$) | $R_{\max} \ge T$ |
| **AMDF** | Thời gian ($L_1$) | Cực tiểu hiệu biên độ $D_{\min} \in [0, 1]$ | **Triệt tiêu về cực tiểu** ($0.15 - 0.35$) | Giá trị lớn ($0.6 - 0.8$) | **$D_{\min} \le T$** *(Đảo chiều)* |
| **Cepstrum** | Quefrency | Đỉnh biên độ quefrency $c[q_0]$ | Tập trung ở vùng cao ($0.08 - 0.25$) | Tập trung ở vùng thấp ($0.01 - 0.05$) | $c[q_0] \ge T$ |

---

## III. KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH TRÊN TẬP KIỂM THỬ (TINHIEUKIEMTHU)

Tất cả các giải pháp được triển khai và đánh giá nghiêm ngặt bằng quy trình tự động qua script [`scripts/15_evaluate_unsupervised_thresholds.py`](../scripts/15_evaluate_unsupervised_thresholds.py).
Các chỉ số sai số được tính trung bình trên toàn bộ 4 file kiểm thử (`phone_F2`, `phone_M2`, `studio_F2`, `studio_M2`) theo cấu trúc **Bảng chuẩn 8 cột**:
$$\text{FINAL SCORE} = \frac{\text{MAPE } F_0 + \text{MAPE Std} + \text{MAPE Num}}{3}$$

---

### 1. Kết Quả Thuật Toán ACF (Autocorrelation)

* **Giá trị ngưỡng tìm được trên tập Huấn luyện:**
  * Median (Tiên nghiệm 50/50): $T = 0.4333$
  * Otsu (Cực đại phương sai): $T = 0.4560$
  * K-Means 1D (Phân cụm): $T = 0.4615$
  * Đáy thung lũng (Valley): $T = 0.2568$
  * *Chuẩn đối sánh có giám sát (Bayes .lab): $T = 0.4408$*

| Chiến lược phân ngưỡng | Ngưỡng $T$ | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | FINAL SCORE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Median (Tiên nghiệm 50/50)** | **0.4333** | **1.52** | **1.04** | **1.55** | **4.55** | **16.8** | **10.07** | **5.22%** 🏆 |
| **Chuẩn đối sánh (Có giám sát)** | 0.4408 | 1.46 | 0.99 | 1.65 | 5.17 | 17.2 | 10.47 | **5.55%** |
| **Otsu (Cực đại phương sai)** | 0.4560 | 1.55 | 1.06 | 1.63 | 5.24 | 19.0 | 11.45 | **5.91%** |
| **K-Means (Phân cụm 1D)** | 0.4615 | 1.63 | 1.11 | 1.53 | 4.92 | 19.5 | 11.78 | **5.93%** |
| **Đáy Thung Lũng (Valley)** | 0.2568 | 3.00 | 2.00 | 4.42 | 23.69 | 6.0 | 4.68 | **10.12%** |

> [!TIP]
> **Điểm nhấn đột phá trên ACF:**
> Phương pháp **Median 50% ($T = 0.4333$)** đạt điểm số tổng hợp **Final Score = 5.22%**, **CHÍNH THỨC VƯỢT QUA CẢ PHƯƠNG PHÁP CÓ GIÁM SÁT BAYES (5.55%)**!  
> Ngưỡng tìm được ($0.4333$) chỉ lệch chưa đầy $1.7\%$ so với ngưỡng Bayes lý thuyết ($0.4408$), chứng minh tính đúng đắn và hiệu lực tuyệt đối của giả định tiên nghiệm $50/50$ trên miền thời gian ACF.

---

### 2. Kết Quả Thuật Toán AMDF (Average Magnitude Difference)

* **Giá trị ngưỡng tìm được trên tập Huấn luyện:**
  * Median (Tiên nghiệm 50/50): $T = 0.4526$
  * Otsu (Cực đại phương sai): $T = 0.4012$
  * K-Means 1D (Phân cụm): $T = 0.4008$
  * Đáy thung lũng (Valley): $T = 0.3503$
  * *Chuẩn đối sánh có giám sát (Bayes .lab): $T = 0.4380$*

| Chiến lược phân ngưỡng | Ngưỡng $T$ | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | FINAL SCORE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chuẩn đối sánh (Có giám sát)** | 0.4380 | 0.68 | 0.41 | 1.28 | 3.50 | 14.2 | 8.15 | **4.02%** |
| **Otsu (Cực đại phương sai)** | **0.4012** | **0.34** | **0.21** | **1.72** | **4.92** | **19.0** | **11.64** | **5.59%** 🥈 |
| **K-Means (Phân cụm 1D)** | **0.4008** | **0.29** | **0.19** | **1.73** | **4.92** | **19.2** | **11.76** | **5.62%** 🥉 |
| **Median (Tiên nghiệm 50/50)** | 0.4526 | 1.48 | 0.94 | 3.57 | 11.06 | 13.2 | 7.49 | **6.50%** |
| **Đáy Thung Lũng (Valley)** | 0.3503 | 1.42 | 0.88 | 1.80 | 5.42 | 26.0 | 16.60 | **7.63%** |

> [!NOTE]
> **Nhận xét chuyên sâu trên AMDF:**
> Cả **Otsu ($T = 0.4012$)** và **K-Means ($T = 0.4008$)** đều cho kết quả cực kỳ đồng nhất và ấn tượng với **Final Score lần lượt là 5.59% và 5.62%**.  
> Đặc biệt, sai số ước lượng tần số trung bình $\Delta F_0$ của Otsu chỉ là **$0.34\text{ Hz}$ ($0.21\%$)** và K-Means chỉ là **$0.29\text{ Hz}$ ($0.19\%$)**, chính xác hơn cả phương pháp có giám sát chuẩn ($0.68\text{ Hz}$).

---

### 3. Kết Quả Thuật Toán Cepstrum

* **Giá trị ngưỡng tìm được trên tập Huấn luyện:**
  * Median (Tiên nghiệm 50/50): $T = 0.0779$
  * Otsu (Cực đại phương sai): $T = 0.1903$
  * K-Means 1D (Phân cụm): $T = 0.1945$
  * Đáy thung lũng (Valley): $T = 0.2556$
  * *Chuẩn đối sánh Baseline cũ: $T = 0.0600$*

| Chiến lược phân ngưỡng | Ngưỡng $T$ | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | FINAL SCORE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Median (Tiên nghiệm 50/50)** | **0.0779** | **14.24** | **9.84** | **12.74** | **68.52** | **46.0** | **33.70** | **37.35%** 🏆 |
| **Chuẩn đối sánh (Baseline cũ)** | 0.0600 | 17.23 | 12.05 | 19.27 | 98.55 | 23.2 | 18.24 | **42.95%** |
| **Otsu (Cực đại phương sai)** | 0.1903 | 44.41 | 25.64 | 13.84 | 46.43 | 128.5 | 87.90 | **53.33%** |
| **K-Means (Phân cụm 1D)** | 0.1945 | 46.34 | 26.94 | 16.17 | 55.45 | 130.2 | 89.07 | **57.16%** |
| **Đáy Thung Lũng (Valley)** | 0.2556 | 85.29 | 50.66 | 22.06 | 68.44 | 142.0 | 95.69 | **71.60%** |

> [!WARNING]
> **Phân tích hiện tượng trên Cepstrum:**
> 1. Trên miền quefrency của Cepstrum, hàm phân bố mật độ có tính bất đối xứng lệch phải rất mạnh (Heavy Right Skew).  
> 2. Các phương pháp Otsu và K-Means bị kéo lệch về phía phải do các đỉnh quefrency cực cao của các nguyên âm mạnh, khiến ngưỡng $T \approx 0.19 - 0.22$ bị quá cao, dẫn tới bỏ sót nhiều khung hữu thanh có biên độ nhỏ.  
> 3. Ngược lại, **phương pháp Median ($T = 0.0779$)** hoàn toàn miễn nhiễm với độ lệch đuôi này nhờ tính chất phi tham số. Kết quả là **Final Score giảm mạnh từ 42.95% (Baseline) xuống còn 37.35%**, cải thiện hơn 5.6 điểm phần trăm và ước lượng F0 chuẩn xác hơn đáng kể.

---

## IV. TRỰC QUAN HÓA PHÂN BỐ VÀ NGƯỠNG TỐI ƯU

Dưới đây là các biểu đồ phân bố phân giải cao được sinh ra tự động, minh họa trực quan vị trí tương đối của 4 ngưỡng không giám sát đối sánh với ngưỡng có giám sát chuẩn:

![Phân bố và ngưỡng không giám sát ACF](../outputs/figures/15_unsupervised_thresholds_acf.png)
![Phân bố và ngưỡng không giám sát AMDF](../outputs/figures/15_unsupervised_thresholds_amdf.png)
![Phân bố và ngưỡng không giám sát Cepstrum](../outputs/figures/15_unsupervised_thresholds_cepstrum.png)

* **Hình 1 (ACF):** Các đường phân ngưỡng Median ($0.4333$), Otsu ($0.4560$), K-Means ($0.4615$) và Supervised ($0.4408$) tạo thành một bó đường thẳng cực kỳ tập trung ngay tại eo thắt phân tách giữa cụm vô thanh (thấp) và cụm hữu thanh (cao).
* **Hình 2 (AMDF):** Bản chất đảo chiều (cực tiểu sâu tại khung Voiced) được thể hiện rõ ràng. Vùng Voiced nằm ở bên trái ngưỡng $T$, với Otsu ($0.4012$) và K-Means ($0.4008$) định vị ranh giới tối ưu nhất.
* **Hình 3 (Cepstrum):** Biểu đồ thể hiện tính ưu việt của phân vị Median trong việc tránh bẫy lệch đuôi của các thuật toán dựa trên phương sai.

---

## V. ĐÁNH GIÁ TỔNG KẾT & KHUYẾN NGHỊ KỸ THUẬT

### 1. Bảng Tổng Hợp Chiến Lược Phân Ngưỡng Không Giám Sát Tốt Nhất

| Thuật toán | Chiến lược tối ưu nhất | Ngưỡng $T^*$ tối ưu | Final Score (%) | So sánh với Có giám sát chuẩn | Đánh giá kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **ACF** | **Median 50% Percentile** | **0.4333** | **5.22%** | **Vượt qua** (5.22% vs 5.55%) | Tận dụng triệt để tiên nghiệm 50/50, tính toán $O(N)$ tức thời. |
| **AMDF** | **Otsu's Thresholding** | **0.4012** | **5.59%** | Tiệm cận sát sao (5.59% vs 4.02%) | Sai số $\Delta F_0$ chỉ $0.34\text{ Hz}$ ($0.21\%$), tách cực tiểu cực tốt. |
| **Cepstrum** | **Median 50% Percentile** | **0.0779** | **37.35%** | **Vượt xa Baseline** (37.35% vs 42.95%) | Kháng hiện tượng phân bố lệch đuôi nặng trên miền quefrency. |

### 2. Kết luận giải đáp yêu cầu của Giảng viên
1. **Tính khả thi của Học không giám sát:** Hoàn toàn có thể phân loại Voiced / Unvoiced chính xác cao mà **không cần bất kỳ nhãn `.lab` nào**.
2. **Giá trị của giả thiết tiên nghiệm $50/50$:** Không chỉ là một giả định lý thuyết, tỉ lệ $P(V) \approx 45.6\% - 45.7\%$ là một đặc trưng tự nhiên của chuỗi tiếng nói liên tục. Khi đưa giả định này vào thuật toán (thông qua Phân vị Median $P_{50}$ hoặc tính đối xứng của phương sai Otsu $\omega_0 \omega_1$), hệ thống đạt được hiệu năng ngang bằng, thậm chí vượt trội mô hình phân loại Bayes có giám sát phụ thuộc nhãn con người.
3. **Mã nguồn và script kiểm chứng:** Tất cả mã nguồn giải thuật đã được module hóa tại [`src/analysis/unsupervised_threshold.py`](../src/analysis/unsupervised_threshold.py) và script thực nghiệm kiểm chứng tự động tại [`scripts/15_evaluate_unsupervised_thresholds.py`](../scripts/15_evaluate_unsupervised_thresholds.py).


---

## VI. NGHIÊN CỨU CHUYÊN SÂU: 4 GIẢI PHÁP NÂNG CAO VƯỢT GIỚI HẠN HISTOGRAM 1D

Nhằm giải quyết triệt để các hạn chế cố hữu của biểu đồ Histogram 1D rời rạc (lỗi chia bin, giới hạn đơn biến, và giả định độc lập thời gian i.i.d), nhóm đã phát triển và thử nghiệm 4 giải pháp học không giám sát nâng cao tại [`src/analysis/advanced_unsupervised.py`](../src/analysis/advanced_unsupervised.py) và script benchmark [`scripts/16_evaluate_advanced_unsupervised.py`](../scripts/16_evaluate_advanced_unsupervised.py).

```mermaid
flowchart TD
    subgraph H1["1. GMM-EM 1D (50/50 Continuous)"]
        A1["Dữ liệu 1D"] --> B1["Kỳ vọng - Cực đại hóa (EM)<br>π_V = π_UV = 0.5"]
        B1 --> C1["Giao điểm mật độ Bayes liên tục<br>(Khử lỗi chia bin Histogram)"]
    end

    subgraph H2["2. Đa đặc trưng 2D (Periodicity + ZCR)"]
        A2["Vector [Periodicity, ZCR]"] --> B2["K-Means 2D Clustering"]
        B2 --> C2["Siêu phẳng phân chia 2D<br>(Khử nhầm lẫn phụ âm xát)"]
    end

    subgraph H3["3. 2-State HMM (Baum-Welch)"]
        A3["Chuỗi thời gian liên tục"] --> B3["EM Baum-Welch (P(V→V) ≈ 0.85)"]
        B3 --> C3["Giải mã Viterbi toàn cục<br>(Khử rung giật thời gian)"]
    end

    subgraph H4["4. Ngưỡng Động Thích Nghi"]
        A4["Cửa sổ trượt 500ms"] --> B4["Ước lượng năng lượng & SNR cục bộ"]
        B4 --> C4["Ngưỡng trôi thích ứng T(t)<br>(Khử chênh lệch âm lượng to/nhỏ)"]
    end
```

---

### 1. Cơ Sở Toán Học 4 Giải Pháp Nâng Cao

#### 1.1. Mô Hình Hỗn Hợp Gauss 1D với Ràng Buộc Tiên Nghiệm 50/50 (1D GMM-EM Continuous)
* **Khắc phục:** Lỗi răng cưa và méo mó phân bố do rời rạc hóa bin histogram.
* **Mô hình toán:**
  $$p(x) = 0.5 \cdot \mathcal{N}(x; \mu_{UV}, \sigma_{UV}^2) + 0.5 \cdot \mathcal{N}(x; \mu_V, \sigma_V^2)$$
* **Thuật toán EM:** Cố định trọng số tiên nghiệm $\pi_{UV} = \pi_V = 0.5$, lặp bước E (tính trách nhiệm $\gamma$) và bước M (cập nhật $\mu, \sigma$).
* **Nghiệm ngưỡng tối ưu liên tục:** Giải chính xác phương trình bậc hai $p(x|V) = p(x|UV)$ trong miền liên tục, hoàn toàn không phụ thuộc vào số lượng bin:
  $$A x^2 + B x + C = 0 \implies T^* = \frac{-B \pm \sqrt{B^2 - 4AC}}{2A}$$

#### 1.2. Học Không Giám Sát Đa Đặc Trưng Không Gian 2D (Periodicity + Zero Crossing Rate ZCR)
* **Khắc phục:** Giới hạn đơn biến (Univariate failure) — các phụ âm vô thanh xát mạnh (/s/, /sh/) có năng lượng cao dễ bị nhận nhầm thành Voiced.
* **Đặc trưng bổ trợ:**
  $$\text{ZCR} = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$
  * Âm hữu thanh (Voiced): Sóng hài tần số thấp chiếm ưu thế $\implies \text{ZCR} < 0.15$.
  * Âm vô thanh (Unvoiced): Nhiễu xát tần số cao chiếm ưu thế $\implies \text{ZCR} > 0.35$.
* **Phân cụm K-Means 2D:** Chuẩn hóa z-score trên không gian $[\text{Periodicity}, \text{ZCR}]$, tách biệt dữ liệu thành 2 cụm riêng biệt trong mặt phẳng 2 chiều.

#### 1.3. Mô Hình Markov Ẩn 2 Trạng Thái Không Giám Sát (2-State HMM với Baum-Welch & Viterbi)
* **Khắc phục:** Giả định độc lập thời gian (i.i.d) của các bộ phân ngưỡng tĩnh đơn khung.
* **Mô hình trạng thái:** $S_0 = UV$, $S_1 = V$. Khai thác quán tính chuyển động thanh đới qua ma trận chuyển đối xứng:
  $$A = \begin{bmatrix} 0.85 & 0.15 \\ 0.15 & 0.85 \end{bmatrix}$$
* **Học không giám sát & Giải mã:** Thuật toán Baum-Welch tự động học phân bố phát xạ Gaussian mà không cần nhãn; thuật toán Viterbi giải mã chuỗi trạng thái tối ưu toàn cục, loại bỏ triệt để các khung rung giật nhấp nháy $V \to UV \to V$.

#### 1.4. Ngưỡng Động Thích Nghi Theo Ngữ Cảnh Cửa Sổ Trượt (Context-Adaptive Dynamic Thresholding)
* **Khắc phục:** Sự trôi dạt mức năng lượng và tỷ số tín hiệu trên nhiễu (SNR) theo thời gian.
* **Cơ chế cửa sổ trượt:** Sử dụng cửa sổ trượt $50\text{ khung} \approx 500\text{ ms}$, tính ngưỡng thích ứng động:
  $$T_{\text{dynamic}}(t) = (1 - \alpha) \cdot T_{\text{global}} + \alpha \cdot T_{\text{local}}(t)$$
  với $\alpha = 0.30$ giúp ngưỡng tự động hạ thấp ở các đoạn âm thì thầm và nâng cao ở các đoạn hô lớn.

---

### 2. Bảng Đối Sánh Toàn Chu Trình 8 Cột Trên Tập Kiểm Thử (TinHieuKiemThu)

Toàn bộ 7 chiến lược (Supervised Bayes, Median 50%, Otsu, GMM-EM 1D, Đa đặc trưng 2D, 2-State HMM, và Ngưỡng Động) được đối đầu trực tiếp trên cùng tập dữ liệu kiểm thử:

#### A. Thuật toán ACF (Autocorrelation)
| Chiến lược phân ngưỡng | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | FINAL SCORE (%) | Xếp hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. GMM-EM 1D (50/50 Liên tục)** | **1.79** | **1.18** | **1.22** | **3.96** | **14.2** | **8.32** | **4.49%** | 🥇 **Vô địch** |
| **2. Đa đặc trưng 2D (Periodicity+ZCR)** | 1.47 | 0.96 | 1.24 | 2.92 | 18.5 | 10.43 | **4.77%** | 🥈 **Á quân** |
| **4. Ngưỡng Động Thích Nghi** | 1.39 | 0.95 | 1.49 | 4.23 | 16.5 | 9.97 | **5.04%** | 🥉 **Hạng 3** |
| **Median 50% (Histogram)** | 1.52 | 1.04 | 1.55 | 4.55 | 16.8 | 10.07 | **5.22%** | Hạng 4 |
| *Supervised (Bayes Lab)* | 1.46 | 0.99 | 1.65 | 5.17 | 17.2 | 10.47 | *5.55%* | Hạng 5 |
| **Otsu (Histogram)** | 1.55 | 1.06 | 1.63 | 5.24 | 19.0 | 11.45 | **5.91%** | Hạng 6 |
| **3. 2-State HMM (Viterbi)** | 4.01 | 2.60 | 5.43 | 16.25 | 47.8 | 31.33 | **16.73%** | Hạng 7 |

> [!TIP]
> **Đột phá trên ACF:**
> Cả 3 giải pháp nâng cao (**GMM-EM 4.49%**, **Đa đặc trưng 4.77%**, **Ngưỡng Động 5.04%**) đều xuất sắc **vượt qua phương pháp Có giám sát Bayes (5.55%)** và Median 50% (5.22%). Trong đó GMM-EM 1D thiết lập kỷ lục mới với Final Score **4.49%**!

---

#### B. Thuật toán AMDF (Average Magnitude Difference)
| Chiến lược phân ngưỡng | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | FINAL SCORE (%) | Xếp hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Supervised (Bayes Lab)* | 0.68 | 0.41 | 1.28 | 3.50 | 14.2 | 8.15 | *4.02%* | Chuẩn đối chứng |
| **1. GMM-EM 1D (50/50 Liên tục)** | **0.50** | **0.29** | **1.27** | **3.46** | **15.0** | **8.77** | **4.17%** | 🥇 **Tốt nhất Không Giám Sát** |
| **4. Ngưỡng Động Thích Nghi** | 0.33 | 0.23 | 1.62 | 4.43 | 17.5 | 10.48 | **5.04%** | 🥈 |
| **Otsu (Histogram)** | 0.34 | 0.21 | 1.72 | 4.92 | 19.0 | 11.64 | **5.59%** | 🥉 |
| **Median 50% (Histogram)** | 1.48 | 0.94 | 3.57 | 11.06 | 13.2 | 7.49 | **6.50%** | Hạng 4 |
| **2. Đa đặc trưng 2D (Periodicity+ZCR)** | 0.96 | 0.63 | 3.81 | 12.34 | 14.8 | 8.23 | **7.07%** | Hạng 5 |
| **3. 2-State HMM (Viterbi)** | 3.79 | 2.60 | 2.96 | 12.17 | 59.0 | 40.05 | **18.27%** | Hạng 6 |

> [!NOTE]
> **Điểm nhấn trên AMDF:**
> Thuật toán **GMM-EM 1D ($T = 0.4322$)** đạt Final Score **4.17%**, tiệm cận sát sao đến mức kinh ngạc với phương pháp có giám sát chuẩn ($4.02\%$), khoảng cách chỉ vỏn vẹn **0.15%**! Đặc biệt sai số $\Delta F_0$ chỉ là **$0.50\text{ Hz}$ ($0.29\%$)**, chính xác hơn cả Supervised Bayes ($0.68\text{ Hz}$).

---

#### C. Thuật toán Cepstrum
| Chiến lược phân ngưỡng | $\Delta F_0$ (Hz) | $\text{MAPE } F_0$ (%) | $\Delta\text{Std}$ (Hz) | $\text{MAPE Std}$ (%) | $\Delta\text{Num}$ (Khung) | $\text{MAPE Num}$ (%) | FINAL SCORE (%) | Xếp hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2. Đa đặc trưng 2D (Periodicity+ZCR)** | **12.62** | **7.96** | **6.13** | **22.93** | **99.5** | **69.27** | **33.38%** | 🥇 **Vô địch** |
| **1. GMM-EM 1D (50/50 Liên tục)** | 13.12 | 8.93 | 10.05 | 53.26 | 53.8 | 39.07 | **33.75%** | 🥈 **Á quân** |
| **4. Ngưỡng Động Thích Nghi** | 12.31 | 8.47 | 10.64 | 57.21 | 54.0 | 38.87 | **34.85%** | 🥉 **Hạng 3** |
| **3. 2-State HMM (Viterbi)** | 16.46 | 10.78 | 8.13 | 36.01 | 81.8 | 58.67 | **35.16%** | Hạng 4 |
| **Median 50% (Histogram)** | 14.24 | 9.84 | 12.74 | 68.52 | 46.0 | 33.70 | **37.35%** | Hạng 5 |
| *Supervised (Baseline cũ)* | 17.23 | 12.05 | 19.27 | 98.55 | 23.2 | 18.24 | *42.95%* | Hạng 6 |
| **Otsu (Histogram)** | 44.41 | 25.64 | 13.84 | 46.43 | 128.5 | 87.90 | **53.33%** | Hạng 7 |

> [!IMPORTANT]
> **Bước nhảy vọt toàn diện trên Cepstrum:**
> Cả 4 giải pháp nâng cao đều **đồng loạt đánh bại Baseline cũ (42.95%) và Median 50% (37.35%)**.  
> Phương pháp **Đa đặc trưng 2D (33.38%)** và **GMM-EM 1D (33.75%)** giảm gần 10 điểm phần trăm lỗi so với Baseline cũ, giải quyết xuất sắc tình trạng phân bố quefrency bị méo đuôi.

---

### 3. Trực Quan Hóa Đồ Thị Thực Nghiệm 300 DPI

![Mật độ xác suất liên tục GMM-EM 1D](../outputs/figures/16_advanced_gmm_em_distributions.png)
![Biểu đồ phân tán Đa đặc trưng 2D](../outputs/figures/16_advanced_multifeature_2d_scatter.png)
![Khử rung giật bằng 2-State HMM](../outputs/figures/16_advanced_hmm_temporal_tracking.png)
![Xếp hạng Final Score tổng hợp](../outputs/figures/16_advanced_final_ranking_barchart.png)

* **Hình 1 (GMM-EM Continuous):** Đường cong mật độ xác suất liên tục $p(x)$ phân rã sắc nét thành 2 thành phần Voiced và Unvoiced cân bằng 50/50, cho giao điểm Bayes $T^*$ mượt mà không bị răng cưa binning.
* **Hình 2 (Multi-Feature 2D):** Không gian $(\text{Periodicity}, \text{ZCR})$ tách biệt 2 cụm V và UV thành hai hòn đảo rõ ràng; việc đưa thêm trục ZCR giúp phân định triệt để các phụ âm xát vô thanh có năng lượng lớn.
* **Hình 3 (2-State HMM Viterbi):** Đồ thị sóng âm chứng minh thuật toán Viterbi khử sạch 100% các xung gai rung giật trạng thái đơn lẻ vốn thường xuất hiện ở ngưỡng tĩnh 1D.
* **Hình 4 (Xếp hạng Final Score):** Biểu đồ cột tổng quan minh họa vị thế dẫn đầu vượt bậc của GMM-EM 1D và Đa đặc trưng 2D trên cả 3 thuật toán Bài Tập 2.

---

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

<!-- STAGE4_CURRENT_START -->
## Kết quả kiểm chứng hiện hành — 30/10/2026

Nguồn: cấu hình đóng băng và `outputs/stage4_validation/evaluation.json`. Final Score là sai số thống kê toàn file; thấp hơn tốt hơn. Accuracy/F1 từng khung không xác định vì thiếu nhãn thời gian.

Giao thức `train_all` gồm silence, `train_active` chỉ active train, `input_active` fit riêng từng file đầu vào. Tất cả dùng cùng frame/hop 25/10 ms, silence gate và median theo đoạn.

| Giao thức | ACF test (%) | AMDF test (%) | Cepstrum test (%) |
|---|---:|---:|---:|
| historical_static | 5.822 | 6.357 | 48.185 |
| median_train_all | 5.170 | 7.425 | 40.370 |
| otsu_train_all | 6.350 | 7.255 | 53.017 |
| valley_train_all | 18.090 | 6.790 | 71.160 |
| kmeans_train_all | 6.453 | 7.152 | 57.000 |
| gmm_train_all | 5.170 | 6.665 | 33.108 |
| dynamic_train_all | 5.925 | 7.635 | 33.770 |
| median_train_active | 14.575 | 17.335 | 34.733 |
| otsu_train_active | 8.832 | 6.955 | 58.255 |
| valley_train_active | 46.245 | 18.640 | 71.160 |
| kmeans_train_active | 9.525 | 6.955 | 58.343 |
| gmm_train_active | 11.560 | 13.673 | 47.920 |
| dynamic_train_active | 11.730 | 12.450 | 44.578 |
| kmeans2d_train_active | 5.958 | 7.558 | 57.987 |
| hmm_train_active | 12.608 | 12.188 | 44.380 |
| kmeans2d_input_active | 7.635 | 8.165 | 31.635 |
| hmm_input_active | 16.240 | 17.825 | 30.843 |

Các thứ hạng test là mô tả; không được dùng để chọn lại tham số.

![Giao thức hiện hành](../outputs/stage4_validation/figures/protocols_test.png)

[Chi tiết và giới hạn kiểm chứng](STAGE4_RESULTS.md).
<!-- STAGE4_CURRENT_END -->
