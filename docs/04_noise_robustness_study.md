# CHƯƠNG IV: KHẢO SÁT CHUYÊN SÂU ĐỘ BỀN VỮNG KHÁNG NHIỄU (NOISE ROBUSTNESS)

> **Dẫn hướng tài liệu:** [Trang chủ README](../README.md) > **Chương IV: Khảo sát kháng nhiễu**

---

## IV.1. Cơ Sở Lý Thuyết & Bản Chất Toán Học

Thực nghiệm ở Chương I và Chương II cho thấy **AMDF vượt trội hơn ACF trên tập dữ liệu kiểm thử sạch** ($1.60\text{ Hz}$ vs $3.17\text{ Hz}$ ở Baseline). Tuy nhiên, đây là kết quả trong điều kiện lý tưởng (ít tạp âm nền). Khi bước ra môi trường thực tế có nhiễu ngẫu nhiên, hai thuật toán phản ứng hoàn toàn trái ngược nhau về mặt toán học:

### 1. Thuật toán ACF (Chuẩn $L_2$ - Tích tương quan)
* Giả sử tín hiệu thu được $s[n] = x[n] + w[n]$, trong đó $x[n]$ là tiếng nói tuần hoàn và $w[n]$ là nhiễu trắng Gauss độc lập (AWGN) với kỳ vọng bằng $0$.
* Hàm tự tương quan tại độ trễ $\tau > 0$:

$$
R_{ss}(\tau) = R_{xx}(\tau) + R_{xw}(\tau) + R_{wx}(\tau) + R_{ww}(\tau)
$$

* Do tiếng nói và nhiễu không tương quan ($R_{xw} \approx 0, R_{wx} \approx 0$), và nhiễu trắng độc lập giữa các mẫu ($R_{ww}(\tau) = \sigma^2 \delta[\tau]$):

$$
\forall \tau > 0: \quad R_{ww}(\tau) = 0 \implies R_{ss}(\tau) \approx R_{xx}(\tau)
$$

* **Kết luận:** Nhiễu trắng chỉ cộng dồn năng lượng vào đỉnh gốc $\tau = 0$ ($R(0) = P_{\text{sig}} + \sigma^2$). Tại các độ trễ chu kỳ $\tau_0 > 0$, **ACF có khả năng tự triệt tiêu nhiễu ngẫu nhiên**, giữ cho đỉnh tương quan $T_0$ nhô cao bền bỉ.

### 2. Thuật toán AMDF (Chuẩn $L_1$ - Hiệu độ lớn tuyệt đối)
* Khi tín hiệu bị pha tạp nhiễu trắng $w[n] \sim \mathcal{N}(0, \sigma^2)$, hiệu số hai biến ngẫu nhiên Gauss độc lập $(w[n] - w[n+\tau])$ là một biến ngẫu nhiên Gauss có phương sai $2\sigma^2$.
* Kỳ vọng toán học của giá trị tuyệt đối hiệu số nhiễu:

$$
\mathbb{E}[\lvert w[n] - w[n+\tau] \rvert] = \frac{2}{\sqrt{\pi}}\sigma \ne 0
$$

* **Kết luận:** Thay vì triệt tiêu về $0$, nhiễu tạo ra một **ngưỡng sàn giá trị dương (Noise Floor)** nâng toàn bộ hàm AMDF lên cao. Khi công suất nhiễu tăng (SNR giảm), sàn nhiễu này lấp phẳng các đáy cực tiểu, khiến đáy thật tại $\tau_0$ bị nông hóa và chìm hoàn toàn vào nhiễu.

---

## IV.2. Thiết Lập Thực Nghiệm Đa Mức Nhiễu AWGN

Để kiểm chứng thực nghiệm hiện tượng trên, đồ án xây dựng kịch bản kiểm thử ứng suất (Stress Test) trong `scripts/06_noise_robustness.py`:
* **Mô hình nhiễu:** Bơm nhiễu trắng Gauss (AWGN) vào toàn bộ 4 file kiểm thử (`phone_F2`, `phone_M2`, `studio_F2`, `studio_M2`) theo công thức:

$$
P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}} / 10}}
$$

* **Dải khảo sát SNR:** Gồm 8 nấc từ hoàn hảo đến cực đoan:

$$
\text{SNR} \in [\text{Clean } (\infty), +25\text{ dB}, +20\text{ dB}, +15\text{ dB}, +10\text{ dB}, +5\text{ dB}, 0\text{ dB}, -5\text{ dB}]
$$

*(Trong đó tại $0\text{ dB}$, công suất nhiễu bằng đúng công suất tiếng nói; tại $-5\text{ dB}$, năng lượng nhiễu lấn át tiếng nói gấp $3.16$ lần).*

* **Đối tượng đối sánh:**
  * `ACF Baseline` ($T = 0.4408$) vs `AMDF Baseline` ($T = 0.4380$)
  * `ACF Enhanced` (Cấu hình 29: BP + Clip + Energy + Viterbi) vs `AMDF Enhanced` (Cấu hình 13: Clip + Viterbi)

---

## IV.3. Bảng Kết Quả Đối Kháng Thực Nghiệm

| Mức SNR (dB) | Môi trường âm học thực tế | Sai số ACF Baseline | Sai số AMDF Baseline | Sai số ACF Enhanced | Sai số AMDF Enhanced | F1 ACF Base | F1 AMDF Base | Thuật toán vượt trội |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | Phòng thu chuẩn (Không nhiễu) | 3.17 Hz | **1.60 Hz** | 1.90 Hz | **1.24 Hz** | 91.03% | **92.30%** | **AMDF** |
| **+25 dB** | Môi trường studio / phòng kín | 3.18 Hz | **1.56 Hz** | 1.87 Hz | **1.19 Hz** | 90.84% | **92.26%** | **AMDF** |
| **+20 dB** | Văn phòng làm việc yên tĩnh | 2.95 Hz | **1.72 Hz** | 1.95 Hz | **1.27 Hz** | 90.88% | **91.73%** | **AMDF** |
| **+15 dB** | Phòng họp có tiếng người xa | 3.06 Hz | **2.29 Hz** | 1.86 Hz | **1.65 Hz** | 89.91% | **90.02%** | **AMDF** |
| **+10 dB** | Quán cà phê / Nhà ăn | 3.60 Hz | **2.86 Hz** | **2.46 Hz** | 3.29 Hz | **86.92%** | 85.52% | **Giao thoa** (ACF Enh dẫn) |
| **+5 dB** | Đường phố đông đúc xe cộ | **4.91 Hz** | 5.43 Hz | **2.79 Hz** | 4.69 Hz | **80.58%** | 77.48% | **ACF** |
| **0 dB** | Tiếng ồn cực lớn ($P_{\text{noise}} = P_{\text{sig}}$) | **9.06 Hz** | 12.07 Hz | **2.93 Hz** | 10.23 Hz | **59.01%** | 46.35% | **ACF** |
| **-5 dB** | Môi trường công trường / bão gió | **27.81 Hz** | 42.83 Hz | **1.97 Hz** | 84.10 Hz | **12.53%** | 3.46% | **ACF** |

---

## IV.4. Phân Tích Điểm Giao Thoa (Cross-over Point)

Thực nghiệm chỉ ra một bức tranh khoa học hoàn chỉnh và nhất quán:

1. **Vùng tín hiệu sạch ($\text{SNR} \ge 15\text{ dB}$): AMDF chiếm ưu thế tuyệt đối:**
   * Độ sắc nhọn của đáy cực tiểu và tính tuyến tính của chuẩn $L_1$ giúp AMDF ít bị ảnh hưởng bởi formant ripple hơn ACF, đạt sai số vượt trội ($1.24\text{ Hz}$ vs $1.90\text{ Hz}$).

2. **Điểm giao thoa (Cross-over Zone tại $\text{SNR} = 7 - 10\text{ dB}$):**
   * Ngay khi mức nhiễu chạm ngưỡng $+10\text{ dB}$, cấu hình `ACF Enhanced` ($2.46\text{ Hz}$) đã chính thức vượt qua `AMDF Enhanced` ($3.29\text{ Hz}$).
   * Ở mức $+5\text{ dB}$, `ACF Baseline` ($4.91\text{ Hz}$) đánh bại `AMDF Baseline` ($5.43\text{ Hz}$) ở cả sai số lẫn độ chính xác phân loại F1 ($80.58\%$ vs $77.48\%$).

3. **Vùng nhiễu cực nặng ($\text{SNR} \le 0\text{ dB}$): ACF chứng minh sức mạnh tự khử nhiễu:**
   * Tại $0\text{ dB}$, đáy cực tiểu của AMDF bị sàn nhiễu nâng cao, dẫn đến lỗi nhảy chu kỳ (Pitch Doubling / Octave Halving), khiến sai số AMDF vọt lên $12.07\text{ Hz}$. Trong khi đó, ACF Baseline chỉ lệch $9.06\text{ Hz}$.
   * Đặc biệt, cấu hình `ACF Enhanced` (với bộ lọc Bandpass chặn nhiễu ngoài dải tần tiếng nói và Viterbi Tracking nắn đường đi) duy trì sai số xuất sắc **$2.93\text{ Hz}$**, trong khi `AMDF Enhanced` bị sụp đổ ở mức **$10.23\text{ Hz}$** (gấp hơn 3.4 lần).
   * Tại $-5\text{ dB}$, AMDF bị phá hủy hoàn toàn (sai số $84.10\text{ Hz}$, F1 chỉ còn $3.46\%$).

### Biểu đồ đường cong suy giảm hiệu năng theo SNR:
![06_noise_robustness_curves.png](../outputs/figures/06_noise_robustness_curves.png)

---

## IV.5. Minh Họa Trực Quan Cơ Chế Kháng Nhiễu Tại 0 dB

Để giải thích trực quan tại sao ACF sống sót còn AMDF thất bại tại $0\text{ dB}$ SNR, biểu đồ dưới đây trích xuất một khung nguyên âm hữu thanh đại diện (`studio_F2.wav`, $F_0 \approx 200\text{ Hz}$, $T_0 \approx 5.0\text{ ms}$):

* **Miền thời gian:** Tín hiệu $0\text{ dB}$ bị nhiễu biến dạng hoàn toàn, mắt thường khó nhận biết chu kỳ sóng.
* **Hàm ACF:** Nhờ tính chất $R_{ww}(\tau) \approx 0$, thành phần nhiễu tự triệt tiêu lẫn nhau, làm cho **đỉnh tương quan tại $T_0 = 5.0\text{ ms}$ vẫn nhô cao sừng sững vượt ngưỡng phân tách**.
* **Hàm AMDF:** Do kỳ vọng tuyệt đối của nhiễu là $\frac{2}{\sqrt{\pi}}\sigma > 0$, toàn bộ đường cong AMDF bị nhấc bổng lên cao ($D(\tau) > 0.6$). **Đáy cực tiểu tại $T_0 = 5.0\text{ ms}$ bị lấp phẳng hoàn toàn**, dẫn đến thuật toán bắt nhầm đáy giả ở tần số cao.

### Biểu đồ minh họa cơ chế trên khung đơn lẻ (0 dB SNR):
![06_noise_mechanism_frame_demo.png](../outputs/figures/06_noise_mechanism_frame_demo.png)

---

**Dẫn hướng:** [← Chương III: Phân tích hiện tượng sai số](03_error_analysis_and_phenomena.md) | [Trang chủ README](../README.md) | [Chương V: Đánh giá chuẩn quốc tế PTDB-TUG →](05_ptdb_tug_international_benchmark.md)
