# CHƯƠNG III: PHÂN TÍCH HIỆN TƯỢNG TRÊN ĐỒ THỊ & NGUYÊN NHÂN SAI SỐ

> **Dẫn hướng tài liệu:** [Trang chủ README](../README.md) > **Chương III: Phân tích hiện tượng sai số**

---

Trong quá trình thực nghiệm đối sánh trên cả hai miền thuật toán ACF và AMDF, nhiều hiện tượng vật lý âm học và xử lý tín hiệu số đặc thù đã xuất hiện. Dưới đây là phân tích chi tiết bản chất nguyên nhân sai số và cơ chế giải quyết của hệ thống Plugins:

## III.1. Sự Suy Giảm Chất Lượng Giữa Kênh Thoại (Phone) và Phòng Thu (Studio)

* **Tín hiệu phòng thu (`studio_*`):**
  * Tỷ số tín hiệu trên nhiễu (SNR) rất cao, dải thông tần số rộng ($[0, f_s/2]$).
  * Cả hai thuật toán đều phát huy tối đa độ chính xác, sai số tiệm cận mức lý tưởng ($\lvert\Delta F_0\rvert < 0.2\text{ Hz}$, độ chính xác phân loại V/UV $> 91\%$, F1-Score $> 95\%$).

* **Tín hiệu điện thoại (`phone_*`):**
  * Tín hiệu kênh thoại bị giới hạn băng thông tiêu chuẩn viễn thông ($300 - 3400\text{ Hz}$).
  * Đối với giọng nam trầm ($F_0 \approx 100 - 150\text{ Hz} < 300\text{ Hz}$), thành phần tần số cơ bản thứ nhất bị bộ lọc kênh thoại triệt tiêu hoàn toàn. Thuật toán thời gian buộc phải bắt chu kỳ dựa trên bao tương quan của các sóng hài bậc cao ($2F_0, 3F_0, \dots$).
  * Hiện tượng này làm giảm độ sắc nét của đỉnh tương quan $R(\tau)$ và đáy cực tiểu $D(\tau)$, dẫn tới sai số trung bình trên kênh thoại cao hơn kênh phòng thu ($3 - 5\text{ Hz}$).

---

## III.2. Hiện Tượng Đứt Gãy Contour Tại Ranh Giới Âm Tiết (Boundary Dropping)

* **Cơ chế vật lý:**
  * Tại vùng bắt đầu phát âm (*onset*) hoặc vùng kết thúc tắt âm (*offset*), dây thanh âm bắt đầu khép lại hoặc mở dần ra.
  * Trong khoảng $20 - 40\text{ ms}$ này, tính tuần hoàn của sóng âm bắt đầu suy giảm nhanh khiến đỉnh tương quan cực đại tụt xuống dưới ngưỡng tĩnh $T$, mặc dù năng lượng âm thanh của âm tiết vẫn còn đáng kể và tai người vẫn cảm nhận được cao độ.
* **Hậu quả trên mô hình Baseline:**
  * Mô hình Baseline đơn ngưỡng tĩnh bị "rụng" mất 1–2 khung viền ở đầu và cuối mỗi từ, gây đứt đoạn đường pitch contour và làm giảm độ bao phủ (Recall / Voiced Accuracy).
* **Giải pháp khắc phục:**
  * **Ngưỡng trễ kép Hysteresis (Schmitt Trigger):** Giữ trạng thái Voiced khi đỉnh trôi xuống mức $T_{\text{low}} = 0.40$, chống hiện tượng rung lật quyết định.
  * **Mở rộng năng lượng khung biên (Energy Extension):** Quét các khung lân cận mép phân đoạn hữu thanh; nếu năng lượng ngắn hạn $\text{STE} \ge 20\% E_{\max}$, khung được phục hồi trạng thái Hữu thanh, giúp đường pitch contour bám trọn vẹn từ đầu đến cuối âm tiết.

---

## III.3. Hiện Tượng Nhảy Quãng Tám (Octave Doubling / Octave Halving)

* **Lỗi nhân đôi cao độ (Octave Doubling - $2F_0$):**
  * Xảy ra khi sóng âm có một formant cộng hưởng rất mạnh ở tần số thấp gần với hài bậc hai (ví dụ nguyên âm /a/, /o/). Đỉnh tương quan phụ tại $\tau = \tau_0 / 2$ nhô cao hơn đỉnh chính tại $\tau_0$, khiến thuật toán chọn nhầm chu kỳ cơ bản bằng một nửa $\implies \hat{F}_0 = 2 F_0$.
* **Lỗi chia đôi cao độ (Octave Halving - $F_0 / 2$):**
  * Thường gặp trên giọng nam trầm khi các xung thanh môn xen kẽ có biên độ chênh lệch nhau (hiện tượng *vocal fry* hoặc *diplophonia*). Thuật toán chỉ nhận diện được sự tuần hoàn sau mỗi 2 chu kỳ thật ($\tau = 2\tau_0$), dẫn tới $\hat{F}_0 = F_0 / 2$.
* **Giải pháp khắc phục:**
  * **Cắt gọt trung tâm (Center Clipping - Sondhi):** Cắt bỏ $40\%$ biên độ trung tâm làm phẳng phổ (Spectral Flattening), triệt tiêu hoàn toàn ảnh hưởng của đỉnh formant phụ, bảo tồn duy nhất các xung thanh môn chính.
  * **Quy hoạch động Viterbi Tracking:** Lưới Trellis phạt nặng các bước nhảy tần số rơi vào vùng quãng tám ($[0.8, 1.2]\text{ octave}$), chọn đường đi tối ưu mượt mà và triệt tiêu hoàn toàn các bước nhảy đột biến.

---

## III.4. Hiệu Quả của Bộ Lọc Trung Vị (Median Filter 3 khung)

* **Cơ chế:**
  * Tại mỗi khung $i$, giá trị cao độ được lọc theo cửa sổ trượt kích thước 3:
    $$\hat{F}_{0, i}^{\text{med}} = \text{median}\left(\hat{F}_{0, i-1}, \hat{F}_{0, i}, \hat{F}_{0, i+1}\right)$$
* **Ưu điểm:**
  * Khử triệt để các xung lỗi cô lập (Impulsive Spike) sinh ra do chuyển âm nhanh hoặc bắt nhầm đỉnh nhiễu trong 1 khung duy nhất.
  * Khác với các bộ lọc thông thấp IIR/FIR làm trễ pha và làm tù các đỉnh nhọn cao độ tự nhiên, bộ lọc trung vị bảo toàn trọn vẹn ranh giới bước nhảy thực của ngôn điệu.

---

**Dẫn hướng:** [← Chương II: Thuật toán AMDF](02_amdf_theory_and_experiments.md) | [Trang chủ README](../README.md) | [Chương IV: Khảo sát kháng nhiễu →](04_noise_robustness_study.md)
