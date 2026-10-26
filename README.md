# XỬ LÝ TÍN HIỆU TIẾNG NÓI (SPEECH PROCESSING)
# BÁO CÁO NGHIÊN CỨU & HIỆN THỰC THUẬT TOÁN ƯỚC LƯỢNG CAO ĐỘ $F_0$ VÀ PHÂN LOẠI HỮU THANH / VÔ THANH

> **Chủ đề:** Ước lượng tần số cơ bản $F_0$ (Pitch Contour Tracking) và phân loại Hữu thanh (Voiced) / Vô thanh (Unvoiced).  
> **Ba thuật toán trọng tâm (Bài Tập 2):**  
> 1. **Thuật toán Hàm tự tương quan (Autocorrelation Function - ACF)** (Miền thời gian chuẩn $L_2$)  
> 2. **Thuật toán Hàm hiệu độ lớn trung bình (Average Magnitude Difference Function - AMDF)** (Miền thời gian chuẩn $L_1$)  
> 3. **Thuật toán Đồng cấu Cepstrum (Quefrency / Homomorphic Deconvolution)** (Miền phổ Logarit & Quefrency)  
> **Bộ dữ liệu chuẩn:**  
> * Dữ liệu huấn luyện: `TinHieuHuanLuyen/` (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1` - $f_s = 16\text{ kHz}$)  
> * Dữ liệu kiểm thử: `TinHieuKiemThu/` (`phone_F2`, `phone_M2`, `studio_F2`, `studio_M2` - $f_s = 16\text{ kHz}$)  

---

## 1. Tổng Quan Dự Án & Điểm Nhấn Khoa Học

Dự án hiện thực toàn diện hệ sinh thái ước lượng cao độ tiếng nói và phân loại âm hữu thanh/vô thanh, kết hợp giữa mô hình phân tích thống kê xác suất (Phân bố Gauss), hệ sinh thái plugin 3 chặng cắm/rút độc lập và 3 thuật toán nền tảng của Bài Tập 2 (ACF, AMDF, Cepstrum):

* **Chuẩn hóa thông số khung phân tích:** Chốt độ dài khung chuẩn **$25\text{ ms}$** (400 mẫu tại $16\text{ kHz}$) và độ dịch khung **$10\text{ ms}$** (160 mẫu) theo nguyên lý bao phủ tối thiểu 2 chu kỳ âm vực nam trầm ($80\text{ Hz}$).
* **Huấn luyện ngưỡng tối ưu Gauss (Bayesian Decision Rule):** Giải phương trình giao điểm mật độ xác suất của 2 phân bố $\mathcal{N}(\mu_V, \sigma_V^2)$ và $\mathcal{N}(\mu_U, \sigma_U^2)$ trên dữ liệu huấn luyện, tìm ra ngưỡng khách quan $T_{\text{ACF}} = 0.4408$ và $T_{\text{AMDF}} = 0.4380$.
* **Hệ sinh thái Plugins cắm/rút độc lập:**
  1. *Tiền xử lý (Pre-processing):* Bộ lọc thông dải Butterworth bậc 2 Zero-Phase ($70 - 900\text{ Hz}$) và Cắt gọt biên độ trung tâm Center Clipping (Sondhi 1968) làm phẳng phổ.
  2. *Quyết định (Decision):* Ngưỡng trễ kép Hysteresis (Schmitt Trigger) chống rung lật trạng thái.
  3. *Hậu xử lý (Post-processing):* Mở rộng vùng hữu thanh theo năng lượng khung biên STE và Quy hoạch động Viterbi Tracking nắn đường contour tối ưu toàn cục.
* **Đánh giá đa tiêu chí trên Tập Huấn Luyện (`TinHieuHuanLuyen`):** Khảo sát toàn bộ 16 tổ hợp plugin cho từng thuật toán và phân thành 4 nhóm tối ưu đại diện cho 3 chỉ số cốt lõi và điểm số toàn diện:
  - **ACF Champion (`BP + Energy + Viterbi`):** Giảm điểm lỗi tổng hợp từ $13.23\%$ xuống **$2.95\%$**, sai số $F_0 = 1.34\text{ Hz}$, $\Delta\text{Std} = 0.66\text{ Hz}$, $\Delta\text{Num} = 8.2\text{ khung}$.
  - **AMDF Champion (`Energy + Viterbi`):** Giảm điểm lỗi tổng hợp từ $7.01\%$ xuống **$3.29\%$**, sai số $F_0 = 1.30\text{ Hz}$, $\Delta\text{Std} = 0.96\text{ Hz}$, $\Delta\text{Num} = 9.2\text{ khung}$.
  - **Cepstrum Champion (`BP + Clip + Energy + Viterbi`):** Khử triệt để lỗi nhảy quãng tám từ Baseline ($39.81\%$), đưa điểm tổng hợp xuống **$4.44\%$**, sai số $F_0 = 1.42\text{ Hz}$, $\Delta\text{Std} = 1.28\text{ Hz}$, $\Delta\text{Num} = 9.2\text{ khung}$.

---

## 2. Cấu Trúc Thư Mục & Thiết Kế Kiến Trúc Module

Dự án áp dụng mô hình thiết kế hướng module, phân lập rõ ràng giữa nhân thuật toán xử lý tín hiệu số, tầng phân tích thống kê toán học, tầng trực quan hóa và hệ thống tài liệu:

```
FAC/
├── TinHieuHuanLuyen/            # Dữ liệu huấn luyện (phone_F1, phone_M1, studio_F1, studio_M1)
├── TinHieuKiemThu/              # Dữ liệu kiểm thử (phone_F2, phone_M2, studio_F2, studio_M2)
│
├── docs/                        # Báo cáo lý thuyết & thực nghiệm chuyên sâu cho từng thuật toán
│   ├── 01_acf_theory_and_experiments.md       # Chương I: Thuật toán ACF & 16 tổ hợp plugin
│   ├── 02_amdf_theory_and_experiments.md      # Chương II: Thuật toán AMDF & 16 tổ hợp plugin
│   ├── 03_cepstrum_theory_and_experiments.md  # Chương III: Thuật toán Cepstrum & Đối sánh BT2
│   └── 04_test_set_benchmark_and_analysis.md  # Chương IV: Đối sánh trên Tập Kiểm Thử & Phân tích hiện tượng
│
├── src/                         # Mã nguồn module hóa
│   ├── core/                    # Tầng xử lý tín hiệu cốt lõi
│   │   ├── audio.py             # Nạp file WAV, phân khung (framing), tính năng lượng ngắn hạn (STE)
│   │   ├── lab_parser.py        # Đọc nhãn mốc thời gian và thống kê chuẩn từ file *.lab
│   │   ├── acf.py               # Thuật toán Hàm tự tương quan (ACF) & dò cực đại
│   │   ├── amdf.py              # Thuật toán Hàm hiệu biên độ trung bình (AMDF) & dò cực tiểu
│   │   ├── cepstrum.py          # Thuật toán Đồng cấu Cepstrum (FFT/Log/IFFT, Quefrency Peak)
│   │   └── pitch_detector.py    # Bộ phát hiện Baseline PitchDetector (acf, amdf, cepstrum)
│   │
│   ├── plugins/                 # Hệ sinh thái Plugin 3 chặng
│   │   ├── base.py              # Interface cơ sở (PreProcessingPlugin, DecisionPlugin, PostProcessingPlugin)
│   │   ├── plugin_detector.py   # Bộ điều phối (Coordinator) thực thi tuần tự pipeline
│   │   ├── pre_processing/      # Chặng 1: Butterworth Bandpass Filter & Sondhi Center Clipping
│   │   ├── decision/            # Chặng 2: Schmitt Trigger Hysteresis Thresholding
│   │   └── post_processing/     # Chặng 3: STE Energy Extension & Viterbi Trellis Tracking
│   │
│   ├── analysis/                # Phân tích thống kê & đánh giá sai số
│   │   ├── threshold.py         # Trích xuất phân bố Gauss (mean, std), giải phương trình tìm T
│   │   └── evaluation.py        # Đánh giá định lượng F0, V/UV Accuracy, F1, MAPE, Final Score
│   │
│   └── visualization/           # Trực quan hóa dữ liệu
│       └── plotter.py           # Xuất đồ thị 300 DPI (Contour, Gaussian, Frames, Rankings)
│
├── scripts/                     # Kịch bản chạy tự động hóa hoàn chỉnh
│   ├── 01_demo_frames.py        # Minh họa 1 khung Voiced vs 1 khung Unvoiced (ACF, AMDF, Cepstrum)
│   ├── 02_train_threshold.py    # Huấn luyện tìm ngưỡng T tối ưu bằng phân bố Gauss
│   ├── 03_compare_params.py     # Khảo sát so sánh chiều dài khung (20ms vs 25ms vs 30ms)
│   ├── 04_run_testing.py        # Chạy kiểm thử tự động trên tập dữ liệu
│   ├── 05_compare_plugins.py    # Khảo sát 16 tổ hợp plugin cho ACF, AMDF, Cepstrum (chia 4 nhóm)
│   ├── 14_compare_acf_amdf_cepstrum.py # Đối sánh toàn diện 3 thuật toán BT2
│   ├── visualize_amdf_effects.py       # Trực quan hóa chi tiết AMDF
│   └── visualize_preprocessing_effects.py # Trực quan hóa tác dụng của Bandpass & Center Clipping
│
├── predict.py                   # Giao diện CLI dự đoán F0, V/UV và xuất biểu đồ contour 300 DPI
├── outputs/                     # Toàn bộ kết quả đầu ra
│   ├── figures/                 # Hình vẽ chất lượng cao (.png, 300 DPI)
│   ├── reports/                 # Báo cáo chi tiết (.csv, .json)
│   └── predictions/             # Kết quả chạy CLI trên file/thư mục tùy chọn
│
├── README.md                    # Báo cáo kỹ thuật tổng quan và mục lục điều hướng
├── requirements.txt             # Danh sách thư viện Python
└── pyproject.toml               # Cấu hình dự án
```

---

## 3. Mục Lục Hệ Thống Tài Liệu Báo Cáo

Hệ thống tài liệu nghiên cứu chi tiết được chia riêng biệt cho từng thuật toán trong thư mục `docs/`:

1. **[Chương I: Thuật Toán Hàm Tự Tương Quan (ACF)](docs/01_acf_theory_and_experiments.md)**
   * Cơ sở lý thuyết toán học của ACF ($L_2$ norm).
   * Minh họa trực quan khung Hữu thanh vs Vô thanh trên `phone_F1.wav`.
   * Huấn luyện ngưỡng Gauss tối ưu Bayes ($T = 0.4408$ và $T_{\text{BP}} = 0.4892$).
   * Luận điểm khoa học chọn khung chuẩn $25\text{ ms}$ (so sánh 20ms vs 25ms vs 30ms).
   * Bảng kết quả Baseline và phân nhóm tối ưu 16 tổ hợp plugin trên Tập Huấn Luyện (`TinHieuHuanLuyen`).

2. **[Chương II: Thuật Toán Hàm Hiệu Độ Lớn Trung Bình (AMDF)](docs/02_amdf_theory_and_experiments.md)**
   * Cơ sở lý thuyết toán học của AMDF ($L_1$ norm) và chuẩn hóa biên độ.
   * Minh họa khung Voiced vs Unvoiced với đáy cực tiểu (Deep Dip).
   * Huấn luyện ngưỡng Gauss AMDF theo các điều kiện tiền xử lý ($T = 0.4380$).
   * Bảng kết quả Baseline và phân nhóm tối ưu 16 tổ hợp plugin trên Tập Huấn Luyện (`TinHieuHuanLuyen`).
   * So sánh đối chiếu đặc tính kỹ thuật: **ACF vs AMDF**.

3. **[Chương III: Thuật Toán Đồng Cấu Cepstrum & Đối Sánh BT2](docs/03_cepstrum_theory_and_experiments.md)**
   * Cơ sở lý thuyết toán học: Mô hình nguồn - bộ lọc, phép biến đổi đồng cấu Homomorphic Deconvolution (FFT -> Log -> IFFT).
   * Phân tách tự nhiên giữa đường bao Formant thanh đạo ($q < 2\text{ ms}$) và xung thanh môn ($q > 2.5\text{ ms}$).
   * Minh họa trực quan khung Hữu thanh vs Vô thanh trên đồ thị miền Quefrency.
   * Bảng phân nhóm tối ưu 16 tổ hợp plugin cho Cepstrum trên Tập Huấn Luyện (`TinHieuHuanLuyen`).
   * Ma trận đối sánh 3 thuật toán trọng tâm của **Bài Tập 2 (BT2)**: ACF vs AMDF vs Cepstrum.

4. **[Chương IV: Đánh Giá Đối Sánh Trên Tập Kiểm Thử (TinHieuKiemThu) & Phân Tích Hiện Tượng](docs/04_test_set_benchmark_and_analysis.md)**
   * Đánh giá độc lập trên 4 file kiểm thử (`phone_F2`, `phone_M2`, `studio_F2`, `studio_M2`).
   * Bảng tổng hợp đối sánh 3 Quán quân vs 3 Baseline theo chuẩn 8 cột (AMDF Champion xuất sắc dẫn đầu với Score **3.78%**).
   * Bảng phân tích chi tiết từng file âm thanh (`studio_M2` đạt Score **0.84%** và 0 khung lỗi; `studio_F2` chỉ lệch 1 khung duy nhất).
   * Phân tích chuyên sâu 4 hiện tượng âm học cốt lõi:
     1. Suy hao sóng hài cơ bản (Missing Fundamental) do giới hạn băng thông $300 - 3400\text{ Hz}$ trong kênh thoại (`phone_M2`).
     2. Rơi rụng khung biên (Boundary Dropping) và giải pháp hàn gắn hình thái học (Morphological Voiced Gap Filling).
     3. Nhảy quãng tám (Octave Jumps) và cơ chế triệt tiêu bằng Viterbi Trellis Tracking ($w_{\text{octave}} = 4.0$).
     4. So sánh bản chất toán học: AMDF ($L_1$) vs ACF ($L_2$) vs Cepstrum (Quefrency).

---

## 4. Bảng Kết Quả Thực Nghiệm Trên Tập Huấn Luyện (`TinHieuHuanLuyen`)

Toàn bộ các khảo sát tổ hợp plugin, lựa chọn cấu hình và phân nhóm chỉ số được thực hiện nghiêm ngặt trên **Tập Huấn Luyện (TinHieuHuanLuyen)** gồm 4 file chuẩn (`phone_F1.wav`, `phone_M1.wav`, `studio_F1.wav`, `studio_M1.wav`).

Hệ thống đánh giá trên 3 chỉ số mục tiêu và 1 chỉ số tổng hợp:
* **$\text{MAPE } F_0$ (%):** Sai số tương đối trung bình tần số $F_0$.
* **$\text{MAPE Std}$ (%):** Sai số tương đối độ lệch chuẩn đường contour.
* **$\text{MAPE Num}$ (%):** Sai số tương đối số lượng khung được phân loại Hữu thanh (Voiced).
* **$\text{FINAL SCORE}$ (%):** Điểm lỗi tổng hợp $= (\text{MAPE } F_0 + \text{MAPE Std} + \text{MAPE Num}) / 3$ (Càng thấp càng tối ưu).

### 4.1. Bảng phân nhóm tối ưu thuật toán ACF:

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
| **Nhóm 4 (Quán quân toàn diện)**| **BP + Energy + Viterbi** | **1.34 Hz** | **0.64%** | **0.66 Hz** | **2.39%** | **8.2 khung** | **5.83%** | **2.95%** 🏆 |

### 4.2. Bảng phân nhóm tối ưu thuật toán AMDF:

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
| **Nhóm 4 (Quán quân toàn diện)**| **Energy + Viterbi** | **1.30 Hz** | **0.80%** | **0.96 Hz** | **3.42%** | **9.2 khung** | **5.64%** | **3.29%** 🏆 |

### 4.3. Bảng phân nhóm tối ưu thuật toán Cepstrum:

| Phân nhóm chỉ số | Cấu hình Plugin | $\Delta F_0$ (Hz) | MAPE $ (%) | $\Delta\text{Std}$ (Hz) | MAPE Std (%) | $\Delta\text{Num}$ (Khung) | MAPE Num (%) | FINAL SCORE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Gốc)** | Baseline | 13.95 Hz | 10.99% | 19.54 Hz | 91.74% | 18.2 khung | 16.71% | **39.81%** |
| **Nhóm 1 (Top 1 F0)** | BP + Clip + Energy + Viterbi | **1.42 Hz** | **0.87%** | 1.28 Hz | 6.24% | 9.2 khung | 6.20% | **4.44%** 🏆 |
| **Nhóm 1 (Top 2 F0)** | BP + Viterbi | 1.53 Hz | 0.94% | **1.21 Hz** | **5.85%** | 14.2 khung | 9.80% | 5.53% |
| **Nhóm 1 (Top 3 F0)** | BP + Energy + Viterbi | 1.59 Hz | 0.97% | 1.38 Hz | 6.32% | 13.8 khung | 9.46% | 5.59% |
| **Nhóm 2 (Top 1 Std)** | BP + Viterbi | 1.53 Hz | 0.94% | **1.21 Hz** | **5.85%** | 14.2 khung | 9.80% | 5.53% |
| **Nhóm 2 (Top 2 Std)** | BP + Clip + Energy + Viterbi | **1.42 Hz** | **0.87%** | 1.28 Hz | 6.24% | 9.2 khung | 6.20% | **4.44%** 🏆 |
| **Nhóm 2 (Top 3 Std)** | BP + Energy + Viterbi | 1.59 Hz | 0.97% | 1.38 Hz | 6.32% | 13.8 khung | 9.46% | 5.59% |
| **Nhóm 3 (Top 1 Num)** | BP + Clip | 2.63 Hz | 1.69% | 5.51 Hz | 27.68% | **8.0 khung** | **5.16%** | 11.50% |
| **Nhóm 3 (Top 2 Num)** | BP + Clip + Viterbi | 2.12 Hz | 1.44% | 2.50 Hz | 11.47% | **8.0 khung** | **5.16%** | 6.03% |
| **Nhóm 3 (Top 3 Num)** | Clip | 4.22 Hz | 3.24% | 9.63 Hz | 45.77% | **7.8 khung** | **5.37%** | 18.12% |
| **Nhóm 4 (Quán quân toàn diện)**| **BP + Clip + Energy + Viterbi** | **1.42 Hz** | **0.87%** | **1.28 Hz** | **6.24%** | **9.2 khung** | **6.20%** | **4.44%** 🏆 |

---

## 4.4. Đánh Giá Khái Quát Hóa Trên Tập Kiểm Thử (`TinHieuKiemThu`)

Xem chi tiết tại [Chương IV: Đánh Giá Đối Sánh Trên Tập Kiểm Thử (TinHieuKiemThu)](docs/04_test_set_benchmark_and_analysis.md).

| STT | Thuật toán / Cấu hình | $\Delta F_0$ (Hz) | MAPE $F_0$ (%) | $\Delta\text{Std}$ (Hz) | MAPE Std (%) | $\Delta\text{Num}$ (Khung) | MAPE Num (%) | FINAL SCORE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **1. ACF Baseline** | 1.46 Hz | 0.99% | 1.65 Hz | 5.17% | 17.2 khung | 10.47% | **5.55%** |
| 2 | **2. ACF Champion (BP+Energy+Viterbi)** | 1.66 Hz | 1.12% | 1.69 Hz | 5.48% | 10.0 khung | 6.06% | **4.22%** |
| 3 | **3. AMDF Baseline** | **0.68 Hz** | **0.41%** | **1.28 Hz** | **3.50%** | 14.2 khung | 8.15% | **4.02%** |
| 4 | **4. AMDF Champion (Energy+Viterbi)** | 1.19 Hz | 0.78% | 1.65 Hz | 5.54% | **9.0 khung** | **5.02%** | **3.78%** 🏆 |
| 5 | **5. Cepstrum Baseline** | 17.23 Hz | 12.05% | 19.27 Hz | 98.55% | 23.2 khung | 18.24% | **42.95%** |
| 6 | **6. Cepstrum Champion (BP+Clip+Energy+Viterbi)** | 1.66 Hz | 1.02% | 2.44 Hz | 11.39% | 14.0 khung | 10.43% | **7.62%** |

---

## 5. Hướng Dẫn Thực Thi Dự Án

### 5.1. Cài đặt môi trường bằng `uv` (Khuyên dùng)
```powershell
# 1. Khởi tạo môi trường ảo với Python 3.12
uv venv --python 3.12

# 2. Cài đặt các gói phụ thuộc ở chế độ editable
uv pip install -e .
```

### 5.2. Chạy các kịch bản thực nghiệm tự động

#### 1. Xem minh họa khung Voiced vs Unvoiced (Chuẩn 25 ms):
```powershell
uv run python scripts/01_demo_frames.py --method acf
uv run python scripts/01_demo_frames.py --method amdf
uv run python scripts/01_demo_frames.py --method cepstrum
```

#### 2. Huấn luyện tìm ngưỡng Gauss tối ưu Bayes:
```powershell
uv run python scripts/02_train_threshold.py               # ACF thô (T = 0.4408)
uv run python scripts/02_train_threshold.py --method amdf # AMDF thô (T = 0.4380)
```

#### 3. Khảo sát chiều dài khung phân tích (20 ms vs 25 ms vs 30 ms):
```powershell
uv run python scripts/03_compare_params.py
```

#### 4. Khảo sát 16 tổ hợp plugin & phân 4 nhóm tối ưu:
```powershell
uv run python scripts/05_compare_plugins.py --method acf       # ACF trên tập Train
uv run python scripts/05_compare_plugins.py --method amdf      # AMDF trên tập Train
uv run python scripts/05_compare_plugins.py --method cepstrum  # Cepstrum trên tập Train
```

#### 5. Đối sánh toàn diện 3 thuật toán Bài Tập 2 (ACF vs AMDF vs Cepstrum):
```powershell
# Chạy đối sánh trên Tập Huấn Luyện (TinHieuHuanLuyen):
uv run python scripts/14_compare_acf_amdf_cepstrum.py --dataset TinHieuHuanLuyen

# Chạy đối sánh trên Tập Kiểm Thử (TinHieuKiemThu):
uv run python scripts/14_compare_acf_amdf_cepstrum.py --dataset TinHieuKiemThu
```

#### 6. Công cụ CLI Dự đoán độc lập & Trực quan hóa (`predict.py`):
```powershell
# Dự đoán toàn bộ thư mục âm thanh với bộ 3 Quán quân tốt nhất
uv run python predict.py -i TinHieuHuanLuyen -m champions

# Dự đoán cho một file âm thanh đơn lẻ
uv run python predict.py -i TinHieuHuanLuyen/phone_F1.wav -m champions

# Lựa chọn các chế độ thuật toán qua tham số -m:
#   champions : Chạy 3 Quán quân (ACF, AMDF, Cepstrum) [Mặc định]
#   acf       : Chạy ACF Champion vs ACF Baseline
#   amdf      : Chạy AMDF Champion vs AMDF Baseline
#   cepstrum  : Chạy Cepstrum Champion vs Cepstrum Baseline
#   baseline  : Chạy cả 3 Baseline gốc
#   all       : Chạy toàn bộ các mô hình
```
* File thống kê tổng hợp: `outputs/predictions/evaluation_summary_metrics.csv`
* File CSV chi tiết từng khung: `outputs/predictions/<tên_file>_pitch_prediction.csv`
* Đồ thị trực quan 300 DPI: `outputs/predictions/<tên_file>_pitch_contour.png`
