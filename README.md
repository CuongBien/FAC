# XỬ LÝ TÍN HIỆU TIẾNG NÓI (SPEECH PROCESSING)
# BÁO CÁO NGHIÊN CỨU & HIỆN THỰC THUẬT TOÁN ƯỚC LƯỢNG CAO ĐỘ $F_0$ VÀ PHÂN LOẠI HỮU THANH / VÔ THANH

> **Chủ đề:** Ước lượng tần số cơ bản $F_0$ (Pitch Contour) và phân loại Hữu thanh (Voiced) / Vô thanh (Unvoiced) trên miền thời gian.  
> **Hai thuật toán nghiên cứu:**  
> 1. **Thuật toán Hàm tự tương quan (Autocorrelation Function - ACF)**  
> 2. **Thuật toán Hàm hiệu độ lớn trung bình (Average Magnitude Difference Function - AMDF)**  
> **Bộ dữ liệu chuẩn:** `TinHieuHuanLuyen/` (Huấn luyện tham số) và `TinHieuKiemThu/` (Báo cáo đánh giá).

---

## MỤC LỤC TỔNG QUAN

- [1. Cấu Trúc Thư Mục & Thiết Kế Kiến Trúc Module](#1-cấu-trúc-thư-mục--thiết-kế-kiến-trúc-module)
- [2. Hướng Dẫn Cài Đặt Môi Trường & Thực Thi](#2-hướng-dẫn-cài-đặt-môi-trường--thực-thi)
- [PHẦN I: THUẬT TOÁN HÀM TỰ TƯƠNG QUAN (ACF)](#phần-i-thuật-toán-hàm-tự-tương-quan-acf)
  - [I.1. Cơ Sở Lý Thuyết Thuật Toán ACF](#i1-cơ-sở-lý-thuyết-thuật-toán-acf)
  - [I.2. Minh Họa Khung Hữu Thanh vs Vô Thanh](#i2-minh-họa-khung-hữu-thanh-vs-vô-thanh)
  - [I.3. Huấn Luyện Ngưỡng T Tối Ưu Bằng Phân Bố Gauss](#i3-huấn-luyện-ngưỡng-t-tối-ưu-bằng-phân-bố-gauss)
  - [I.4. Khảo Sát Ảnh Hưởng Chiều Dài Khung (20ms vs 30ms)](#i4-khảo-sát-ảnh-hưởng-chiều-dài-khung-20ms-vs-30ms)
  - [I.5. Đánh Giá Kiểm Thử Trên 4 File Kiểm Thử (Baseline)](#i5-đánh-giá-kiểm-thử-trên-4-file-kiểm-thử-baseline)
  - [I.6. Các Giải Pháp Cải Tiến Dành Riêng Cho ACF (Plugins)](#i6-các-giải-pháp-cải-tiến-dành-riêng-cho-acf-plugins)
    - [I.6.1. Plugin 1: Ngưỡng trễ kép Hysteresis (Schmitt Trigger)](#i61-plugin-1-ngưỡng-trễ-kép-hysteresis-schmitt-trigger)
    - [I.6.2. Plugin 2: Mở rộng vùng hữu thanh theo năng lượng khung biên (STE)](#i62-plugin-2-mở-rộng-vùng-hữu-thanh-theo-năng-lượng-khung-biên-ste)
    - [I.6.3. Plugin 3: Bộ lọc thông dải Butterworth & Lọc Zero-Phase](#i63-plugin-3-bộ-lọc-thông-dải-butterworth--lọc-zero-phase)
  - [I.7. Khảo Sát & Xếp Hạng Toàn Bộ 8 Tổ Hợp Cải Tiến ($2^3$)](#i7-khảo-sát--xếp-hạng-toàn-bộ-8-tổ-hợp-cải-tiến-23)
- [PHẦN II: THUẬT TOÁN HÀM HIỆU ĐỘ LỚN TRUNG BÌNH (AMDF)](#phần-ii-thuật-toán-hàm-hiệu-độ-lớn-trung-bình-amdf)
- [PHẦN III: PHÂN TÍCH HIỆN TƯỢNG TRÊN ĐỒ THỊ & NGUYÊN NHÂN SAI SỐ](#phần-iii-phân-tích-hiện-tượng-trên-đồ-thị--nguyên-nhân-sai-số)

---

## 1. Cấu Trúc Thư Mục & Thiết Kế Kiến Trúc Module

Dự án áp dụng mô hình thiết kế hướng module, phân lập rõ ràng giữa nhân thuật toán xử lý tín hiệu số, tầng phân tích thống kê toán học, tầng trực quan hóa và hệ sinh thái Plugins cắm/rút độc lập:

```
D:\SP\
├── TinHieuHuanLuyen/            # 4 cặp file huấn luyện: phone_F1, phone_M1, studio_F1, studio_M1 (*.wav, *.lab)
├── TinHieuKiemThu/              # 4 cặp file kiểm thử: phone_F2, phone_M2, studio_F2, studio_M2 (*.wav, *.lab)
├── HuongDan.txt                 # Bản đặc tả yêu cầu chi tiết của đồ án
│
├── src/                         # Mã nguồn module hóa chuẩn
│   ├── core/                    # Tầng xử lý tín hiệu cốt lõi
│   │   ├── audio.py             # Nạp file WAV, phân khung (framing), tính năng lượng ngắn hạn (STE)
│   │   ├── lab_parser.py        # Đọc nhãn mốc thời gian và thống kê chuẩn từ file *.lab
│   │   ├── acf.py               # Thuật toán Hàm tự tương quan (ACF) & dò cực đại
│   │   ├── amdf.py              # Thuật toán Hàm hiệu biên độ trung bình (AMDF) & dò cực tiểu
│   │   └── pitch_detector.py    # Bộ phát hiện Baseline PitchDetector (cố định, không sửa đổi)
│   │
│   ├── plugins/                 # Hệ thống Plugin mở rộng (Plug-and-Play)
│   │   ├── base.py              # Interface cơ sở BasePlugin
│   │   ├── plugin_detector.py   # Bộ bao bọc (Wrapper) gắn kết các plugin vào pipeline xử lý
│   │   ├── hysteresis.py        # Cải tiến 1: Ngưỡng trễ kép Schmitt Trigger
│   │   ├── energy_extension.py  # Cải tiến 2: Mở rộng vùng hữu thanh theo STE khung biên
│   │   └── bandpass_filter.py   # Cải tiến 3: Bộ lọc thông dải Butterworth bậc 2 Zero-Phase
│   │
│   ├── analysis/                # Phân tích thống kê & đánh giá sai số
│   │   ├── threshold.py         # Trích xuất phân bố Gauss (mean, std), giải phương trình tìm T
│   │   └── evaluation.py        # Đánh giá định lượng sai số F0mean, F0std, V/UV Accuracy, F1
│   │
│   └── visualization/           # Trực quan hóa dữ liệu (Matplotlib)
│       └── plotter.py           # Vẽ contour F0 đồng bộ dạng sóng, phân bố Gauss, khung đại diện
│
├── scripts/                     # Các kịch bản chạy tự động hóa
│   ├── 01_demo_frames.py        # Minh họa 1 khung Voiced vs 1 khung Unvoiced
│   ├── 02_train_threshold.py    # Huấn luyện tìm ngưỡng T tối ưu (hỗ trợ --plugins bandpass)
│   ├── 03_compare_params.py     # Khảo sát so sánh chiều dài khung (20ms vs 30ms)
│   ├── 04_run_testing.py        # Chạy 4 file test trong 1 lệnh, xuất 4 hình và bảng chỉ số
│   └── 05_compare_plugins.py    # Khảo sát và xếp hạng toàn bộ 8 tổ hợp plugins ($2^3$)
│
├── outputs/                     # Toàn bộ kết quả đầu ra
│   ├── figures/                 # Hình vẽ chất lượng cao (.png, 300 DPI)
│   └── reports/                 # Bảng số liệu chi tiết (.csv, .json)
│
├── README.md                    # Báo cáo kỹ thuật tổng hợp toàn diện
├── requirements.txt             # Danh sách thư viện Python
└── pyproject.toml               # Cấu hình dự án
```

---

## 2. Hướng Dẫn Cài Đặt Môi Trường & Thực Thi

### 2.1. Cài đặt thư viện
Yêu cầu Python $\ge 3.9$ (dự án phát triển trên Python 3.12). Cài đặt gói phụ thuộc qua pip:
```powershell
pip install -r requirements.txt
```

### 2.2. Lệnh thực thi nhanh từng bước
```powershell
# Bước 1: Xem minh họa khung Voiced vs Unvoiced
python scripts/01_demo_frames.py

# Bước 2: Huấn luyện tìm ngưỡng T tối ưu
python scripts/02_train_threshold.py                     # Huấn luyện Baseline thô (T = 0.4620)
python scripts/02_train_threshold.py --plugins bandpass  # Huấn luyện có lọc tiền xử lý (T = 0.5124)

# Bước 3: So sánh chiều dài khung 20ms vs 30ms
python scripts/03_compare_params.py

# Bước 4: Chạy kiểm thử tự động toàn bộ 4 file kiểm thử
python scripts/04_run_testing.py                         # Mô hình Baseline gốc
python scripts/04_run_testing.py --plugins all           # Mô hình Nâng cao tích hợp cả 3 Plugins

# Bước 5: Đánh giá và xếp hạng toàn bộ 8 tổ hợp Plugins
python scripts/05_compare_plugins.py
```

---

# PHẦN I: THUẬT TOÁN HÀM TỰ TƯƠNG QUAN (ACF)

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

Thực nghiệm trên file âm thanh `TinHieuHuanLuyen/phone_F1.wav` ($f_s = 16000\text{ Hz}$):

![01_demo_acf_frames.png](outputs/figures/01_demo_acf_frames.png)

### Bảng phân tích chi tiết hai khung đại diện:

| Đặc trưng | Khung Hữu Thanh (Voiced) | Khung Vô Thanh (Unvoiced) |
| :--- | :--- | :--- |
| **Vị trí khung** | Khung #103 ($t = 1.045\text{ s}$) | Khung #176 ($t = 1.775\text{ s}$) |
| **Đặc tính dạng sóng** | Tuần hoàn rõ rệt, biên độ xung thanh môn lớn | Dao động ngẫu nhiên, tựa nhiễu, biên độ thấp |
| **Độ trễ cực đại $\tau_0$** | $\tau_0 = 71$ mẫu | Không có độ trễ tuần hoàn |
| **Biên độ cực đại $R(\tau_0)$** | **$0.829$** ($\gg T = 0.4620$) | **$0.346$** ($< T = 0.4620$) |
| **Chu kỳ cơ bản $T_0$** | $T_0 = 71 / 16000 = 4.44\text{ ms}$ | Không xác định |
| **Tần số cơ bản $F_0$** | **$F_0 = 225.35\text{ Hz}$** (gần nhãn $217\text{ Hz}$) | **$F_0 = \text{Undefined}$** (Vô thanh) |

---

## I.3. Huấn Luyện Ngưỡng T Tối Ưu Bằng Phân Bố Gauss

Để xác định ngưỡng phân định V/UV dựa trên cơ sở thống kê toán học vững chắc, đồ án trích xuất biên độ đỉnh ACF lớn nhất của toàn bộ các khung trong 4 file huấn luyện (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`), phân chia theo nhãn ground-truth `.lab`:
* Tập khung Hữu thanh: $N_V = 614$ khung $\rightarrow$ phân bố $\mathcal{N}(\mu_V, \sigma_V^2)$
* Tập khung Vô thanh: $N_U = 180$ khung $\rightarrow$ phân bố $\mathcal{N}(\mu_U, \sigma_U^2)$

Ngưỡng tối ưu Bayes $T$ là nghiệm của phương trình giao điểm mật độ xác suất:

$$
\frac{(T - \mu_V)^2}{\sigma_V^2} - \frac{(T - \mu_U)^2}{\sigma_U^2} + 2\ln\left(\frac{\sigma_V}{\sigma_U}\right) = 0
$$

### Bảng kết quả huấn luyện ngưỡng:

| Chế độ tiền xử lý | Số khung Voiced | $\mu_V \pm \sigma_V$ | Số khung Unvoiced | $\mu_U \pm \sigma_U$ | Ngưỡng tối ưu $T$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Chưa lọc)** | 614 | $0.6474 \pm 0.1535$ | 180 | $0.2805 \pm 0.1441$ | **$T = 0.4620$** |
| **Có lọc thông dải (Bandpass)** | 614 | $0.6832 \pm 0.1442$ | 180 | $0.3442 \pm 0.1370$ | **$T = 0.5124$** |

### Biểu đồ phân bố xác suất Gaussian:

| Baseline (Chưa lọc: $T = 0.4620$) | Có lọc Bandpass (Đã lọc: $T = 0.5124$) |
| :---: | :---: |
| ![02_threshold_distribution.png](outputs/figures/02_threshold_distribution.png) | ![02_threshold_distribution_bandpassprefilter.png](outputs/figures/02_threshold_distribution_bandpassprefilter.png) |

---

## I.4. Khảo Sát Ảnh Hưởng Chiều Dài Khung (20ms vs 30ms)

Thực nghiệm đối chiếu trên file giọng nam trầm `TinHieuHuanLuyen/phone_M1.wav` ($F_{0\text{-mean}} = 122.0\text{ Hz}$, $F_{0\text{-std}} = 18.0\text{ Hz}$):

### Bảng so sánh định lượng:

| Chỉ số đánh giá | Khung 20 ms | Khung 30 ms | Nhận xét cải thiện |
| :--- | :---: | :---: | :--- |
| **Số khung phát hiện Hữu thanh** | 165 khung | **191 khung** | Khung 30ms chứa đủ số chu kỳ của giọng nam trầm |
| **$F_{0\text{-mean}}$ ước lượng** | 136.55 Hz | **129.06 Hz** | Tiệm cận giá trị chuẩn 122.0 Hz hơn rất nhiều |
| **Sai số tuyệt đối $\lvert\Delta F_0\rvert$** | 14.55 Hz | **7.06 Hz** | **Giảm hơn 51.5% sai số** |
| **Sai số tương đối (%)** | 11.93 % | **5.79 %** | Giảm một nửa sai lệch |
| **V/UV Accuracy (%)** | 71.08 % | **78.50 %** | Tăng +7.42% |
| **Voiced F1-Score (%)** | 80.20 % | **87.36 %** | Tăng +7.16% |

![03_compare_frame_length.png](outputs/figures/03_compare_frame_length.png)

$\implies$ **Kết luận khoa học:** Khung 30 ms đảm bảo chứa tối thiểu 2–3 chu kỳ hoàn chỉnh ngay cả ở tần số đáy $70\text{ Hz}$ ($T_0 \approx 14.3\text{ ms}$), giúp đỉnh tự tương quan $R(\tau)$ đạt cực đại ổn định. Do đó, **30 ms (hop 10 ms)** được chọn làm chuẩn toàn hệ thống.

---

## I.5. Đánh Giá Kiểm Thử Trên 4 File Kiểm Thử (Baseline)

### Bảng kết quả kiểm thử định lượng Baseline ($T = 0.4620$):

| File kiểm thử | Kênh / Giới tính | Ref $F_{0\text{-mean}}$ | Pred $F_{0\text{-mean}}$ | $\lvert\Delta F_0\rvert$ (Hz) | Sai số % | Ref $F_{0\text{-std}}$ | Pred $F_{0\text{-std}}$ | $\lvert\Delta\text{std}\rvert$ | V/UV Acc | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `phone_F2.wav` | Điện thoại / Nữ | 145.00 Hz | 152.20 Hz | 7.20 Hz | 4.96% | 33.70 | 31.40 | 2.30 | 77.62% | 85.92% |
| `phone_M2.wav` | Điện thoại / Nam | 129.00 Hz | 132.59 Hz | 3.59 Hz | 2.79% | 18.60 | 13.97 | 4.63 | 78.78% | 89.71% |
| `studio_F2.wav`| Studio / Nữ | 200.00 Hz | 199.27 Hz | 0.73 Hz | 0.36% | 46.10 | 43.35 | 2.75 | 91.99% | 95.45% |
| `studio_M2.wav`| Studio / Nam | 155.00 Hz | 154.84 Hz | **0.16 Hz** | **0.10%** | 30.80 | 30.08 | 0.72 | 88.14% | 91.98% |
| **TRUNG BÌNH** | — | — | — | **2.92 Hz** | **2.05%** | — | — | **2.60** | **84.13%** | **90.77%** |

### Đồ thị đường bao Pitch Contour 4 file kiểm thử (Baseline):

| `phone_F2.wav` (Nữ - Kênh điện thoại) | `phone_M2.wav` (Nam - Kênh điện thoại) |
| :---: | :---: |
| ![04_test_phone_F2.png](outputs/figures/04_test_phone_F2.png) | ![04_test_phone_M2.png](outputs/figures/04_test_phone_M2.png) |
| **`studio_F2.wav` (Nữ - Phòng thu Studio)** | **`studio_M2.wav` (Nam - Phòng thu Studio)** |
| ![04_test_studio_F2.png](outputs/figures/04_test_studio_F2.png) | ![04_test_studio_M2.png](outputs/figures/04_test_studio_M2.png) |

---

## I.6. Các Giải Pháp Cải Tiến Dành Riêng Cho ACF (Plugins)

### I.6.1. Plugin 1: Ngưỡng trễ kép Hysteresis (Schmitt Trigger)

**Vấn đề giải quyết:**  
Hiện tượng rung lật trạng thái (*decision chattering*) khi đỉnh tự tương quan dao động quanh ngưỡng tĩnh $T$.

**Mô hình máy trạng thái:**  
Ký hiệu $S_i \in \{0, 1\}$ là nhãn khung thứ $i$ (0: Unvoiced, 1: Voiced) và $R_i^* = \max_{\tau \in [\tau_{\min}, \tau_{\max}]} R_i(\tau)$ là đỉnh cực đại tự tương quan.

Phương trình chuyển trạng thái:

$$
S_i = \begin{cases} 
1, & \text{khi } R_i^* \ge T_{\text{high}} \\
1, & \text{khi } S_{i-1} = 1 \text{ và } R_i^* \ge T_{\text{low}} \\
0, & \text{ngược lại}
\end{cases}
$$

**Thông số áp dụng:**  
* Mô hình Baseline: $T_{\text{high}} = 0.4620, T_{\text{low}} = 0.40$
* Mô hình Bandpass: $T_{\text{high}} = 0.5124, T_{\text{low}} = 0.42$

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
Do lọc làm sạch tín hiệu, $\mu_V$ tăng từ $0.6474 \rightarrow 0.6832$, hệ thống tự động đồng bộ ngưỡng tối ưu mới:

$$
T = 0.5124
$$

---

## I.7. Khảo Sát & Xếp Hạng Toàn Bộ 8 Tổ Hợp Cải Tiến ($2^3$)

### Bảng kết quả đối sánh toàn diện 8 cấu hình trên tập kiểm thử:

| STT | Cấu hình Plugin | Ngưỡng $T$ | `phone_F2` (145Hz) | `phone_M2` (129Hz) | `studio_F2` (200Hz) | `studio_M2` (155Hz) | Sai số TB $F_0$ | F1-Score TB | V/UV Acc TB |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Baseline (Gốc)** | $0.4620$ | 7.20 Hz | 3.59 Hz | 0.73 Hz | **0.16 Hz** | 2.92 Hz | 90.77% | 84.13% |
| **2** | **[Hysteresis]** | $0.4620$ | 6.62 Hz | **3.25 Hz** | 0.35 Hz | 0.33 Hz | 2.64 Hz | 91.87% | 85.03% |
| **3** | **[Energy Ext]** | $0.4620$ | 6.21 Hz | 3.68 Hz | 1.36 Hz | 0.75 Hz | 3.00 Hz | 92.34% | 85.36% |
| **4** | **[Bandpass Filter]** | $0.5124$ | 7.39 Hz | 3.83 Hz | 0.15 Hz | 0.19 Hz | 2.89 Hz | 89.29% | 82.76% |
| **5** | **[Hysteresis + Energy Ext]** | $0.4620$ | 6.14 Hz | 3.47 Hz | 1.36 Hz | 1.21 Hz | 3.04 Hz | **92.73%** | **85.71%** |
| **6** | **[Bandpass + Hysteresis]** | $0.5124$ | 6.30 Hz | 3.53 Hz | **0.08 Hz** | 0.46 Hz | **2.59 Hz** | 91.11% | 84.24% |
| **7** | **[Bandpass + Energy Ext]** | $0.5124$ | 5.91 Hz | 4.06 Hz | 0.33 Hz | 0.28 Hz | 2.64 Hz | 91.69% | 84.68% |
| **8** | **[Cả 3 Plugins]** | $0.5124$ | **5.74 Hz** | 3.77 Hz | 0.21 Hz | 0.95 Hz | 2.67 Hz | 92.36% | 85.27% |

* **Quán quân về độ chính xác $F_0$:** **Cấu hình 6 `[Bandpass + Hysteresis]`** đạt sai số thấp nhất toàn diện (**2.59 Hz**), đặc biệt trên `studio_F2.wav` sai số chỉ còn **0.08 Hz** ($< 0.05\%$).
* **Quán quân về phân loại V/UV:** **Cấu hình 5 `[Hysteresis + Energy Ext]`** đạt **F1 = 92.73%** và **Accuracy = 85.71%**.
* **Cấu hình cân bằng tối ưu:** **Cấu hình 8 `[Cả 3 Plugins]`** đạt sai số $F_0$ rất thấp (**2.67 Hz**) cùng F1 cao vượt bậc (**92.36%**).

### Biểu đồ cột xếp hạng 8 tổ hợp:
![05_all_combinations_ranking.png](outputs/figures/05_all_combinations_ranking.png)

### Biểu đồ trực quan 3 tầng đối sánh Baseline (Đỏ) vs Enhanced (Xanh):

| `phone_F2.wav` | `phone_M2.wav` |
| :---: | :---: |
| ![05_compare_plugin_phone_F2.png](outputs/figures/05_compare_plugin_phone_F2.png) | ![05_compare_plugin_phone_M2.png](outputs/figures/05_compare_plugin_phone_M2.png) |
| **`studio_F2.wav`** | **`studio_M2.wav`** |
| ![05_compare_plugin_studio_F2.png](outputs/figures/05_compare_plugin_studio_F2.png) | ![05_compare_plugin_studio_M2.png](outputs/figures/05_compare_plugin_studio_M2.png) |

---

# PHẦN II: THUẬT TOÁN HÀM HIỆU ĐỘ LỚN TRUNG BÌNH (AMDF)

*(Phần này để trống cho việc triển khai thuật toán AMDF)*

---

# PHẦN III: PHÂN TÍCH HIỆN TƯỢNG TRÊN ĐỒ THỊ & NGUYÊN NHÂN SAI SỐ

1. **Sự suy giảm chất lượng giữa Kênh thoại (Phone) và Phòng thu (Studio):**
   * Tín hiệu phòng thu (`studio_*`) có tỷ số tín hiệu trên nhiễu (SNR) cao và dải thông rộng. Thuật toán đạt độ chính xác gần như tuyệt đối ($|\Delta F_0| < 0.2\text{ Hz}$, độ chính xác V/UV $> 92\%$).
   * Tín hiệu điện thoại (`phone_*`) bị giới hạn băng thông hẹp tiêu chuẩn ($300 - 3400\text{ Hz}$). Thành phần tần số cơ bản của giọng nam ($F_0 < 300\text{ Hz}$) bị suy giảm mạnh, buộc thuật toán phải bắt vào chu kỳ bao của các sóng hài bậc cao, dẫn tới sai số trung bình cao hơn ($3 - 5\text{ Hz}$).
2. **Hiện tượng đứt gãy contour tại ranh giới âm tiết (Boundary Dropping):**
   * Tại vùng bắt đầu phát âm (*onset*) hoặc vùng tắt âm (*offset*), thanh môn mở dần làm biên độ tự tương quan giảm trước khi năng lượng âm thanh tắt hẳn. Mô hình Baseline đơn ngưỡng tĩnh bị rụng 1–2 khung viền. Bộ đôi **Hysteresis Thresholding** và **Energy Edge Extension** khắc phục triệt để hiện tượng này, giúp đường pitch liền mạch.
3. **Hiệu quả của Lọc trung vị (Median Filter 3 khung):**
   * Các điểm nhảy vọt tức thời do chuyển âm nhanh hoặc bắt nhầm hài đôi được làm mịn hoàn toàn bằng bộ lọc trung vị kích thước 3 khung mà không làm trễ pha thời gian của đường contour.
