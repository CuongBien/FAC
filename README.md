# XỬ LÝ TÍN HIỆU TIẾNG NÓI (SPEECH PROCESSING)
# BÁO CÁO NGHIÊN CỨU & HIỆN THỰC THUẬT TOÁN ƯỚC LƯỢNG CAO ĐỘ $F_0$ VÀ PHÂN LOẠI HỮU THANH / VÔ THANH

> **Chủ đề:** Ước lượng tần số cơ bản $F_0$ (Pitch Contour Tracking) và phân loại Hữu thanh (Voiced) / Vô thanh (Unvoiced) trên miền thời gian.  
> **Ba thuật toán nghiên cứu:**  
> 1. **Thuật toán Hàm tự tương quan (Autocorrelation Function - ACF)**  
> 2. **Thuật toán Hàm hiệu độ lớn trung bình (Average Magnitude Difference Function - AMDF)**  
> 3. **Thuật toán YIN (YIN Pitch Tracking Algorithm - de Cheveigné & Kawahara 2002)**  
> **Bộ dữ liệu chuẩn:**  
> * Bộ dữ liệu môn học tiếng Việt: `TinHieuHuanLuyen/` và `TinHieuKiemThu/` ($f_s = 16\text{ kHz}$).  
> * Bộ cơ sở dữ liệu mở chuẩn quốc tế: **PTDB-TUG** ($f_s = 48\text{ kHz}$, có nhãn Ground Truth đo bằng điện cực thanh quản Laryngograph EGG).

---

## 1. Tổng Quan Dự Án & Điểm Nhấn Khoa Học

Dự án hiện thực toàn diện hệ thống ước lượng cao độ tiếng nói và phân loại âm hữu thanh/vô thanh trên miền thời gian, kết hợp giữa mô hình phân tích thống kê xác suất (Phân bố Gauss), hệ sinh thái plugin nâng cao 3 chặng và thuật toán kinh điển thế hệ mới YIN:

* **Chuẩn hóa thông số khung phân tích:** Chốt độ dài khung chuẩn **$25\text{ ms}$** (400 mẫu tại $16\text{ kHz}$) và độ dịch khung **$10\text{ ms}$** (160 mẫu) theo nguyên lý bao phủ tối thiểu $2$ chu kỳ âm vực nam trầm ($80\text{ Hz}$).
* **Huấn luyện ngưỡng tối ưu Gauss (Bayesian Decision Rule):** Giải phương trình giao điểm mật độ xác suất của 2 phân bố $\mathcal{N}(\mu_V, \sigma_V^2)$ và $\mathcal{N}(\mu_U, \sigma_U^2)$ trên hơn 790 khung dữ liệu huấn luyện, tìm ra ngưỡng khách quan $T_{\text{ACF}} = 0.4408$ và $T_{\text{AMDF}} = 0.4380$.
* **Hệ sinh thái 5 Plugins cắm/rút độc lập:**
  1. *Tiền xử lý (Pre-processing):* Bộ lọc thông dải Butterworth bậc 2 Zero-Phase ($70 - 900\text{ Hz}$) và Cắt gọt biên độ trung tâm Center Clipping (Sondhi 1968) làm phẳng phổ.
  2. *Quyết định (Decision):* Ngưỡng trễ kép Hysteresis (Schmitt Trigger) chống rung lật trạng thái.
  3. *Hậu xử lý (Post-processing):* Mở rộng vùng hữu thanh theo năng lượng khung biên STE và Quy hoạch động Viterbi Tracking nắn đường contour tối ưu toàn cục.
* **Khảo sát toàn bộ 32 tổ hợp Plugins ($2^5$):** Cấu hình ACF tối ưu giảm **$41.3\%$** sai số ($3.17\text{ Hz} \rightarrow 1.86\text{ Hz}$); cấu hình AMDF tối ưu đạt sai số trung bình chỉ **$1.50\text{ Hz}$** (và $1.24\text{ Hz}$ khi lọc tiền xử lý).
* **Triển khai chuẩn mực Thuật toán YIN & Đối sánh 3 Quán quân:** Thuật toán YIN kết hợp chuẩn hóa tích lũy trung bình (CMND) và nội suy parabol dưới mức mẫu, đạt sai số thấp kỷ lục **$1.27\text{ Hz}$** (toàn diện) và **$1.08\text{ Hz}$** (kênh thoại), vượt trội cả ACF Champion ($1.86\text{ Hz}$) và AMDF Champion ($1.50\text{ Hz}$).
* **Khám phá điểm giao thoa kháng nhiễu (Noise Robustness Cross-over):** Chứng minh trên cả lý thuyết toán học (Chuẩn $L_2$ vs $L_1$ vs $L_2^2 + \text{CMND}$) và thực nghiệm: YIN dẫn đầu ở dải sạch đến nhiễu nhẹ ($\text{SNR} \ge 10\text{ dB}$), AMDF suy sụp nhanh khi SNR giảm, và ACF chiếm ưu thế tuyệt đối khi nhiễu cực đoan ($\text{SNR} \le 0\text{ dB}$).
* **Kiểm chứng chuẩn quốc tế trên PTDB-TUG (EGG Ground Truth):** 
  * Enhanced ACF triệt tiêu **$86.2\%$** lỗi nhảy quãng tám (GPE) trên giọng nam ($6.09\% \rightarrow 0.60\%$).
  * Thử nghiệm huấn luyện chéo (Cross-Dataset Training) chứng minh ngưỡng tối ưu thu được từ tập tiếng Việt và tập EGG quốc tế gần như trùng khớp hoàn toàn (độ lệch $< 0.013$).

---

## 2. Cấu Trúc Thư Mục & Thiết Kế Kiến Trúc Module

Dự án áp dụng mô hình thiết kế hướng module, phân lập rõ ràng giữa nhân thuật toán xử lý tín hiệu số, tầng phân tích thống kê toán học, tầng trực quan hóa và hệ thống tài liệu chuyên sâu:

```
D:\SP\
├── TinHieuHuanLuyen/            # Dữ liệu huấn luyện tiếng Việt (phone_F1, phone_M1, studio_F1, studio_M1)
├── TinHieuKiemThu/              # Dữ liệu kiểm thử tiếng Việt (phone_F2, phone_M2, studio_F2, studio_M2)
├── HuongDan.txt                 # Đặc tả yêu cầu chi tiết của đồ án
│
├── docs/                        # Hệ thống tài liệu báo cáo nghiên cứu chuyên sâu
│   ├── 01_acf_theory_and_experiments.md     # Chương I: Thuật toán ACF & 32 tổ hợp
│   ├── 02_amdf_theory_and_experiments.md    # Chương II: Thuật toán AMDF & Đối sánh ACF vs AMDF
│   ├── 03_error_analysis_and_phenomena.md   # Chương III: Phân tích hiện tượng sai số trên đồ thị
│   ├── 04_yin_theory_and_benchmarks.md       # Chương IV: Thuật toán YIN & Đối sánh 3 Quán quân
│   ├── 05_noise_robustness_study.md         # Chương V: Khảo sát chuyên sâu độ bền vững kháng nhiễu
│   └── 06_ptdb_tug_international_benchmark.md # Chương VI: Đánh giá chuẩn quốc tế PTDB-TUG (EGG)
│
├── src/                         # Mã nguồn module hóa
│   ├── core/                    # Tầng xử lý tín hiệu cốt lõi
│   │   ├── audio.py             # Nạp file WAV, phân khung (framing), tính năng lượng ngắn hạn (STE)
│   │   ├── lab_parser.py        # Đọc nhãn mốc thời gian và thống kê chuẩn từ file *.lab
│   │   ├── ptdb_loader.py       # Bộ đọc dữ liệu chuẩn quốc tế PTDB-TUG (.wav & .f0 EGG ground truth)
│   │   ├── acf.py               # Thuật toán Hàm tự tương quan (ACF) & dò cực đại
│   │   ├── amdf.py              # Thuật toán Hàm hiệu biên độ trung bình (AMDF) & dò cực tiểu
│   │   ├── yin.py               # Thuật toán YIN (Hiệu bình phương, CMND, Ngưỡng tuyệt đối, Nội suy Parabol)
│   │   └── pitch_detector.py    # Bộ phát hiện Baseline PitchDetector (hỗ trợ acf, amdf, yin)
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
│   │   └── evaluation.py        # Đánh giá định lượng F0, V/UV Accuracy, F1, VDE, GPE, FFE, FPE
│   │
│   └── visualization/           # Trực quan hóa dữ liệu
│       └── plotter.py           # Xuất đồ thị 300 DPI (Contour, Gaussian, Benchmark, Noise curves)
│
├── scripts/                     # Kịch bản chạy tự động hóa hoàn chỉnh
│   ├── 01_demo_frames.py        # Minh họa 1 khung Voiced vs 1 khung Unvoiced
│   ├── 02_train_threshold.py    # Huấn luyện tìm ngưỡng T tối ưu bằng phân bố Gauss
│   ├── 03_compare_params.py     # Khảo sát so sánh chiều dài khung (20ms vs 25ms vs 30ms)
│   ├── 04_run_testing.py        # Chạy 4 file test trong 1 lệnh, xuất đồ thị và bảng chỉ số
│   ├── 05_compare_plugins.py    # Khảo sát và xếp hạng toàn bộ 32 tổ hợp plugins ($2^5$)
│   ├── 06_noise_robustness.py   # Khảo sát chuyên sâu độ bền vững kháng nhiễu (AWGN 25dB đến -5dB)
│   ├── download_ptdb_tug.py     # Tải tập dữ liệu chuẩn quốc tế PTDB-TUG (Audio + EGG ground truth)
│   ├── 07_benchmark_ptdb.py     # Đánh giá benchmark chuẩn quốc tế (VDE, GPE, FFE, FPE)
│   ├── 08_cross_dataset_training.py # Huấn luyện ngưỡng trên PTDB-TUG & kiểm thử chéo (Cross-Dataset)
│   ├── 09_noise_robustness_ptdb.py  # Khảo sát kháng nhiễu dưới nhãn thanh quản EGG Ground Truth
│   ├── 11_benchmark_yin.py      # Benchmark thuật toán YIN Baseline vs. ACF vs. AMDF
│   └── 12_compare_three_champions.py # Đánh giá YIN Enhanced & Đối sánh Đỉnh cao 3 Quán quân
│
├── outputs/                     # Toàn bộ kết quả đầu ra
│   ├── figures/                 # Hình vẽ chất lượng cao (.png, 300 DPI)
│   └── reports/                 # Báo cáo chi tiết (.csv, .json)
│
├── README.md                    # Báo cáo kỹ thuật tổng quan và mục lục điều hướng
├── requirements.txt             # Danh sách thư viện Python
└── pyproject.toml               # Cấu hình dự án
```

---

## 3. Mục Lục Hệ Thống Tài Liệu Báo Cáo

Hệ thống tài liệu nghiên cứu chi tiết được cấu trúc thành 6 chuyên đề chuyên sâu trong thư mục `docs/`:

1. **[Chương I: Thuật Toán Hàm Tự Tương Quan (ACF)](docs/01_acf_theory_and_experiments.md)**
   * Cơ sở lý thuyết toán học của ACF ($L_2$ norm).
   * Minh họa trực quan khung Hữu thanh vs Vô thanh (`phone_F1.wav`).
   * Huấn luyện ngưỡng Gauss tối ưu Bayes ($T = 0.4408$ và $T_{\text{BP}} = 0.4892$).
   * Luận điểm khoa học chọn khung chuẩn $25\text{ ms}$ (so sánh 20ms vs 25ms vs 30ms).
   * Kết quả Baseline trên 4 file kiểm thử.
   * Chi tiết thiết kế toán học của 5 Plugins cải tiến.
   * Bảng xếp hạng và đồ thị phân tích toàn bộ 32 tổ hợp ACF.

2. **[Chương II: Thuật Toán Hàm Hiệu Độ Lớn Trung Bình (AMDF)](docs/02_amdf_theory_and_experiments.md)**
   * Cơ sở lý thuyết toán học của AMDF ($L_1$ norm) và chuẩn hóa biên độ.
   * Minh họa khung Voiced vs Unvoiced với đáy cực tiểu (Deep Dip).
   * Huấn luyện ngưỡng Gauss AMDF theo các điều kiện tiền xử lý ($T = 0.4380$).
   * Kết quả Baseline AMDF trên 4 file kiểm thử (sai số chỉ $1.60\text{ Hz}$).
   * Chi tiết thiết kế toán học của 5 Plugins thích ứng riêng cho cực tiểu AMDF.
   * Bảng xếp hạng toàn bộ 32 tổ hợp AMDF (cấu hình tối ưu đạt $1.24\text{ Hz}$).
   * Bảng đối sánh trực tiếp toàn diện: **ACF vs AMDF**.

3. **[Chương III: Phân Tích Hiện Tượng Trên Đồ Thị & Nguyên Nhân Sai Số](docs/03_error_analysis_and_phenomena.md)**
   * Sự suy giảm chất lượng giữa kênh thoại (Phone) và phòng thu (Studio).
   * Hiện tượng đứt gãy contour tại ranh giới âm tiết (Boundary Dropping) và giải pháp Hysteresis + Energy Extension.
   * Bản chất hiện tượng nhảy quãng tám (Octave Doubling / Halving) và cơ chế triệt tiêu của Center Clipping + Viterbi Tracking.
   * Đánh giá hiệu quả làm mịn của Lọc trung vị 3 khung (Median Filter).

4. **[Chương IV: Thuật Toán YIN & Đối Sánh Tam Giác Miền Thời Gian (ACF - AMDF - YIN)](docs/04_yin_theory_and_benchmarks.md)**
   * Nguồn gốc thuật toán YIN (triết lý Âm - Dương, dung hòa ưu điểm ACF và AMDF).
   * Cơ sở toán học 6 bước: Hiệu bình phương, Chuẩn hóa CMND, Ngưỡng tuyệt đối, Nội suy Parabol dưới mức mẫu.
   * Kết quả thực nghiệm đối sánh Tam giác Baseline trên 4 file kiểm thử.
   * Khảo sát các cấu hình YIN Enhanced: Lý do không dùng Center Clipping; Sức mạnh của Viterbi Tracking và Bandpass Filter.
   * Đối sánh trực diện đỉnh cao giữa **3 Quán quân**: ACF Champion ($1.86\text{ Hz}$) vs AMDF Champion ($1.50\text{ Hz}$) vs YIN Champion ($1.27\text{ Hz}$ / $1.08\text{ Hz}$).

5. **[Chương V: Khảo Sát Chuyên Sâu Độ Bền Vững Kháng Nhiễu (Noise Robustness)](docs/05_noise_robustness_study.md)**
   * Bản chất toán học về khả năng tự triệt tiêu nhiễu của chuẩn $L_2$ (ACF) vs hiện tượng sàn nhiễu dương của chuẩn $L_1$ (AMDF) vs cơ chế khử sàn nhiễu CMND của YIN.
   * Thiết lập thực nghiệm Stress Test với 8 mức AWGN từ $+25\text{ dB}$ đến $-5\text{ dB}$ trên cả 3 thuật toán.
   * Bảng kết quả đối kháng thực nghiệm đầy đủ qua các dải SNR (YIN dẫn đầu ở dải sạch đến trung bình, ACF Enhanced thống trị dải nhiễu cực đoan).
   * Xác định và phân tích vùng giao thoa (Cross-over Point tại $\text{SNR} \approx 7 - 10\text{ dB}$).
   * Phân tích minh họa cơ chế kháng nhiễu trên khung đơn lẻ tại $0\text{ dB}$.

6. **[Chương VI: Đánh Giá Chuẩn Quốc Tế Trên Cơ Sở Dữ Liệu PTDB-TUG (EGG Ground Truth)](docs/06_ptdb_tug_international_benchmark.md)**
   * Giới thiệu bộ dữ liệu quốc tế PTDB-TUG và giá trị "tiêu chuẩn vàng" của nhãn đo điện cực thanh quản (Laryngograph EGG).
   * Hệ thống công thức chuẩn quốc tế: **VDE, GPE, FFE, FPE**.
   * Bảng kết quả benchmark đối sánh Baseline vs Enhanced trên 3 thuật toán (ACF, AMDF, YIN).
   * Phân tích hiện tượng triệt tiêu nhảy quãng tám trên giọng nam (GPE giảm từ $6.09\%$ xuống $0.60\%$).
   * Thử nghiệm huấn luyện ngưỡng chéo (Cross-Dataset Training: VN vs PTDB-TUG).
   * Khảo sát độ bền vững kháng nhiễu dưới nhãn EGG Ground Truth hoàn toàn không bị ô nhiễm bởi âm học phòng.

---

## 4. Hướng Dẫn Cài Đặt Môi Trường & Thực Thi

### 4.1. Cài đặt môi trường bằng `uv` (Khuyên dùng)
Dự án được cấu hình chuẩn với `pyproject.toml`. Khuyến nghị sử dụng **`uv`** để tự động quản lý phiên bản Python 3.12 và dependencies:

```powershell
# 1. Khởi tạo môi trường ảo với Python 3.12
uv venv --python 3.12

# 2. Cài đặt các gói phụ thuộc ở chế độ editable
uv pip install -e .
```

*(Hoặc dùng `pip install -r requirements.txt` nếu dùng môi trường Python thông thường).*

---

### 4.2. Hướng dẫn chạy 9 kịch bản tự động hóa (Scripts 01 – 09)

Hệ thống được thiết kế hoàn toàn tự động hóa. Người dùng có thể chạy từng bước hoặc kiểm tra bất kỳ hợp phần nào thông qua các lệnh dưới đây:

#### Bước 1: Xem minh họa khung Voiced vs Unvoiced (Chuẩn 25 ms)
```powershell
uv run python scripts/01_demo_frames.py
```
* Xuất hình: `outputs/figures/01_demo_acf_frames.png`, `outputs/figures/01_demo_amdf_frames.png`

#### Bước 2: Huấn luyện tìm ngưỡng Gauss tối ưu
```powershell
# Huấn luyện cho ACF
uv run python scripts/02_train_threshold.py                     # Baseline thô (T = 0.4408)
uv run python scripts/02_train_threshold.py --plugins bandpass  # Có lọc tiền xử lý (T = 0.4892)

# Huấn luyện cho AMDF
uv run python scripts/02_train_threshold.py --method amdf       # Baseline thô (T = 0.4380)
```
* Xuất hình: `outputs/figures/02_threshold_distribution*.png`

#### Bước 3: Đối sánh ảnh hưởng chiều dài khung (20 ms vs 25 ms vs 30 ms)
```powershell
uv run python scripts/03_compare_params.py
```
* Xuất hình: `outputs/figures/03_compare_frame_length.png`

#### Bước 4: Chạy kiểm thử tự động 4 file kiểm thử tiếng Việt
```powershell
# Kiểm thử mô hình Baseline gốc
uv run python scripts/04_run_testing.py                         # ACF Baseline (T = 0.4408)
uv run python scripts/04_run_testing.py --method amdf          # AMDF Baseline (T = 0.4380)

# Kiểm thử mô hình Enhanced tích hợp trọn bộ Plugins
uv run python scripts/04_run_testing.py --plugins all
uv run python scripts/04_run_testing.py --method amdf --plugins all
```
* Xuất hình: `outputs/figures/04_test_*.png`

#### Bước 5: Đánh giá và xếp hạng toàn bộ 32 tổ hợp Plugins ($2^5$)
```powershell
uv run python scripts/05_compare_plugins.py                     # Khảo sát 32 tổ hợp cho ACF
uv run python scripts/05_compare_plugins.py --method amdf       # Khảo sát 32 tổ hợp cho AMDF
```
* Báo cáo: `outputs/reports/05_all_combinations_ranking*.csv`
* Xuất hình: `outputs/figures/05_all_combinations_ranking*.png`

#### Bước 6: Khảo sát độ bền vững kháng nhiễu trên tập tiếng Việt (Stress Test)
```powershell
uv run python scripts/06_noise_robustness.py
```
* Báo cáo: `outputs/reports/06_noise_robustness_summary.csv`
* Xuất hình: `outputs/figures/06_noise_robustness_curves.png`, `outputs/figures/06_noise_mechanism_frame_demo.png`

#### Bước 7: Đánh giá Benchmark chuẩn quốc tế trên PTDB-TUG (EGG Ground Truth)
```powershell
# 1. Tải 10 câu mẫu cho F01 và M01 (~25 MB)
uv run python scripts/download_ptdb_tug.py --speakers F01,M01 --num-utterances 10

# 2. Chạy Benchmark tính các chỉ số VDE, GPE, FFE, FPE
uv run python scripts/07_benchmark_ptdb.py
```
* Báo cáo: `outputs/reports/07_ptdb_benchmark.csv`, `outputs/reports/07_ptdb_benchmark.json`
* Xuất hình: `outputs/figures/07_ptdb_benchmark.png`

#### Bước 8: Huấn luyện ngưỡng chéo Cross-Dataset (VN vs. PTDB-TUG)
```powershell
uv run python scripts/08_cross_dataset_training.py
```
* Báo cáo: `outputs/reports/08_cross_dataset_training_report.json`
* Xuất hình: `outputs/figures/08_threshold_distribution_ptdb.png`

#### Bước 9: Khảo sát kháng nhiễu chuẩn hóa quốc tế dưới nhãn EGG
```powershell
uv run python scripts/09_noise_robustness_ptdb.py
```
* Báo cáo: `outputs/reports/09_ptdb_noise_robustness.csv`
* Xuất hình: `outputs/figures/09_ptdb_noise_robustness_curves.png`

#### Bước 10: Benchmark thuật toán YIN Baseline vs. ACF vs. AMDF
```powershell
uv run python scripts/11_benchmark_yin.py
```
* Báo cáo: `outputs/reports/compare_yin.csv`, `outputs/reports/compare_yin.json`
* Xuất hình: `outputs/figures/11_compare_acf_amdf_yin.png`

#### Bước 11: Đánh giá YIN Enhanced & Đối sánh Đỉnh cao 3 Quán quân
```powershell
uv run python scripts/12_compare_three_champions.py
```
* Báo cáo: `outputs/reports/compare_three_champions.csv`, `outputs/reports/compare_three_champions.json`
* Xuất hình: `outputs/figures/12_compare_three_champions.png`

---

## 5. Bảng Tổng Hợp Kết Quả Thực Nghiệm Cốt Lõi

| Tiêu chí / Kịch bản thực nghiệm | Thuật toán ACF | Thuật toán AMDF | Thuật toán YIN | Đánh giá & Kết luận khoa học |
| :--- | :---: | :---: | :---: | :--- |
| **Ngưỡng tối ưu ($T_{\text{opt}}$)** | $T = 0.4408$ | $T = 0.4380$ | $T = 0.2500$ | ACF-AMDF đối ngẫu quanh 0.44; YIN dùng ngưỡng CMND tuyệt đối |
| **Sai số $F_0$ Baseline (Sạch)** | $3.17\text{ Hz}$ ($2.19\%$) | $1.60\text{ Hz}$ ($1.09\%$) | **$1.40\text{ Hz}$ ($0.97\%$)** | YIN dẫn đầu nhờ hàm CMND và nội suy parabol dưới mức mẫu |
| **Sai số $F_0$ Tối ưu (Best Plugins)** | $1.86\text{ Hz}$ (Config 29) | $1.50\text{ Hz}$ (Config 13) | **$1.27\text{ Hz}$ (Viterbi) / $1.08\text{ Hz}$ (BP)** | YIN Champion đạt độ chính xác F0 cao nhất toàn bộ đồ án |
| **F1-Score phân loại V/UV tối ưu** | $91.39\%$ | **$93.83\%$** (Config 05) | **$92.22\%$** (Viterbi) | Ranh giới CMNDF 0.25 phân định dứt khoát tuần hoàn / vô thanh |
| **Hiệu năng tại SNR = 0 dB (Nhiễu lớn)** | **$2.93\text{ Hz}$** (Enhanced) | $10.23\text{ Hz}$ (Enhanced) | **$5.19\text{ Hz}$** (Enhanced) | ACF kháng nhiễu cực đoan mạnh nhất; YIN vượt trội AMDF |
| **Lỗi GPE giọng nam PTDB-TUG** | $6.09\% \rightarrow \mathbf{0.60\%}$ | $0.17\% \rightarrow \mathbf{0.58\%}$ | $5.82\% \rightarrow \mathbf{5.68\%}$ | Plugins triệt tiêu $86.2\%$ lỗi nhảy quãng tám trên ACF |
| **Độ lệch ngưỡng Cross-Dataset** | $\lvert \Delta T \rvert = \mathbf{0.0132}$ | $\lvert \Delta T \rvert = \mathbf{0.0073}$ | $\text{Chuẩn hóa CMND}$ | Ngưỡng hội tụ phổ quát độc lập ngôn ngữ và người nói |

