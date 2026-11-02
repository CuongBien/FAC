# Toàn bộ kết quả FAC sau khi merge main — 02/11/2026

Main đã nhận toàn bộ chuỗi refactor tại `12ec529` từ `f059592` (bảy commit: một kế hoạch và sáu đợt). Báo cáo này được lập trên nhánh riêng rồi gộp vào main; không push remote. Lỗi unlink khi checkout trên Windows được xử lý bằng đối chiếu và khôi phục đúng byte từ Git/snapshot trước khi fast-forward; không giải quyết xung đột bằng cách bỏ code.

Đây là tổng hợp từ artifact đã có, không phải một lần fit hoặc lựa chọn lại theo test. Mọi bảng ghi rõ giao thức và nguồn. Bảng hiển thị ba chữ số; CSV giữ độ chính xác số đã lưu. `N/A` không được biến thành zero.

## Phạm vi dữ liệu và cách đọc điểm

Bốn WAV train (F1/M1) và bốn WAV test (F2/M2), phone 16 kHz, studio 44,1 kHz. Hop 10 ms, dải F0 70–400 Hz; baseline 25 ms và Champion 30 ms. LAB chỉ có thống kê toàn file, không có contour F0 chuẩn hoặc nhãn V/UV theo thời gian.

`Final Score` là trung bình ba MAPE (mean F0, std F0, số frame voiced), chấm từng file theo hợp đồng legacy-v1 rồi tổng hợp đúng profile. **Thấp hơn tốt hơn; không phải Accuracy/F1 và không được diễn giải 100 − Score là accuracy.** Tham chiếu zero/missing làm thành phần MAPE và Final Score không xác định. Benchmark 14 còn dùng component-macro riêng nên có thể lệch 0,01 do thứ tự làm tròn.

| File | Tập | Hz | Giây | F0mean chuẩn (Hz) | F0std chuẩn (Hz) | F0num chuẩn |
|---|---|---|---|---|---|---|
| phone_F1.wav | train | 16000 | 3.240 | 215.6 | 20.6 | 148 |
| phone_M1.wav | train | 16000 | 4.160 | 123.7 | 16.8 | 232 |
| studio_F1.wav | train | 44100 | 2.863 | 229.6 | 36.8 | 127 |
| studio_M1.wav | train | 44100 | 2.730 | 116.9 | 26.4 | 82 |
| phone_F2.wav | test | 16000 | 4.800 | 150.2 | 30.7 | 219 |
| phone_M2.wav | test | 16000 | 2.800 | 130.2 | 15.4 | 123 |
| studio_F2.wav | test | 44100 | 3.149 | 198.6 | 47.8 | 139 |
| studio_M2.wav | test | 44100 | 2.382 | 154.7 | 30.3 | 116 |

## Kết quả dọn dẹp

- Di chuyển 50 file cũ: ZIP, hai hướng dẫn, 17 file dự đoán, 25 hình và năm báo cáo; sao lưu thêm README và năm chương. Tổng 56 file, 44,174,788 byte (~42.13 MiB) được lưu, không phải dung lượng đã xóa.
- Xác minh lại 56/56 SHA; hai file văn bản cần phục hồi EOL trong bộ nhớ để khớp SHA gốc. ZIP được giữ cục bộ và bị Git ignore; khi clone mới phải sao lưu/chuyển ZIP riêng.
- Bỏ năm wrapper plugin không còn dùng, năm helper và một alias; dọn 29 nhóm import, tám thư mục cache và hai `.gitkeep` thừa. Code cũ còn trong lịch sử Git.
- Hợp nhất hướng dẫn, sửa link ảnh/LaTeX/sample rate, bỏ bảng trùng; tách dự đoán train/test. Giữ WAV/LAB, lock, Gaussian và các bảng lịch sử để truy vết.
- Dọn dẹp giữ nguyên thân hàm/class còn lại theo AST; đây là cải thiện tổ chức, không phải tuyên bố tăng chất lượng F0 hoặc giảm dung lượng toàn repo.

Nguồn: [docs/CLEANUP.md](../docs/CLEANUP.md); [archive/2026-10-28-pre-cleanup/manifest.json](../archive/2026-10-28-pre-cleanup/manifest.json).

## Baseline, Champion và hiệu chỉnh tham số

Stage 4 khảo sát 222 cấu hình / 888 cặp cấu hình-file chỉ trên train: đủ 16 tổ hợp BP/Clip/Energy/Viterbi tại 25 ms, sau đó mở lưới khung 20/25/30 ms và ngưỡng cho các tổ hợp được chọn. Mỗi thuật toán có 74 trial. Không chứng minh tối ưu toàn bộ không gian tham số.

| Thuật toán | Plugin Champion | Frame/hop ms | T baseline → Champion | Baseline train | Champion train | Baseline test | Champion test | Giảm Score test tương đối |
|---|---|---|---|---|---|---|---|---|
| ACF | BP + Energy + Viterbi | 30/10 | 0.4408 → 0.45 | 13.887 | 2.215 | 5.822 | 2.385 | 59.04% |
| AMDF | Energy + Viterbi | 30/10 | 0.438 → 0.5 | 6.760 | 2.078 | 6.357 | 2.755 | 56.67% |
| CEPSTRUM | BP + Clip + Energy + Viterbi | 30/10 | 0.06 → 0.1 | 44.540 | 11.390 | 48.185 | 22.942 | 52.39% |

Mức giảm trên là của **toàn bộ pipeline**, gồm cả đổi frame/ngưỡng; không gán riêng cho BP, Energy hoặc Viterbi. Baseline/Champion khác frame nên đây là so sánh hệ thống đã định nghĩa, không phải ablation chỉ một biến.

So với Champion trước hiệu chỉnh (đã sửa pipeline, dùng cấu hình lịch sử):

| Thuật toán | Train trước → sau | Test trước → sau | Test sau − trước (điểm %) |
|---|---|---|---|
| ACF | 3.880 → 2.215 | 4.223 → 2.385 | -1.837 |
| AMDF | 3.757 → 2.078 | 4.125 → 2.755 | -1.370 |
| CEPSTRUM | 15.160 → 11.390 | 21.203 → 22.942 | 1.740 |

ACF và AMDF cải thiện; Cepstrum giảm train nhưng **tăng** test. Giữ cấu hình đã chọn trên train, không đảo lựa chọn sau khi xem test.

Các giải pháp đang triển khai: BP 70–900 Hz; center clipping 40%; silence gate dựa STE sau trừ DC; median riêng từng đoạn voiced; Energy extension có điều kiện năng lượng/periodicity/liên tục pitch; Viterbi trên đúng frame đã tiền xử lý (w_freq=3, w_octave=4, tối đa năm ứng viên). Hysteresis có API riêng nhưng không thuộc ba preset hiện hành và chưa có bảng đo đóng góp riêng trong benchmark này.

Nguồn: [outputs/stage4_validation/evaluation.json](../outputs/stage4_validation/evaluation.json); [src/calibrated_presets.py](../src/calibrated_presets.py).

## Tám phương pháp trong đúng pipeline Champion

Phần quyết định mới fit trên 659 frame active của bốn file train cho mỗi thuật toán, đúng BP/Clip và 30/10 ms. Median, Otsu, Valley, K-Means 1D, GMM 1D, K-Means 2D và emission HMM được chốt trước test; ngưỡng động dùng thêm median/STE theo ngữ cảnh đầu vào. HMM giữ A/pi cố định, reset tại silence và file boundary. Cửa sổ ngưỡng động 50 frame, alpha=0,30, có nhìn tương lai nên là offline.

Pipeline kế thừa Champion đã hiệu chỉnh bằng LAB train; Energy guard cố định 0,45 / 0,50 / 0,10. Chỉ phần quyết định mới được fit không giám sát, không gọi toàn hệ thống là không dùng tham chiếu. Hai dòng cuối fit riêng trên mỗi input và thuộc giao thức thích nghi khác.

| Phương pháp | ACF train | ACF test | AMDF train | AMDF test | Cepstrum train | Cepstrum test |
|---|---|---|---|---|---|---|
| Baseline không plugin | 13.887 | 5.822 | 6.760 | 6.357 | 44.540 | 48.185 |
| Champion đã chốt | 2.215 | 2.385 | 2.078 | 2.755 | 11.390 | 22.942 |
| Median | 14.145 | 6.673 | 5.820 | 7.787 | 14.140 | 18.017 |
| Otsu | 3.033 | 3.380 | 3.495 | 2.875 | 45.157 | 27.837 |
| Valley | 40.335 | 28.652 | 11.043 | 11.740 | 49.788 | 34.718 |
| K-Means 1D | 3.033 | 3.617 | 3.495 | 2.875 | 45.157 | 27.797 |
| GMM 1D | 6.935 | 5.003 | 4.465 | 4.928 | 19.668 | 18.797 |
| K-Means 2D | 9.640 | 14.920 | 2.585 | 2.735 | 18.960 | 28.047 |
| HMM | 13.005 | 5.535 | 3.663 | 4.647 | 23.178 | 20.717 |
| Ngưỡng động | 4.487 | 4.367 | 4.238 | 3.535 | 18.828 | 16.480 |
| 2D fit từng input | 7.585 | 8.852 | 2.738 | 3.170 | 22.612 | 10.375 |
| HMM fit từng input | 3.888 | 8.303 | 4.445 | 7.045 | 12.445 | 14.482 |

- ACF: Champion 2,385% thấp hơn cả tám phương pháp train-fitted trong lần đo này.
- AMDF: K-Means 2D train-fitted đạt 2,735% so Champion 2,755%, chỉ hơn 0,020 điểm; chưa đủ bằng chứng về ý nghĩa thống kê/tổng quát.
- Cepstrum: ngưỡng động 16,480% test so Champion 22,942%, nhưng train lại 18,828% so 11,390%; không thay preset theo test.
- Cepstrum 2D fit mỗi input đạt 10,375%, HMM mỗi input 14,482%; không xem là kết quả suy luận từ mô hình train cố định.
- Những số 14,31% / 14,40% / 2,17% từng xuất hiện trước đây chưa có đủ giao thức/mô hình/file tương ứng; không dùng như thành tích được xác nhận.

Nguồn: [outputs/champion_unsupervised_2026-11-01/evaluation.json](../outputs/champion_unsupervised_2026-11-01/evaluation.json); [docs/CHAMPION_UNSUPERVISED_RESULTS.md](../docs/CHAMPION_UNSUPERVISED_RESULTS.md).

### Ngưỡng scalar đã fit trong không gian Champion

| Thuật toán | Median | Otsu | Valley | K-Means 1D | GMM |
|---|---|---|---|---|---|
| ACF | 0.692230 | 0.572564 | 0.780229 | 0.574101 | 0.648525 |
| AMDF | 0.228717 | 0.359532 | 0.191470 | 0.361251 | 0.274803 |
| CEPSTRUM | 0.145473 | 0.254098 | 0.312026 | 0.251127 | 0.179100 |

K-Means 2D/HMM có mô hình riêng trong frozen_models; ngưỡng động thay đổi theo frame. Không lấy các ngưỡng trên áp vào raw 25 ms. Cấu hình prior 0,5 không chứng minh tỷ lệ V/UV thật bằng 50/50.

### Phân rã sai số test của toàn bộ 12 hệ thống

#### ACF

| Phương pháp | Sai số mean (Hz) | MAPE mean % | Sai số std (Hz) | MAPE std % | Sai số count (frame) | MAPE num % | Final Score % |
|---|---|---|---|---|---|---|---|
| Baseline không plugin | 1.788 | 1.210 | 2.105 | 5.787 | 17.250 | 10.473 | 5.822 |
| Champion đã chốt | 1.803 | 1.120 | 0.902 | 2.600 | 6.000 | 3.433 | 2.385 |
| Median | 2.823 | 1.860 | 1.377 | 5.545 | 20.750 | 12.615 | 6.673 |
| Otsu | 1.953 | 1.278 | 0.993 | 3.650 | 8.750 | 5.210 | 3.380 |
| Valley | 15.902 | 10.905 | 3.595 | 19.715 | 84.500 | 55.343 | 28.652 |
| K-Means 1D | 2.025 | 1.333 | 1.062 | 4.110 | 9.000 | 5.415 | 3.617 |
| GMM 1D | 2.605 | 1.718 | 1.412 | 5.130 | 13.500 | 8.160 | 5.003 |
| K-Means 2D | 5.992 | 3.962 | 4.342 | 14.988 | 47.000 | 25.817 | 14.920 |
| HMM | 2.385 | 1.565 | 1.423 | 5.420 | 15.500 | 9.618 | 5.535 |
| Ngưỡng động | 2.312 | 1.518 | 1.403 | 5.018 | 11.000 | 6.570 | 4.367 |
| 2D fit từng input | 3.422 | 2.147 | 5.138 | 12.785 | 18.750 | 11.633 | 8.852 |
| HMM fit từng input | 2.973 | 1.860 | 5.163 | 12.855 | 16.000 | 10.198 | 8.303 |

#### AMDF

| Phương pháp | Sai số mean (Hz) | MAPE mean % | Sai số std (Hz) | MAPE std % | Sai số count (frame) | MAPE num % | Final Score % |
|---|---|---|---|---|---|---|---|
| Baseline không plugin | 0.885 | 0.540 | 3.110 | 10.390 | 14.250 | 8.148 | 6.357 |
| Champion đã chốt | 0.557 | 0.330 | 1.298 | 3.763 | 7.250 | 4.175 | 2.755 |
| Median | 1.550 | 1.010 | 1.750 | 6.532 | 24.750 | 15.820 | 7.787 |
| Otsu | 0.820 | 0.532 | 1.130 | 2.715 | 9.750 | 5.377 | 2.875 |
| Valley | 3.140 | 2.143 | 1.720 | 8.207 | 38.250 | 24.875 | 11.740 |
| K-Means 1D | 0.820 | 0.532 | 1.130 | 2.715 | 9.750 | 5.377 | 2.875 |
| GMM 1D | 1.552 | 1.032 | 1.673 | 5.745 | 14.000 | 8.008 | 4.928 |
| K-Means 2D | 0.650 | 0.400 | 1.290 | 3.445 | 8.250 | 4.365 | 2.735 |
| HMM | 1.280 | 0.888 | 1.662 | 5.860 | 12.750 | 7.192 | 4.647 |
| Ngưỡng động | 0.935 | 0.597 | 1.205 | 2.772 | 12.750 | 7.238 | 3.535 |
| 2D fit từng input | 0.488 | 0.287 | 1.368 | 4.075 | 9.250 | 5.155 | 3.170 |
| HMM fit từng input | 2.667 | 1.730 | 1.405 | 5.412 | 21.500 | 13.997 | 7.045 |

#### CEPSTRUM

| Phương pháp | Sai số mean (Hz) | MAPE mean % | Sai số std (Hz) | MAPE std % | Sai số count (frame) | MAPE num % | Final Score % |
|---|---|---|---|---|---|---|---|
| Baseline không plugin | 18.615 | 12.955 | 22.695 | 113.567 | 23.000 | 18.038 | 48.185 |
| Champion đã chốt | 6.120 | 4.127 | 9.615 | 51.617 | 18.750 | 13.085 | 22.942 |
| Median | 6.777 | 4.252 | 3.138 | 17.683 | 44.500 | 32.112 | 18.017 |
| Otsu | 12.633 | 7.735 | 6.570 | 18.860 | 80.250 | 56.920 | 27.837 |
| Valley | 14.558 | 9.125 | 7.005 | 20.623 | 108.750 | 74.407 | 34.718 |
| K-Means 1D | 12.703 | 7.787 | 6.635 | 19.297 | 79.500 | 56.310 | 27.797 |
| GMM 1D | 9.170 | 5.848 | 2.452 | 8.998 | 57.000 | 41.537 | 18.797 |
| K-Means 2D | 8.630 | 6.255 | 11.115 | 63.680 | 19.250 | 14.215 | 28.047 |
| HMM | 9.372 | 5.963 | 5.022 | 15.938 | 54.500 | 40.250 | 20.717 |
| Ngưỡng động | 6.978 | 4.412 | 1.138 | 5.595 | 55.250 | 39.435 | 16.480 |
| 2D fit từng input | 3.598 | 2.188 | 2.263 | 8.188 | 31.750 | 20.758 | 10.375 |
| HMM fit từng input | 5.607 | 3.538 | 1.363 | 3.860 | 51.750 | 36.055 | 14.482 |

MAPE std lớn là nguồn lỗi đáng kể của Cepstrum; điểm mean thấp không tự chứng minh contour đúng. Toàn bộ metric train và từng file nằm trong CSV kèm nguồn, không bị lược bỏ.

### Baseline và Champion theo từng file test

#### ACF

| File | Baseline Score % | Champion mean Hz | Champion std Hz | Champion voiced frame | Champion Score % |
|---|---|---|---|---|---|
| phone_F2.wav | 7.170 | 150.98 | 31.38 | 203 | 3.350 |
| phone_M2.wav | 4.960 | 132.19 | 15.83 | 124 | 1.710 |
| studio_F2.wav | 6.220 | 201.64 | 45.45 | 136 | 2.870 |
| studio_M2.wav | 4.940 | 156.10 | 30.15 | 120 | 1.610 |

#### AMDF

| File | Baseline Score % | Champion mean Hz | Champion std Hz | Champion voiced frame | Champion Score % |
|---|---|---|---|---|---|
| phone_F2.wav | 8.110 | 150.07 | 30.86 | 201 | 2.940 |
| phone_M2.wav | 4.110 | 130.07 | 16.23 | 129 | 3.450 |
| studio_F2.wav | 5.020 | 199.65 | 43.92 | 134 | 4.080 |
| studio_M2.wav | 8.190 | 155.62 | 30.62 | 116 | 0.550 |

#### CEPSTRUM

| File | Baseline Score % | Champion mean Hz | Champion std Hz | Champion voiced frame | Champion Score % |
|---|---|---|---|---|---|
| phone_F2.wav | 28.870 | 156.19 | 35.54 | 198 | 9.780 |
| phone_M2.wav | 122.290 | 139.64 | 41.04 | 125 | 58.460 |
| studio_F2.wav | 9.010 | 202.62 | 49.58 | 113 | 8.150 |
| studio_M2.wav | 32.570 | 159.73 | 36.50 | 90 | 15.380 |

## Đối chứng raw 25 ms sau sửa Otsu — benchmark 15/16

Bốn train / bốn test; 1.291 frame train gồm silence; không tích hợp plugin Champion. Median/Otsu/Valley/K-Means/GMM fit train_all; 2D/HMM benchmark 16 fit active từng file test. Ngưỡng động dùng GMM train và ngữ cảnh input. Dòng `supervised` trong JSON chỉ ngưỡng tĩnh lịch sử, không phải fit có giám sát lại từ LAB hiện hành.

| Chiến lược raw 25 ms | ACF test % | AMDF test % | Cepstrum test % |
|---|---|---|---|
| Ngưỡng tĩnh lịch sử | 5.82 | 6.36 | 48.18 |
| Median | 5.17 | 7.43 | 40.37 |
| Otsu | 6.45 | 7.15 | 53.68 |
| K-Means 1D | 6.45 | 7.15 | 57.00 |
| Valley | 18.09 | 6.79 | 71.16 |
| GMM 1D | 5.17 | 6.67 | 33.11 |
| 2D fit từng input | 7.63 | 8.16 | 31.63 |
| HMM fit từng input | 16.24 | 17.82 | 30.84 |
| Ngưỡng động | 5.93 | 7.63 | 33.77 |

Nguồn mới nhất sau sửa Otsu: [outputs/main_review_2026-11-01/README.md](../outputs/main_review_2026-11-01/README.md). Không trộn bảng này với Champion 30 ms hoặc Stage 4 snapshot bên dưới.

### Ảnh hưởng riêng của sửa Otsu

| Thuật toán | T cũ → đúng | Score test cũ → đúng |
|---|---|---|
| ACF | 0.455915 → 0.460244 | 6.35 → 6.45 |
| AMDF | 0.394971 → 0.399144 | 7.25 → 7.15 |
| CEPSTRUM | 0.190325 → 0.193046 | 53.02 → 53.68 |

Ngưỡng đổi để đúng ranh giới histogram; Score không giảm đồng loạt. Các model Stage 4 đã đóng băng không tự được fit lại sau sửa này.

## Đủ 17 giao thức raw Stage 4 — snapshot lịch sử

Các bảng này giữ thuật toán và model lúc Stage 4; Otsu còn phiên bản trước sửa cạnh bin. `train_all` có silence, `train_active` chỉ active train, `input_active` fit từng đầu vào. Cùng raw 25/10 ms, không dùng preprocessing/postprocessing Champion. Không gọi chúng là kết quả chạy lại bằng HEAD hiện tại.

### TRAIN

| Giao thức | ACF % | AMDF % | Cepstrum % |
|---|---|---|---|
| historical_static | 13.887 | 6.760 | 44.540 |
| median_train_all | 13.115 | 6.235 | 40.945 |
| otsu_train_all | 14.540 | 8.473 | 50.822 |
| valley_train_all | 13.583 | 8.162 | 60.038 |
| kmeans_train_all | 14.652 | 8.197 | 50.953 |
| gmm_train_all | 13.188 | 7.388 | 35.360 |
| dynamic_train_all | 14.065 | 8.232 | 33.442 |
| median_train_active | 23.777 | 17.168 | 35.345 |
| otsu_train_active | 15.072 | 8.523 | 58.028 |
| valley_train_active | 51.812 | 19.535 | 60.038 |
| kmeans_train_active | 16.130 | 9.473 | 58.053 |
| gmm_train_active | 18.933 | 11.613 | 34.307 |
| dynamic_train_active | 18.195 | 11.672 | 33.773 |
| kmeans2d_train_active | 11.053 | 6.395 | 57.808 |
| hmm_train_active | 20.377 | 9.312 | 43.250 |
| kmeans2d_input_active | 11.920 | 9.585 | 25.950 |
| hmm_input_active | 15.207 | 9.960 | 34.438 |

### TEST

| Giao thức | ACF % | AMDF % | Cepstrum % |
|---|---|---|---|
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

Nguồn: [outputs/stage4_validation/evaluation.json](../outputs/stage4_validation/evaluation.json). Bản mới sau sửa Otsu cho train_all nằm ở mục benchmark 15/16, không sửa đè snapshot này.

## Đủ 16 tổ hợp plugin trên train và tìm kiếm 222 cấu hình

Bảng sweep pha 1 dùng 25/10 ms và ngưỡng lịch sử theo tổ hợp BP/Clip. Đây là điểm train để chọn nhánh tìm kiếm, không phải 16 ablation test với một threshold chung.

| Plugin | ACF train % | AMDF train % | Cepstrum train % |
|---|---|---|---|
| Không plugin | 13.887 | 6.760 | 44.540 |
| BP | 13.160 | 8.340 | 47.473 |
| Clip | 13.350 | 27.730 | 32.312 |
| Energy | 11.835 | 6.220 | 39.265 |
| Viterbi | 8.938 | 4.605 | 24.475 |
| BP + Clip | 8.800 | 16.335 | 38.495 |
| BP + Energy | 11.255 | 6.580 | 45.303 |
| BP + Viterbi | 9.440 | 6.030 | 40.157 |
| Clip + Energy | 11.425 | 24.920 | 30.268 |
| Clip + Viterbi | 13.348 | 15.085 | 20.303 |
| Energy + Viterbi | 6.465 | 3.757 | 19.905 |
| BP + Clip + Energy | 7.192 | 14.157 | 36.120 |
| BP + Clip + Viterbi | 8.808 | 10.758 | 15.763 |
| BP + Energy + Viterbi | 3.880 | 4.303 | 38.495 |
| Clip + Energy + Viterbi | 11.528 | 10.718 | 17.760 |
| BP + Clip + Energy + Viterbi | 7.182 | 8.745 | 15.160 |

Toàn bộ 222 trial (gồm lựa chọn frame/ngưỡng) có trong `aggregate_metrics.csv`; 888 kết quả theo file trong `per_file_metrics.csv`. Tiêu chí chọn: mean Score train; hòa thì xét file tệ nhất, số plugin, khoảng cách tới 25 ms, tên plugin và ngưỡng. Không chọn trên test.

Cache Stage 4 ghi nhận 213364 hit / 28945 miss; đây là thống kê của lần chạy đó, không phải phép đo tăng tốc trước/sau refactor.

## Các đợt sửa logic và ảnh hưởng đã đo

### Đợt 1 — sửa tính đúng pipeline

Đã sửa PCM unsigned/stereo, DC và silence gate, file ngắn, biên lag/nội suy F0, Viterbi trên đúng pitch_frames, median theo đoạn và lấp UV có điều kiện. Không đổi ngưỡng để cố giữ điểm cũ.

| Hệ thống | Train trước → sau % | Test trước → sau % |
|---|---|---|
| 1. ACF Baseline | 13.23 → 13.89 | 5.55 → 5.82 |
| 2. ACF Champion (BP+Energy+Viterbi) | 2.95 → 3.88 | 4.22 → 4.22 |
| 3. AMDF Baseline | 7.01 → 6.76 | 4.02 → 6.36 |
| 4. AMDF Champion (Energy+Viterbi) | 3.29 → 3.76 | 3.78 → 4.12 |
| 5. Cepstrum Baseline | 39.81 → 44.54 | 42.95 → 48.19 |
| 6. Cepstrum Champion (BP+Clip+Energy+Viterbi) | 6.84 → 15.16 | 13.24 → 21.20 |

Điểm tăng ở nhiều hệ thống: sửa đúng đầu vào/hậu xử lý không đảm bảo giảm sai số thống kê. Không che các lần tăng này.

### Đợt 2 — GMM, 2D, HMM, cửa sổ động

Sửa dấu/prior của giao điểm Gaussian, EM ổn định số, gán cluster 2D trong không gian chuẩn hóa, reset HMM qua silence và đúng 50 frame cho cửa sổ chẵn. Trọng số/transition được ghi rõ; không coi dữ liệu suy biến là hai lớp hợp lệ.

| Chiến lược | ACF trước → sau % | AMDF trước → sau % | Cepstrum trước → sau % |
|---|---|---|---|
| Supervised (Bayes Lab) | 5.82 → 5.82 | 6.36 → 6.36 | 48.18 → 48.18 |
| Median 50% (Histogram) | 5.17 → 5.17 | 7.43 → 7.43 | 40.37 → 40.37 |
| Otsu (Histogram) | 6.35 → 6.35 | 7.25 → 7.25 | 53.02 → 53.02 |
| 1. GMM-EM 1D (50/50 Liên tục) | 5.09 → 5.17 | 6.30 → 6.67 | 40.55 → 33.11 |
| 2. Đa đặc trưng 2D (Periodicity+ZCR) | 7.63 → 7.63 | 8.16 → 8.16 | 31.63 → 31.63 |
| 3. 2-State HMM (Viterbi Decoding) | 16.55 → 16.24 | 20.53 → 17.82 | 39.14 → 30.84 |
| 4. Ngưỡng Động Thích Nghi Ngữ Cảnh | 5.94 → 5.93 | 7.32 → 7.63 | 41.45 → 33.77 |

Tên “50/50”/“Supervised” trong bảng trên là tên lưu ở benchmark lịch sử; không được dùng để suy ra tỷ lệ lớp thật hoặc tuyên bố đã train lại Gaussian có nhãn.

### Đợt 3 và rà soát bổ sung

- MAPE reference zero/missing giữ null; Score cần đủ ba thành phần; aggregate không bỏ file khó. Gaussian có giám sát từ LAB summary-only bị chặn thay vì tạo mô hình giả.
- Script 03 xử lý N/A; CSV benchmark 15 lấy đúng threshold bằng strategy ID; API/CLI/benchmark thống nhất factory và ngưỡng baseline.
- Otsu trả cạnh bin, dữ liệu hằng trả giá trị quan sát; CLI có mean nhưng thiếu std vẫn xuất hình/CSV.
- Parser LAB đọc trọn ký pháp khoa học và BOM; số sai/nonfinite/âm và frame count không nguyên bị từ chối. Cấu hình F0 sai bị phát hiện trước nhánh im lặng ở cả ba thuật toán.
- Preset checksum, runtime và snapshot Stage 4 được đồng bộ đúng freeze; snapshot cũ lưu riêng. Script 19 không còn ghi đè vùng thủ công, và sau refactor chỉ render gói báo cáo.

### Truy vết Cepstrum và mốc không được so sánh nhầm

| Mốc | Train % | Test % | Diễn giải |
|---|---:|---:|---|
| Artifact thiếu nhánh Cepstrum | 4.43 | 7.62 | Quyết định rơi vào ACF, hậu xử lý dùng Cepstrum: pipeline lai |
| Cùng replay, chỉ sửa nhánh | 6.84 | 13.24 | Đối chứng nguyên nhân, chưa có các sửa pipeline đợt 1 |
| Sau sửa pipeline, ngưỡng lịch sử | 15.160 | 21.203 | Mốc hợp lệ trước hiệu chỉnh train |
| Preset chọn trên train | 11.390 | 22.942 | Hiện hành; giảm train nhưng tăng test |

Nguồn: [docs/STAGE1_FIXES.md](../docs/STAGE1_FIXES.md), [docs/STAGE2_FIXES.md](../docs/STAGE2_FIXES.md), [docs/STAGE3_FIXES.md](../docs/STAGE3_FIXES.md), [docs/MAIN_REVIEW_FIXES.md](../docs/MAIN_REVIEW_FIXES.md), [docs/FINAL_EDGE_CASE_FIXES.md](../docs/FINAL_EDGE_CASE_FIXES.md).

## Các khảo sát bổ sung và ngưỡng lịch sử

Khảo sát frame đơn lẻ ACF trên phone_M1.wav, T=0.4408 (không phải kết luận chung cho mọi thuật toán):

| Frame ms | Tổng frame | Voiced frame | Mean F0 Hz | Std F0 Hz | Sai số mean (Hz) | MAPE mean % |
|---|---|---|---|---|---|---|
| 20 | 415 | 179 | 136.04 | 38.05 | 12.34 | 9.98 |
| 25 | 414 | 193 | 131.20 | 34.10 | 7.50 | 6.06 |
| 30 | 414 | 203 | 130.71 | 32.36 | 7.01 | 5.67 |

Nguồn: [outputs/stage3_evaluation/frame_comparison.json](../outputs/stage3_evaluation/frame_comparison.json). Accuracy/F1 ở cả ba cấu hình đều N/A.

Các ngưỡng Gaussian lưu từ nhãn lịch sử (614 voiced / 180 UV), không fit lại từ LAB hiện hành:

| Thuật toán | Raw | BP | Clip | BP + Clip |
|---|---|---|---|---|
| ACF | 0.440835 | 0.489180 | 0.335238 | 0.395257 |
| AMDF | 0.438040 | 0.373372 | 0.648843 | 0.562814 |

Cepstrum baseline dùng T=0,06 theo cấu hình lịch sử; không có bộ Gaussian Cepstrum tương đương trong JSON này. YIN, AWGN, PTDB-TUG ở ZIP lịch sử, không phải tính năng hay kết quả đã chạy lại bằng CLI hiện tại.

## Kết quả sáu đợt refactor

| Đợt | Commit | Kết quả | Test tại mốc |
|---|---|---|---|
| 1 | 14d58fb | Spec, fingerprint, snapshot, run store/registry, legacy reader | 137 |
| 2 | fe3f855 | Feature contract/engine/cache chung; chặn dùng nhầm feature space | 152 |
| 3 | c5324e2 | Runner train/evaluate, OS lock, journal, phục hồi công bố sau crash | 170 |
| 4 | 7d62cda | Metric legacy-v1, profile aggregate, N/A và đối sánh có metadata | 189 |
| 5 | 28be815 | View/report bundle, dựng lại ngoài Git, publisher marker, danh mục | 206 |
| 6 | 12ec529 | fac CLI, wheel/sdist, tài nguyên có SHA, môi trường sạch, CI | 217 |

Đây là cải thiện tính tái lập và quản lý; **không tuyên bố tăng độ chính xác thuật toán nhờ refactor**.

- 217/217 unittest đạt tại mã `12ec529`; sau merge đã đối chiếu lại source/test SHA, dữ liệu và hai run chính. Bản tổng hợp chỉ thay tài liệu và bổ sung bảng dẫn xuất, không thay mã đã kiểm thử.
- Đối chiếu 288 trường hợp Champion, 93,423 dòng contour, 3,873 dòng đặc trưng và ba model; thêm 72 pipeline + 408 protocol cases Stage 4. Không quét lại 222 cấu hình chỉ vì sửa renderer.
- Metric: 89 vector, 72 nhóm tổng hợp, 48 baseline/Champion và 288 bộ metric từ contour khớp trước refactor. Bốn profile làm tròn/tổng hợp được đặt tên, không ép chúng thành cùng kết quả.
- Reporting: 105 dòng bảng và summary Stage 4 khớp; sáu hàm vẽ giữ AST; gói báo cáo dựng lại bằng chính source archive ngoài Git; lỗi marker không ghi dở và giữ byte vùng thủ công.
- Packaging: wheel non-editable chạy ngoài checkout, không có Git/PYTHONPATH; Python 3.12.14 so đúng 12 dự đoán/contour trước refactor; Python 3.11.16 kiểm tra cùng hành vi. Mỗi runtime chạy thêm native 72 series, train không đọc test, resume không ghi lại, report/rebuild và từ chối đối sánh sai giao thức. `uv sync --locked` cũng được kiểm tra trong venv tạm.
- CI Windows/Linux × Python 3.9/3.10/3.11/3.12/3.13 đã cấu hình, **chưa push/chạy remote**; không coi matrix chưa chạy là bằng chứng đã đạt.

Nguồn: [outputs/refactor_phase3/validation.json](../outputs/refactor_phase3/validation.json), [outputs/refactor_phase4/validation.json](../outputs/refactor_phase4/validation.json), [outputs/refactor_phase5/validation.json](../outputs/refactor_phase5/validation.json), [outputs/refactor_phase6/README.md](../outputs/refactor_phase6/README.md).

## Số liệu máy đọc và kiểm chứng của báo cáo này

- [462 dòng metric tổng hợp](../outputs/project_summary_2026-11-02/aggregate_metrics.csv): 72 Champion + 120 Stage 4 + 222 trial train + 15 benchmark 15 + 21 benchmark 16 + 12 benchmark 14.
- [1.704 kết quả từng file](../outputs/project_summary_2026-11-02/per_file_metrics.csv): 288 Champion + 480 Stage 4 + 888 trial train + 48 benchmark 14. Các dòng ở nhiều family có thể là cùng cấu hình/file được lưu ở hai giao thức; không cộng chúng thành số mẫu độc lập.
- Mỗi row có source path/SHA, split, method, strategy, group, aggregation profile và configuration. Không gộp raw 25 ms với Champion 30 ms hoặc frozen với input-adaptive. Cột thiếu là ô trống. Benchmark 15/16 mới chỉ lưu aggregate trong các JSON này; không bịa kết quả từng file khi nguồn không lưu.
- [Kiểm chứng merge, archive và nguồn bảng](../outputs/project_summary_2026-11-02/verification.json); [Dữ liệu tổng hợp đủ độ chính xác](../outputs/project_summary_2026-11-02/results.json). Không sửa run gốc hoặc expected value.

## Những gì vẫn còn thiếu

- Chỉ bốn train/bốn test, test đã được quan sát trong lịch sử; cần tập độc lập mới trước khi kết luận tổng quát hoặc đổi preset.
- Không có F0 chuẩn/nhãn thời gian, nên chưa thể kết luận Accuracy/F1, tỷ lệ octave error, “khử sạch lỗi” hoặc tỷ lệ V/UV luôn 50/50. Non-voiced còn gồm silence.
- Cepstrum còn yếu về khả năng tổng quát. Ngưỡng động/2D thích nghi có điểm thấp hơn ở giao thức riêng, chưa phải lý do thay preset mặc định.
- CLI predict hiện có baseline/Champion; tám phương pháp là API/benchmark với model/spec, chưa có lựa chọn predict tổng quát cho từng phương pháp.
- Word, slide, đo T0 thủ công và benchmark YIN/AWGN/PTDB-TUG hiện hành chưa được tạo/xác minh trong checkout.
- Publisher thay từng file atomically và rollback lỗi I/O; process crash giữa nhiều tài liệu có thể cần preview/apply lại. Gói báo cáo bất biến vẫn là nguồn chuẩn.

Refactor sáu đợt và merge mã đã hoàn tất. Các mục trên là công việc dữ liệu/nghiên cứu/đầu ra môn học còn lại, không phải đợt refactor thứ bảy.
