# Tích hợp tám phương pháp không giám sát với Champion — 01/11/2026

Nhánh `codex/fix-main-review`. Nguồn đầy đủ và báo cáo sinh tự động: [outputs/champion_unsupervised_2026-11-01](../outputs/champion_unsupervised_2026-11-01/README.md). Không thay preset Champion hoặc chọn lại phương pháp theo test.

> Sau khi sửa ba lỗi biên, đã đối chiếu lại cả 288 trường hợp bằng code mới với mô hình cũ: label/time/F0 và metric đều khớp chính xác, 120 test đạt. Freeze/snapshot dưới đây vẫn giữ mã lúc benchmark; bằng chứng cập nhật nằm riêng tại [final_review_2026-11-01](../outputs/final_review_2026-11-01/README.md).

## Giao thức và phạm vi không giám sát

Phần quyết định mới fit từ 659 khung active trên bốn file train cho mỗi thuật toán. Dùng đúng frame/hop 30/10 ms, BP/Clipping của Champion cho cả train và inference. ZCR tính trên frame pitch sau tiền xử lý và bỏ DC. Không fit từ LAB; LAB chỉ được hash cho provenance trong pha train, và được đọc để chấm điểm ở pha evaluate.

Pipeline vẫn kế thừa Champion đã hiệu chỉnh bằng thống kê LAB train ở Stage 4. Đặc biệt, Energy dùng ngưỡng periodicity cố định 0,45 / 0,50 / 0,10 cho ACF/AMDF/Cepstrum. Vì vậy, chỉ gọi các mô hình quyết định mới là không giám sát; không gọi toàn bộ hệ thống là học hoàn toàn không dùng tham chiếu.

Tám phương pháp chính: Median, Otsu, Valley, K-Means 1D, GMM 1D, K-Means 2D, HMM và ngưỡng động. Tất cả dùng mô hình train đã đóng băng; ngưỡng động thêm ngữ cảnh median/STE trong từng đoạn active, cửa sổ 50 khung, alpha 0,30. Cửa sổ có nhìn tương lai nên đây là xử lý offline.

HMM giữ A/pi cố định, chỉ fit Gaussian emission; không nối trạng thái xuyên qua silence hoặc giữa hai file. Hai đối chứng 2D/HMM fit trên từng input được báo cáo riêng và lưu mô hình từng file, không trộn với mô hình train.

Giữ cùng silence gate, ứng viên pitch, median theo đoạn, Energy và Viterbi. Ngưỡng quyết định scalar/vector không làm đổi ngưỡng guard của Energy. Baseline đối chứng không plugin dùng 25 ms; các phương pháp còn lại dùng 30 ms.

## Kết quả test

| Nhóm | Phương pháp | ACF test (%) | AMDF test (%) | Cepstrum test (%) |
|---|---|---:|---:|---:|
| control | Baseline không plugin | 5.822 | 6.357 | 48.185 |
| control | Champion đã chốt | 2.385 | 2.755 | 22.942 |
| train_fitted | Median | 6.673 | 7.787 | 18.017 |
| train_fitted | Otsu | 3.380 | 2.875 | 27.837 |
| train_fitted | Valley | 28.652 | 11.740 | 34.718 |
| train_fitted | K-Means 1D | 3.617 | 2.875 | 27.797 |
| train_fitted | GMM 1D | 5.003 | 4.928 | 18.797 |
| train_fitted | K-Means 2D | 14.920 | 2.735 | 28.047 |
| train_fitted | HMM | 5.535 | 4.647 | 20.717 |
| train_fitted | Ngưỡng động | 4.367 | 3.535 | 16.480 |
| input_adaptive | 2D fit từng input | 8.852 | 3.170 | 10.375 |
| input_adaptive | HMM fit từng input | 8.303 | 7.045 | 14.482 |

- ACF: Champion 2,385% vẫn thấp hơn cả tám mô hình fit trên train.
- AMDF: 2D fit trên train đạt 2,735%, gần ngang Champion 2,755%; bốn file chưa đủ để kết luận mức chênh 0,020 điểm phần trăm có ý nghĩa tổng quát.
- Cepstrum: ngưỡng động đạt 16,480% trên test, so với Champion 22,942%. Tuy nhiên trên train, ngưỡng động 18,828% cao hơn Champion 11,390%, nên không tự thay Champion theo thứ hạng test.
- Cepstrum 2D fit từng input đạt 10,375%; đây là thích nghi trên chính input, không phải kết quả từ centroid cố định đã fit trên train.
- Các kết quả mới không được dùng để khẳng định đã tái lập giao thức chưa lưu của các số 14,31% / 14,40% / 2,17% trước đây. Quần thể fit và quy tắc Energy đã được định nghĩa rõ trong lần mới.

## Bảng tám cột theo thuật toán

Mỗi ô là trung bình theo bốn file test. Final Score là trung bình ba MAPE mean/std/count ở từng file rồi lấy trung bình tập; không phải Accuracy/F1. Các dòng input_active là đối chứng thích nghi riêng.

### ACF

| Phương pháp | ΔF0 (Hz) | MAPE F0 (%) | ΔStd (Hz) | MAPE Std (%) | ΔNum (khung) | MAPE Num (%) | Final Score (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
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

### AMDF

| Phương pháp | ΔF0 (Hz) | MAPE F0 (%) | ΔStd (Hz) | MAPE Std (%) | ΔNum (khung) | MAPE Num (%) | Final Score (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
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

### CEPSTRUM

| Phương pháp | ΔF0 (Hz) | MAPE F0 (%) | ΔStd (Hz) | MAPE Std (%) | ΔNum (khung) | MAPE Num (%) | Final Score (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
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

## Bằng chứng và kiểm thử

- 288 kết quả/file: 192 cho tám phương pháp fit trên train, 48 đối chứng thích nghi từng input, 48 baseline/Champion.
- 24 đối sánh Champion khớp từng label, tâm khung và F0 với đường native; 48 bản ghi baseline/Champion khớp metric đã lưu ở Stage 4.
- 93.423 dòng contour đã được kiểm tra timeline theo sample rate, label/silence, F0 hữu hạn trong 70–400 Hz, số khung voiced và tính lại mean/std/Final Score.
- Lưu 12 CSV đặc trưng train, 24 JSON mô hình thích nghi, 288 CSV contour và bảng per_file.csv. Các quyết định trước/sau hậu xử lý cùng số khung thêm/bớt đều có trong artifact.
- Snapshot 49 file nguồn khớp freeze; dữ liệu train/test và checksum report khớp. Kiểm tra cú pháp Python 3.9 trên 55 file đạt.
- 108/108 test hồi quy đạt, gồm tám test mới cho feature space, hồi quy native, không refit khi inference, ranh giới HMM/dynamic, gate/range và luồng train–freeze–evaluate–report.
- Chi tiết máy đọc: [validation.json](../outputs/champion_unsupervised_2026-11-01/validation.json).

## Tái lập

```powershell
uv run python -B scripts/21_benchmark_champion_unsupervised.py train --output-dir outputs/champion_unsupervised_recheck
uv run python -B scripts/21_benchmark_champion_unsupervised.py evaluate --output-dir outputs/champion_unsupervised_recheck
uv run python -B scripts/21_benchmark_champion_unsupervised.py report --output-dir outputs/champion_unsupervised_recheck
```

Train từ chối ghi đè run cũ. Evaluate kiểm tra checksum, snapshot, source và cấu hình native; không cho dùng mô hình với mã hoặc dữ liệu train đã đổi. Report chỉ đọc kết quả đã lưu và sinh bảng/hình trong thư mục run.

Giới hạn còn lại: test đã được xem trong lịch sử; chỉ bốn train/bốn test; LAB thiếu nhãn thời gian và contour tham chiếu. Điểm thấp không chứng minh đúng contour, hết lỗi octave hoặc xác nhận giả thiết V/UV 50/50. Chưa thêm tùy chọn chọn các phương pháp mới trong CLI predict.py.
