# Toàn bộ kết quả hiện hành — chạy lại train/test

Thời gian bắt đầu (UTC+7): 2026-11-02T14:30:47.973324+07:00
Nhánh: `codex/retain-selected-unsupervised`. Mỗi tập có bốn WAV. Tất cả các bảng chính dùng metric legacy-v1: Final Score (%) càng thấp càng tốt.

## Phạm vi và giao thức

- Giữ nguyên các preset/plugin Champion. Baseline 25/10 ms; Champion và các biến thể 30/10 ms, F0 70–400 Hz.
- Otsu, K-means 1D/2D và dynamic fit từ train rồi đóng băng; dynamic lấy Otsu train làm nền. K-means 2D thích nghi input được tách riêng.
- Mọi nhánh chính dùng ACF–ACF, AMDF–AMDF hoặc Cepstrum–Cepstrum theo thuật toán. ACF-only cũ chỉ giữ riêng làm lịch sử/standalone ACF. Không chia sẻ nhãn ACF sang nhánh khác.
- LAB chỉ có F0mean/F0std/F0num; chưa có Accuracy/F1 hoặc sai số contour từng khung. Test đã được xem trong các lần trước; đây không phải test độc lập mới.

## 1. Baseline và Champion

| Phương pháp | ACF train → test | AMDF train → test | CEPSTRUM train → test |
|---|---:|---:|---:|
| Baseline | 13.887 → 5.822 | 6.760 → 6.357 | 44.540 → 48.185 |
| Champion | 2.215 → 2.385 | 2.078 → 2.755 | 11.390 → 22.942 |

## 2. Không giám sát trên Champion

| Phương pháp | ACF train → test | AMDF train → test | CEPSTRUM train → test |
|---|---:|---:|---:|
| Otsu | 3.033 → 3.380 | 3.495 → 2.875 | 45.157 → 27.837 |
| K-means 1D | 3.033 → 3.617 | 3.495 → 2.875 | 45.157 → 27.797 |
| K-means 2D (train-fitted) | 9.640 → 14.920 | 2.585 → 2.735 | 18.960 → 28.047 |
| Ngưỡng động (nền Otsu) | 3.198 → 2.860 | 3.393 → 2.862 | 35.140 → 26.697 |
| K-means 2D (fit từng input) | 7.585 → 8.852 | 2.738 → 3.170 | 22.612 → 10.375 |

## 3. Pitch-Shift không model

| Phương pháp | ACF train → test | AMDF train → test | CEPSTRUM train → test |
|---|---:|---:|---:|
| Pitch-Shift theo từng thuật toán | 2.960 → 2.745 | 2.227 → 3.028 | 16.815 → 18.730 |

[Tổng hợp CSV](summary.csv) · [Từng file](per_file.csv).
