# FAC — Ước lượng F0 và phân loại hữu thanh/vô thanh

So sánh ACF, AMDF và Real Cepstrum. Phạm vi hiện tại gồm **baseline/Champion**, **Otsu, K-means 1D, K-means 2D, ngưỡng động** và **Pitch-Shift không model**.

## Chạy toàn bộ và tạo báo cáo

```powershell
uv sync --locked
uv run python scripts/32_run_all_benchmarks.py --output-dir outputs/current_results
```

Báo cáo chỉ gồm ba nhóm: baseline/Champion, không giám sát và Pitch-Shift. Thư mục kết quả chứa bảng train/test, CSV từng file, contour, ảnh dự đoán và log kiểm thử. Dùng thư mục mới cho mỗi run.

## Pipeline Champion

| Thuật toán | Plugin | Ngưỡng | Frame/hop |
|---|---|---:|---:|
| ACF | Bandpass + Energy extension + Viterbi | 0,45 | 30/10 ms |
| AMDF | Energy extension + Viterbi | 0,50 | 30/10 ms |
| Cepstrum | Bandpass + Center clipping + Energy extension + Viterbi | 0,10 | 30/10 ms |

Giữ nguyên mọi plugin Champion đang dùng. Baseline dùng 25/10 ms. Các nhánh dùng cùng vùng F0 70–400 Hz, bỏ DC và gate năng lượng.

## Không giám sát

```powershell
uv run python scripts/21_benchmark_champion_unsupervised.py train --output-dir outputs/unsupervised_new_run
uv run python scripts/21_benchmark_champion_unsupervised.py evaluate --output-dir outputs/unsupervised_new_run
uv run python scripts/21_benchmark_champion_unsupervised.py report --output-dir outputs/unsupervised_new_run
```

Fit ngưỡng/tâm cụm trên train rồi đóng băng trước test; K-means 2D thích nghi từng input được báo cáo riêng. Ngưỡng động dùng Otsu train làm nền, median cục bộ và trọng số STE; không nối qua im lặng.

## Pitch-Shift

```powershell
uv run python scripts/31_compare_native_pitch_shift.py --output-dir outputs/pitch_shift_new_run
```

Mỗi nhánh dùng bộ ước lượng riêng trên khung gốc và khung resample: **ACF–ACF, AMDF–AMDF, Cepstrum–Cepstrum**. Hệ số dịch 1,25; sigma 0,12; ngưỡng điểm 0,5. AMDF dùng độ mạnh 1-d_min; Cepstrum dùng c_positive/(c_positive+0,1), với 0,1 lấy từ preset đã chốt. F0 đầu ra và các plugin Champion giữ nguyên.

File `src/analysis/ssl_pitch_shift.py` giữ thuật toán ACF-only ban đầu làm đối chứng. Script 27 đánh giá bản ACF-only độc lập; script 28 dùng Pitch-Shift riêng theo thuật toán. Script 31 chỉ bật đối chứng lịch sử dùng chung ACF khi chỉ định rõ `--include-legacy-shared-acf`; bảng mặc định không dùng nhãn ACF chung. Pitch-Shift hiện là kiểm tra nhất quán dưới phép biến đổi, không có quá trình huấn luyện mạng.

## Dự đoán và kiểm thử

```powershell
uv run fac predict -i TinHieuKiemThu -m champions -o outputs/predictions
uv run python -B -m unittest discover -s tests
```

Train/test có bốn WAV mỗi tập. LAB chỉ chứa F0mean/F0std/F0num. Final Score là trung bình sai số phần trăm ba thống kê, thấp hơn tốt hơn; không phải Accuracy/F1 từng khung.

[Thuật toán](docs/METHODS.md) · [Hướng dẫn kiểm thử](docs/TESTING.md) · [Kết quả không giám sát](docs/CHAMPION_UNSUPERVISED_RESULTS.md).
