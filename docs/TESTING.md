# Hướng dẫn kiểm thử

Chạy lệnh từ thư mục gốc FAC. Phiên bản hiện tại hỗ trợ ACF, AMDF, Cepstrum. Dùng thư mục đầu ra riêng cho mỗi tập hoặc lần chạy để tránh trộn kết quả.

## Môi trường

```powershell
uv sync
```

Có thể thay `uv run python` bằng `.\.venv\Scripts\python.exe` nếu môi trường đã được cài đặt. Không xóa `.venv` hoặc metadata cài đặt khi dọn cache.

## Test hồi quy pipeline

```powershell
uv run python -B -m unittest discover -s tests -v
```

Có 90 test: 27 test [đợt 1](STAGE1_FIXES.md), 26 test [đợt 2](STAGE2_FIXES.md), 26 test [đợt 3](STAGE3_FIXES.md) và 11 test [đợt 4](STAGE4_RESULTS.md) về chọn trên train, freeze, cache, giao thức và timeline riêng. Detector yêu cầu audio dài ít nhất một khung; CLI thoát mã 2 với thông báo rõ nếu input không đủ dài.

## Dự đoán file hoặc thư mục

```powershell
# Ba cấu hình Champion trên toàn bộ test
uv run python predict.py -i TinHieuKiemThu -m champions -o outputs/predictions/test

# Train có summary riêng
uv run python predict.py -i TinHieuHuanLuyen -m champions -o outputs/predictions/train

# Một WAV mới, đầu ra tách riêng
uv run python predict.py -i "D:\Data\audio.wav" -m champions -o outputs/predictions/custom

# So sánh ACF Champion và ACF Baseline
uv run python predict.py -i TinHieuKiemThu -m acf -o outputs/predictions/acf
```

| Mode `-m` | Hệ thống được chạy |
|---|---|
| `champions` | ACF, AMDF, Cepstrum Champion; mặc định |
| `baseline` | Ba baseline |
| `all` | Ba baseline và ba Champion |
| `acf` | ACF baseline và Champion |
| `amdf` | AMDF baseline và Champion |
| `cepstrum` | Cepstrum baseline và Champion |

Các Champion đã chọn trên train dùng frame/hop 30/10 ms: ACF = BP + Energy + Viterbi, T=0,45; AMDF = Energy + Viterbi, T=0,50; Cepstrum = BP + Clip + Energy + Viterbi, T=0,10. Baseline vẫn dùng frame/hop 25/10 ms và ngưỡng lịch sử. Tên Champion chỉ preset chọn trên train; Cepstrum tăng sai số test sau hiệu chỉnh. Xem [đợt 4](STAGE4_RESULTS.md).

`outputs/predictions/train/` và `test/` đang giữ lần chạy trước đợt 1. Đổi `-o` thành thư mục mới (ví dụ `outputs/predictions/stage1_test`) để giữ các bản đối chiếu này.

Nếu có LAB cùng tên, CLI tính sai lệch mean/std/count. Nếu không có LAB, CLI vẫn dự đoán và xuất thống kê. LAB hiện hành không có segments nên Accuracy/F1 để trống.

## Đọc kết quả

Trong thư mục `-o`:

- `<tên>_pitch_prediction.csv`: dạng dài `Method,Frame_Index,Time_Sec,Label,F0_Hz`, time ở đơn vị giây, giữ đúng tâm/số khung cho từng phương pháp. F0 bằng 0 biểu thị không có pitch được xuất; không nội suy ghép timeline của khung 25 và 30 ms. Artifact cũ giữ schema của phiên bản tạo nó.
- `<tên>_pitch_contour.png`: waveform và contour của các hệ thống được chọn.
- `evaluation_summary_metrics.csv`: summary của các file trong **lần chạy vừa thực hiện**; chạy lại cùng thư mục sẽ ghi đè file này.

Final Score thấp hơn là tốt hơn; không chuyển `100 - Final Score` thành Accuracy. Final Score yêu cầu đủ ba MAPE mean/std/count. MAPE không xác định khi tham chiếu bằng 0 hoặc thiếu; JSON giữ null, CSV để trống, màn hình hiện N/A. Trung bình toàn tập không bỏ qua file thiếu thành phần. `F0num` bằng tham chiếu không chứng minh mọi vị trí voiced đều đúng.

## Script thực nghiệm

```powershell
# Demo hai khung: thay acf bằng amdf hoặc cepstrum
uv run python scripts/01_demo_frames.py --method acf

# Khảo sát 16 tổ hợp trên train: thay method khi cần
uv run python scripts/05_compare_plugins.py --method acf

# Sáu cấu hình trên cả train và test
uv run python scripts/14_compare_acf_amdf_cepstrum.py --dataset both

# Các phương pháp không giám sát
uv run python scripts/15_evaluate_unsupervised_thresholds.py --output-dir outputs/stage3_evaluation/recheck15
uv run python scripts/16_evaluate_advanced_unsupervised.py --output-dir outputs/stage2_thresholds/recheck
```

Các script benchmark ghi vào tên báo cáo/hình cố định, có thể thay thế artifact đã lưu. Tham khảo [danh mục kết quả](../outputs/README.md) trước khi chạy lại.

Script 15/16 hỗ trợ `--output-dir` để xuất `figures/` và `reports/` dưới thư mục riêng. Bỏ tùy chọn này vẫn ghi vào `outputs/` như trước. Đợt 2 giữ bản đối chiếu ở `outputs/stage2_thresholds/before/` và `after/`; đợt 3 lưu ở `outputs/stage3_evaluation/`. Benchmark 15 mới dùng các ID median/otsu/kmeans/valley/supervised trong JSON và CSV, với tên hiển thị riêng.

`04_run_testing.py` vẫn hữu ích để chạy ACF/AMDF với các plugin BP/Hysteresis/Energy. `--plugins all` ở script này không tương đương Champion có Viterbi.

`02_train_threshold.py` cần tối thiểu hai khung có nhãn cho mỗi lớp V/UV và std dương. LAB hiện hành thiếu segments nên chương trình báo thiếu mẫu và thoát mã 2 trước khi xuất artifact. `03_compare_params.py` đã chạy được với LAB hiện hành, hiển thị Accuracy/F1 là N/A. Để giữ đầu ra cũ, truyền `--output_fig` và `--output_json` sang thư mục mới; xem [đợt 3](STAGE3_FIXES.md).

API, CLI và benchmark dùng `src/config.py` cùng factory trong `src/pipelines.py`. Baseline giữ ngưỡng lịch sử; Champion dùng preset đã chọn trên train ở đợt 4. Nếu cần ngưỡng khác hãy truyền `threshold` tường minh. Đối chứng script 04 có thể đọc file JSON ngưỡng riêng và ghi nguồn trong báo cáo.

## Hiệu chỉnh và đóng băng trước test

```powershell
uv run python scripts/17_validate_and_calibrate.py train --output-dir outputs/stage4_recheck
uv run python scripts/17_validate_and_calibrate.py evaluate --output-dir outputs/stage4_recheck
```

Train chỉ đọc `--train-dir`, ghi hai freeze cùng SHA-256 trước khi evaluate. Evaluate kiểm tra dữ liệu train/checksum và dùng đúng cấu hình đã chọn. Dùng thư mục mới khi chạy lại; không ghi đè freeze/evaluation cũ. `src/calibrated_presets.py` chứa preset hiện hành và checksum lựa chọn, không cần đọc artifact ngoài package khi chạy CLI.

Script 05 vẫn xuất các nhóm top như trước, đồng thời thêm `*_raw.json` đủ 16 cấu hình và `*_per_file.csv`; mỗi raw result có base commit, branch và checksum source. [Kết quả đợt 4](../outputs/stage4_validation/README.md) lưu 888 lượt hiệu chỉnh, 72 dòng pipeline và 408 dòng giao thức theo file.

## Nội dung lịch sử

Các hướng dẫn có YIN và các hình cấu hình cũ nằm trong [archive](../archive/2026-10-28-pre-cleanup/README.md). Không dùng các lệnh đó cho checkout hiện tại.
