# Đợt 4 — 30/10/2026

> Bộ freeze/evaluation đã được tạo lại ngày 31/10 ở commit `43b1de0`. Ngày 01/11, snapshot được phục dựng để khớp đúng provenance này, checksum preset được đồng bộ và `validation.json` được sinh lại bằng kiểm tra thực tế. Bản snapshot/runtime/validation cũ nằm trong `historical_source_snapshot/`. Các điểm Stage 4 vẫn giữ Otsu trước sửa; xem [kết quả sau sửa](../main_review_2026-11-01/README.md).

Nguồn hiện hành: [docs/STAGE4_RESULTS.md](../../docs/STAGE4_RESULTS.md). Điểm là sai số thống kê toàn file, thấp hơn tốt hơn; không phải Accuracy/F1 từng khung.

## Chọn cấu hình trước test

- `frozen_config.json` + `.sha256`: 222 cấu hình khảo sát trên train, 888 cặp cấu hình/file, quy tắc lựa chọn và ba preset đã chốt.
- `frozen_models.json` + `.sha256`: ngưỡng/GMM, K-Means 2D và emission HMM fit trên train; ghi rõ toàn bộ khung hay chỉ active.
- `train/*_trials.json`: mọi ứng viên được khảo sát cho từng thuật toán, với cấu hình và kết quả từng file.
- `train/plugin_sweep.json`: đủ 16 tổ hợp cho mỗi thuật toán tại tham số lịch sử.
- `train/all_trials_per_file.csv`: 888 dòng kết quả từng file, không chỉ các nhóm top có lặp cấu hình.
- `training_source_snapshot.tar.gz`: 46 file nguồn khớp từng SHA trong provenance của freeze ngày 31/10, được phục dựng từ byte Git có đối chiếu LF/CRLF ngày 01/11. `runtime.json` giữ phiên bản thư viện đã ghi và bổ sung nguồn phục dựng/checksum snapshot. Không diễn giải việc sửa bằng chứng này là một lần train mới.

## Đánh giá sau khi chốt

- `evaluation.json`: train/test cho baseline, Champion trước hiệu chỉnh và preset đã chốt, cùng 17 giao thức không giám sát cho mỗi thuật toán.
- `evaluation_pipelines_per_file.csv`: 72 dòng (2 tập × 3 thuật toán × 3 cấu hình × 4 file).
- `evaluation_protocols_per_file.csv`: 408 dòng (2 tập × 3 thuật toán × 17 giao thức × 4 file).
- `contours/`: contour preset đã chốt, tách tập/thuật toán/file.
- `predictions/train/`, `predictions/test/`: mỗi tập có bốn hình, bốn CSV từng khung và summary 24 dòng cho sáu cấu hình API/CLI.
- `figures/`: so sánh pipeline train/test, khảo sát ngưỡng trên train và bảng màu các giao thức test.
- `summary.json`, `validation.json`: tóm tắt và kiểm tra tính nhất quán.

`validation.json` hiện chỉ khẳng định những kiểm tra đã thực hiện: liên kết evaluation–freeze, snapshot/runtime, preset xuất ra và test nếu chạy `--run-tests`. Các cờ của lần kiểm chứng cũ không được sao chép thành kết luận cho lần tạo lại.

CSV dự đoán CLI hiện dùng dạng dài `Method, Frame_Index, Time_Sec, Label, F0_Hz`. Mỗi thuật toán giữ đúng các tâm khung của mình; không ghép giả timeline 25 ms và 30 ms. Artifact các đợt trước giữ nguyên schema lúc tạo.

## Chẩn đoán kết quả lịch sử

`cepstrum_trace.json` replay đúng commit `caaf751`, tái lập 4,43%/7,62% nhưng bộ đếm cho thấy detector gọi ACF, không gọi Cepstrum. `legacy_with_cepstrum_branch/cepstrum_trace.json` chỉ thay `plugin_detector.py` bằng bản có nhánh Cepstrum: điểm thành 6,84%/13,24%. Đây là kiểm chứng nhân quả của báo cáo pipeline lai trước đây.

## Tái lập

```powershell
.\.venv\Scripts\python.exe -B scripts/17_validate_and_calibrate.py train --output-dir outputs/stage4_recheck
.\.venv\Scripts\python.exe -B scripts/17_validate_and_calibrate.py evaluate --output-dir outputs/stage4_recheck
```

Không ghi đè các freeze/evaluation đã có. Bốn file train nhỏ chưa đảm bảo tổng quát: ACF/AMDF giảm lỗi test, Cepstrum tăng lỗi test dù giảm lỗi train. Preset được giữ nguyên sau khi xem test. Test đã từng được xem trong lịch sử phát triển dự án; không xem đây là test hoàn toàn mới.
