# Metric và đối sánh — refactor đợt 4

`src/metrics` là nơi sở hữu metric, tổng hợp và đối sánh. Các công thức từng file, alias, xử lý reference thiếu/zero và định dạng cũ được giữ nguyên; `analysis.evaluation` và `analysis.validation` còn re-export để tương thích. CLI và benchmark gọi trực tiếp module đúng trách nhiệm thay vì lặp phép tính trung bình.

## Hợp đồng legacy-v1

- Mỗi file tính sai số tuyệt đối và ba MAPE cho F0mean, F0std, F0num. **Final Score là trung bình ba MAPE chưa làm tròn, sau đó mới làm tròn hai chữ số**. Nó không nhất thiết bằng trung bình ba MAPE đã hiển thị.
- Reference bằng zero vẫn có sai số tuyệt đối, nhưng MAPE tương ứng là null. Reference thiếu hoặc không hữu hạn cũng thành null. Thiếu bất kỳ MAPE nào thì Final Score null; không thay thành zero hoặc trung bình hai thành phần.
- Mean/std và MAPE xuất theo quy tắc hai chữ số hiện hành. Sai số count giữ quy tắc cũ: làm tròn absolute error hai chữ số, rồi làm tròn số nguyên. Các alias như `composite_score`, `mape_mean`, `mape_f0_std` giữ nguyên.
- LAB không có nhãn thời gian thì Accuracy/F1 là null. Điểm summary không chứng minh đúng contour hoặc tỷ lệ V/UV bằng 50/50.
- Nội bộ/JSON dùng null cho N/A. Hiển thị dùng `N/A`, CSV dùng ô trống. Không biến chuỗi hiển thị trở lại thành số hoặc dùng format để tính thứ hạng.

Đơn vị: absolute mean/std là Hz, absolute count là frame; MAPE và Final Score là phần trăm. `metric_contract()` trả metadata của hợp đồng, gồm đơn vị từng trường.

## Profile tổng hợp giữ số liệu cũ

| Profile | Dùng ở đâu | Quy tắc |
|---|---|---|
| `file-score-macro-v1` | Runner 17/21 | Trung bình từng metric đã tính theo file, không làm tròn thêm; Final Score là macro của điểm từng file |
| `file-score-macro-rounded-2-v1` | CLI, benchmark 04/05/15 | Macro từng file rồi làm tròn hai chữ số |
| `file-score-macro-count-1-v1` | Benchmark 16 | Như profile rounded, riêng mean absolute count dùng một chữ số |
| `component-score-macro-v1` | Benchmark 14 | Trung bình MAPE theo file, rồi trung bình ba thành phần; adapter bảng làm tròn hai chữ số |

Các profile không được gộp ngầm. Ví dụ raw MAPE 0,0149% và 0,0049% cùng count MAPE 0% cho điểm từng file 0,01%; lấy trung bình ba MAPE đã hiển thị 0,01/0/0 rồi làm tròn lại sẽ thành 0,00%. Cả hai đường lịch sử được đặt tên và giữ nguyên, không cập nhật expected cho chúng giống nhau.

`aggregate_rows()` luôn dùng đủ population, kể cả row có metric null. Row thiếu metric làm aggregate trường đó null, không bị loại khỏi mẫu. `MetricSeries` nghiêm ngặt hơn: yêu cầu đủ trường, file ID không trùng và toàn bộ danh sách file đã khai báo trong context.

## Metadata và sidecar của run mới

Pha evaluate của runner tạo thêm `metrics.json`, được ghim trong manifest cùng SHA. File này chứa:

- Contract/version/profile và SHA của spec, frozen config/model, evaluation.
- Dataset identity, split và danh sách WAV đầy đủ từ dataset manifest.
- Feature contract thực tế: method, frame/hop/range/gate, raw/processed, preprocessing/ZCR; cùng cấu hình smoothing/post-processing.
- Fit mode/population/training identity và strategy/model identity.
- Các row metric canonical và aggregate đã kiểm tra khớp evaluation cũ.

Sidecar là phần bổ sung; không đổi shape hoặc điểm trong `evaluation.json`. Verify kiểm tra lại sidecar từ các artifact đã ghim, không chạy detector hoặc fit model. Run native đợt 3 thiếu sidecar vẫn đọc/xác minh được, nhưng không tự đoán metadata để đối sánh. Run lịch sử không bị bổ sung file tại chỗ.

Champion input-adaptive có mapping SHA tới từng file model đã lưu; context pin digest của tập đó. Giao thức raw Stage 4 trước đây không lưu model từng input: adapter ghi `model_sha256: null` và `input_models_recorded: false`, không bịa digest hoặc refit khi đọc metric. Nguồn/dữ liệu/config của run vẫn được ghim. Control là cấu hình đối sánh cố định; tên mode không phủ nhận việc Champion kế thừa calibration từ train ở các đợt trước.

## Đối sánh có kiểm tra

```python
from src.metrics.comparison import read_series, compare_series

gmm = read_series('outputs/runs/champion-recheck', 'test/acf/gmm')
hmm = read_series('outputs/runs/champion-recheck', 'test/acf/hmm')
comparison = compare_series(gmm, hmm)
```

CLI chỉ đọc và in JSON:

```powershell
uv run python -B -m src.metrics outputs/runs/champion-recheck test/acf/gmm outputs/runs/champion-recheck test/acf/hmm
```

Mặc định phải trùng dataset, split, metric version, aggregation, feature contract, smoothing, post-processing và fit protocol. Không cho trộn frame 25/30 ms, train_all/train_active, frozen/input-adaptive hoặc các bộ train khác mà không khai báo. Strategy/model SHA có thể khác vì đó là biến đang được đối sánh; cả hai context được giữ trong kết quả.

Thí nghiệm ablation có thể khai báo chính xác khác biệt ở `feature_contract`, `post_processors`, `smoothing`, `fit` qua `allow_differences` hoặc `--allow`. Kết quả ghi lại cả giá trị trái/phải của từng khác biệt, không coi chúng tương đương. Không được miễn kiểm tra dataset/split/version/profile hoặc bỏ file bằng inner join.

Delta là **right − left**; sai số giảm tương ứng delta âm. Nếu một bên thiếu số liệu, delta null kèm trạng thái `missing_left`/`missing_right`/`missing_both`, không tạo mức cải thiện giả. Aggregate giữ đúng thứ tự file và số học của profile cũ; không tự thay bằng phép mean khác để bỏ sai khác float.

## Source và kiểm chứng

Numeric/evaluation không import tầng định dạng hoặc experiment runner. `formatting.py` thuộc scope report; sửa riêng nó không vô hiệu train/predict. Shim cũ lazy-load phần format. Runner Stage 4 vẫn cần metric để chọn theo train, nên các dependency số học/evaluation còn được giữ bảo thủ trong source scope; tách evaluation revision đầy đủ sẽ tiếp tục cùng reporting.

189 test đạt, gồm 19 test mới cho rounding/profile, N/A, population và metadata. Kiểm chứng riêng so 89 vector trước refactor, 72 nhóm aggregate, 48 baseline/Champion, 288 bộ metric từ 93.423 dòng contour và đối sánh typed series. AST các hàm metric/format di chuyển khớp snapshot đợt 3.

```powershell
uv run python -B -m unittest discover -s tests
uv run python -B scripts/24_verify_metric_refactor.py verify
```

[Bằng chứng đợt 4](../outputs/refactor_phase4/README.md) nằm riêng; model/preset/dataset và các run cũ giữ nguyên. Đây là refactor hợp đồng v1, không thêm metric raw/v2, không chọn lại mô hình theo test. Đợt 5 sẽ tách report view model/bundle/publisher và cho renderer dùng metric đã xác minh.
