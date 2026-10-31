# Sửa đánh giá, cấu hình và script đợt 3 — 30/10/2026

Thực hiện trên nhánh `codex/fix-stage1-pipeline`, chưa merge vào `main`. Đợt này hoàn tất các mục 12–16 trong kế hoạch, giữ dữ liệu và kết quả lịch sử để đối chiếu. Không hiệu chỉnh ngưỡng theo điểm test.

## 1. MAPE và Final Score khi thiếu tham chiếu hợp lệ

- Tham chiếu bằng 0: sai số tuyệt đối vẫn có giá trị; MAPE tương ứng là `None`, kể cả khi dự đoán cũng bằng 0. Không còn báo MAPE bằng 0 giả tạo.
- Thiếu thống kê tham chiếu hoặc giá trị không hữu hạn: thành phần không xác định giữ `None`.
- Final Score luôn là trung bình **đủ ba MAPE** mean/std/count. Nếu thiếu một thành phần thì Final Score là `None`; không chuyển sang trung bình hai thành phần.
- Trung bình toàn tập không tự loại các file có chỉ số thiếu. Nếu một file không có thành phần hợp lệ, trung bình thành phần đó và Final Score toàn tập cũng không xác định.
- JSON dùng `null`, CSV để trống, bảng/biểu đồ hiện `N/A`. Cột không có giá trị không được vẽ thành thanh 0 hoặc chọn làm cấu hình tốt nhất. Accuracy/F1 vẫn chỉ được tính khi có nhãn thời gian.

Các quy tắc làm tròn, tính trung bình và hiển thị được dùng chung trong `src/analysis/evaluation.py`, áp dụng cho CLI và các script 03–05, 14–16. Đã bỏ cách gọi `100 − Final Score` là độ chính xác tương đối.

## 2. Chặn fit và vẽ Gaussian thiếu dữ liệu

`extract_training_distributions()` yêu cầu ít nhất hai khung được gán nhãn cho **mỗi lớp V và UV**. LAB chỉ có F0mean/std/num không thể thay cho segments theo thời gian; hàm báo lỗi kèm số mẫu thay vì trả một bộ mean/std toàn 0.

`find_gaussian_threshold()` kiểm tra mean hữu hạn, std dương và khả năng phân tách; dùng chung bộ giải giao điểm đã kiểm chứng ở đợt 2. Hàm vẽ Gaussian cũng kiểm tra mẫu và tham số trước khi tạo hình. Script 02 thoát mã 2 với thông báo rõ khi thiếu dữ liệu, không tạo báo cáo/ảnh huấn luyện giả.

Tiền xử lý khi trích mẫu có nhãn đã đồng bộ quy tắc trừ DC trước frame plugin, và reset plugin giữa các file. Ngưỡng đã lưu trong lịch sử vẫn được giữ nguyên; cấu hình ghi rõ nguồn đó.

## 3. Script 03 hỗ trợ Accuracy/F1 không có

Chạy được với LAB hiện hành ở các khung 20, 25 và 30 ms. Bảng hiện `N/A`, JSON giữ `null`, biểu đồ ghi `N/A` tại cột không đo được. Các sai số thống kê mean/std vẫn được báo riêng.

Kết quả chạy thật: [frame_comparison.json](../outputs/stage3_evaluation/frame_comparison.json), [biểu đồ](../outputs/stage3_evaluation/frame_comparison.png). Không diễn giải số khung voiced khác nhau giữa các frame length thành Accuracy/F1.

## 4. CSV benchmark 15 dùng đúng ngưỡng

JSON, CSV, đánh giá và biểu đồ cùng dùng các ID `median`, `otsu`, `kmeans`, `valley`, `supervised`. Tên hiển thị nằm riêng trong `strategy_labels` và cột CSV `Strategy_Label`.

`Threshold_T` lấy trực tiếp từ dictionary ngưỡng dùng để chạy detector; không còn lookup bằng display name rồi mặc định về 0. JSON/CSV giữ đầy đủ độ chính xác số của ngưỡng; làm tròn chỉ phục vụ hiển thị.

**Thay đổi schema:** người đọc kết quả benchmark 15 mới cần dùng các ID trên, thay vì các khóa tên dài cũ. Các báo cáo lịch sử vẫn giữ schema cũ. Script 15 có thêm `--output-dir`, tương tự script 16, để xuất sang thư mục riêng.

## 5. Một nguồn cấu hình cho API, CLI và benchmark

`src/config.py` giữ các ngưỡng và hệ số lịch sử. `src/pipelines.py` tạo mới baseline/Champion; CLI và benchmark 14 gọi cùng factory. Sweep plugin dùng cùng bảng ngưỡng theo BP/Clip. Demo AMDF/Cepstrum cũng lấy ngưỡng đối chứng từ cấu hình chung.

| Baseline | Ngưỡng mặc định thống nhất |
|---|---:|
| ACF | 0,4408 |
| AMDF | 0,4380 |
| Cepstrum | 0,0600 |

Đây là các giá trị đã được dùng trong benchmark, không phải kết quả huấn luyện mới. **API mặc định thay đổi:** AMDF trước đây là 0,35, Cepstrum là 0,18; nếu cần cấu hình cũ phải truyền `threshold` tường minh. Hai ngưỡng baseline CLI trước đây ghi nhiều chữ số hơn (0,440788/0,437976), nay thống nhất với 0,4408/0,4380 của benchmark và JSON lịch sử.

Các Champion giữ cùng plugin/hệ số đang dùng sau đợt 1. Frame/hop mặc định 25/10 ms, dải F0 70–400 Hz. Factory tạo plugin mới mỗi lần, không chia sẻ trạng thái. Các đối chứng đọc JSON ngưỡng hoặc nhận override vẫn ghi riêng nguồn cấu hình.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts/15_evaluate_unsupervised_thresholds.py --output-dir outputs/stage3_evaluation/recheck15
.\.venv\Scripts\python.exe -B scripts/16_evaluate_advanced_unsupervised.py --output-dir outputs/stage3_evaluation/recheck16
.\.venv\Scripts\python.exe -B scripts/03_compare_params.py --output_fig outputs/stage3_evaluation/recheck_frame.png --output_json outputs/stage3_evaluation/recheck_frame.json
```

- **79 test đạt:** 53 test đợt 1–2 và 26 test mới. Có phản ví dụ zero/missing reference, tổng hợp không bỏ file thiếu, Gaussian thiếu lớp, script 03/04/05/14/15/16, CLI nhiều file, và so sánh contour giữa API/CLI/benchmark.
- **15 dòng CSV benchmark 15** khớp tuyệt đối ngưỡng đầy đủ độ chính xác trong JSON và giá trị dùng để đánh giá; không dòng nào bị thay bằng 0.
- **21 bản ghi chiến lược benchmark 16** giữ nguyên toàn bộ chỉ số so với cuối đợt 2.
- **48 cặp cấu hình/file** (sáu baseline/Champion trên tám WAV) đã chạy lại. Cả 12 điểm tổng hợp train/test giữ nguyên mốc sau đợt 1. Điều này không có nghĩa API mặc định AMDF/Cepstrum cũ giữ nguyên hành vi; các cấu hình benchmark đã truyền đúng ngưỡng mới từ trước.
- Script 03 xuất đủ ba cấu hình khung; CLI xuất đủ sáu dòng cho `phone_F2.wav`; ba thuật toán đều từ chối fit Gaussian có giám sát từ LAB train hiện tại (V=0, UV=0).
- Dữ liệu gốc, báo cáo cũ và checksum archive được kiểm tra giữ nguyên.

Bằng chứng và đường dẫn đầu ra nằm trong [outputs/stage3_evaluation](../outputs/stage3_evaluation/README.md), tóm tắt máy đọc được tại [validation.json](../outputs/stage3_evaluation/validation.json).

## Chưa thuộc đợt này

Chưa hiệu chỉnh ngưỡng/hệ số, truy vết đầy đủ chênh lệch Cepstrum Champion, chạy lại toàn bộ plugin sweep trên train rồi chốt cấu hình trước test, hoặc đối chứng các giao thức fit không giám sát. Các việc đó thuộc đợt 4. Thiếu nhãn thời gian vẫn là giới hạn dữ liệu thực tế, không được khắc phục bằng cách đổi cách tính điểm.
