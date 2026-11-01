# Sửa lỗi sau rà soát main — 01/11/2026

Nhánh `codex/fix-main-review`, xuất phát từ `main` tại `5a9273f`. Hai commit mới sau đợt 4 tạo lại outputs và bổ sung tài liệu; không thay đổi mã Python. Các sửa dưới đây xử lý lỗi còn sót và tính nhất quán của bằng chứng.

## Otsu và CLI

Otsu trả cạnh phải của bin cuối thuộc nhóm thấp, thay vì tâm bin. Ví dụ `[0.1, 0.4, 0.6, 0.9]` với hai bin phải trả 0.5, để giữ nguyên cách chia histogram đã tối ưu; trước đây trả 0.3 và xếp nhầm 0.4. Dữ liệu hằng trả giá trị quan sát, không dùng đầu histogram mở rộng giả của NumPy. Đầu vào phải một chiều và có ít nhất hai bin; giá trị NaN/Inf vẫn bị loại trước khi fit. Trường hợp hằng không chứng minh có hai lớp phân biệt.

CLI kiểm tra `F0std is not None` trước khi vẽ dải tham chiếu. LAB có mean/count nhưng thiếu std vẫn xuất PNG/CSV; metric thiếu giữ N/A/ô trống và Final Score không được tính bằng hai thành phần.

Benchmark 15/16 chạy lại trong thư mục riêng, fit ngưỡng trên 1.291 khung train gồm silence, đánh giá bốn file test với khung/hop 25/10 ms. Đây không phải benchmark tích hợp Champion. Otsu cho kết quả sau; thấp hơn tốt hơn:

| Thuật toán | Ngưỡng cũ → mới | Final Score cũ → mới (%) |
|---|---:|---:|
| ACF | 0,455915 → 0,460244 | 6,35 → 6,45 |
| AMDF | 0,394971 → 0,399144 | 7,25 → 7,15 |
| Cepstrum | 0,190325 → 0,193046 | 53,02 → 53,68 |

Sửa logic không đảm bảo giảm mọi điểm test. Không chọn lại tham số để lấy điểm test thấp hơn. Các chiến lược không bị ảnh hưởng trong hai benchmark giữ kết quả cũ. [JSON/CSV và hình mới](../outputs/main_review_2026-11-01/README.md).

## Bộ bằng chứng Stage 4

Freeze/evaluation được tạo lại ngày 31/10 nhưng vẫn giữ snapshot/runtime/validation ngày 30/10. Preset xuất ra cũng còn checksum cũ. Các giá trị Champion vẫn khớp, nên không cần hiệu chỉnh hay thay preset.

- Đồng bộ checksum preset với `frozen_config.json` hiện có: `83dfda76c5a2ce34846a8c4eff9132296e2c09846e4d8e2cf44bbac5be884dc5`.
- Giữ snapshot/runtime/validation cũ trong `outputs/stage4_validation/historical_source_snapshot/`.
- Phục dựng 46 file nguồn theo SHA đã ghi trong provenance, từ byte Git tại `a9073ff` hoặc working tree khi đúng SHA. Chỉ chuyển LF/CRLF nếu kết quả khớp SHA; không chèn mã vừa sửa vào snapshot cũ.
- Kiểm tra hai freeze, liên kết hash của evaluation, snapshot, runtime và giá trị/checksum preset trước khi ghi kết luận mới. `validation.json` không sao chép các cờ kiểm chứng của lần cũ.
- Script 17 tự lưu snapshot/runtime từ đầu những lần train sau. Script 19 từ chối sinh báo cáo nếu freeze/evaluation/snapshot/runtime không khớp.

Đây là sửa bằng chứng nguồn, không phải chạy lại Stage 4. Điểm và 17 giao thức Stage 4 vẫn là snapshot trước sửa Otsu; mô hình đã đóng băng không tự được fit lại. Champion không dùng Otsu nên ba preset và điểm Champion giữ nguyên.

```powershell
uv run python -B scripts/20_verify_validation_artifacts.py
uv run python -B scripts/20_verify_validation_artifacts.py --run-tests --write-report
```

## Báo cáo và giới hạn

Chương V đã phân biệt baseline không plugin 25 ms với Champion 30 ms dùng ngưỡng lịch sử. Bảng bổ sung trước đây được giữ trong [archive có nhãn chưa kiểm chứng](../archive/2026-11-01-main-review/champion_unsupervised_unverified.md). Các mốc HMM Cepstrum 14,31%, GMM BP+Clip 14,40% và Adaptive ACF 2,17% chưa có đủ script/mô hình/kết quả từng file tương ứng, nên không được dùng làm kỷ lục hoặc lý do thay preset. Đợt tiếp theo cùng ngày đã hoàn tất [benchmark tám phương pháp với giao thức được lưu đầy đủ](CHAMPION_UNSUPERVISED_RESULTS.md); không coi giao thức mới là tái lập giao thức cũ chưa lưu.

Đã bỏ kết luận coi số liệu toàn file là chứng minh giả thiết V/UV 50/50. Non-voiced hiện gộp UV và silence; Otsu không ép hai nhóm có kích thước bằng nhau. LAB không có nhãn thời gian, nên Final Score không chứng minh đúng contour, Accuracy/F1 hoặc loại hết octave error.

## Kiểm tra

Bộ hồi quy đạt **100/100 test**, gồm 90 test trước đây và 10 test mới trong `tests/test_main_review.py`: ngưỡng Otsu bảo toàn cách chia histogram, dữ liệu hằng, đầu vào sai, CLI xuất thật PNG/CSV khi thiếu std, freeze lệch với evaluation, preset lệch checksum, snapshot/runtime sai, phục dựng đúng byte, bảo toàn bằng chứng cũ khi sửa và chặn đường dẫn vượt repository. Kết quả thực chạy nằm tại [validation.json](../outputs/stage4_validation/validation.json). Kiểm tra cú pháp Python 3.9 trên 52 file cũng đạt; các dòng benchmark không dùng Otsu giữ nguyên.
