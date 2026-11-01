# Hoàn tất ba lỗi biên — 01/11/2026

Nhánh `codex/fix-main-review`. Đây là phần tiếp theo sau rà soát main và benchmark tám phương pháp. Không thay preset hoặc chọn lại mô hình theo điểm test.

## Parser LAB

- Đọc cả UTF-8 thường và UTF-8 có BOM bằng `utf-8-sig`.
- Đọc trọn số thập phân/ký pháp khoa học: `1.8e2`, `+1.8E+02`, `.1e2`, `2.e2`. Không cắt `1.8e2` thành `1.8`.
- Các dòng F0mean/F0std/F0num nhận đúng một giá trị, có thể có chú thích `#` sau giá trị. Số không hợp lệ, âm, NaN/Inf hoặc tràn số bị từ chối với tên file và số dòng; không âm thầm lấy phần đầu chuỗi. F0num phải là số nguyên không âm.
- Trường thiếu vẫn là None, giữ quy tắc N/A/CSV trống. Các đoạn thời gian giữ cách đọc hiện có.

Tám LAB gốc vẫn có đúng các giá trị tham chiếu cũ; không chỉnh dữ liệu để thích nghi với parser mới. CLI đã được chạy thật với LAB có BOM và số mũ, xuất PNG/CSV và dùng đúng các tham chiếu 180 Hz / 10 Hz / 200 khung.

## Phạm vi F0

Quy tắc chung: sample rate và giới hạn phải hữu hạn, `0 < f0_min < f0_max <= sample_rate / 2`. ACF, AMDF, Cepstrum kiểm tra trước nhánh trả sớm cho khung rỗng/im lặng. Hai detector cũng kiểm tra trước silence gate, nên cấu hình sai không thể bị che thành kết quả không voiced.

Không đổi cách chọn lag, nội suy hay tham số mặc định 70–400 Hz. Tín hiệu im lặng với cấu hình hợp lệ vẫn trả F0 bằng 0 như trước.

## Script 19 và quyền ghi tài liệu

Script 19 chỉ cập nhật các khối được đánh dấu:

| Tài liệu | Khối được quản lý |
|---|---|
| README | `STAGE4_README_START` / `STAGE4_README_END`: bảng Champion Stage 4 |
| docs/STAGE4_RESULTS.md | `STAGE4_REPORT_START` / `STAGE4_REPORT_END`: nội dung sinh từ Stage 4 |
| Năm chương báo cáo | `STAGE4_CURRENT_START` / `STAGE4_CURRENT_END`: phần bổ sung Stage 4 |

Tiêu đề, ghi chú sửa snapshot, giới thiệu benchmark tám phương pháp, các chương/phần mới và liên kết nằm ngoài khối được giữ nguyên. Không viết lại toàn bộ mục “Kết quả và trạng thái kiểm chứng” của README nữa. Mỗi cặp dấu phải xuất hiện đúng một lần và đúng thứ tự; công cụ kiểm tra toàn bộ trước khi ghi bất kỳ hình, tài liệu hoặc summary nào.

Đã chạy script 19 thật trên repository và đối chiếu phần thủ công của cả bảy tài liệu trước/sau. Đã kiểm thử chạy lặp lại, thiếu dấu và dấu trùng/đảo thứ tự. `outputs/stage4_validation/report_generation.json` ghi SHA của script, nguồn dữ liệu và tài liệu tại thời điểm sinh; không thay provenance train trong freeze.

## Kiểm chứng kết quả sau sửa

- **120/120 test đạt**: 108 test trước đây và 12 test mới trong `tests/test_final_edge_cases.py`.
- Đối chiếu lại **288 trường hợp** bằng mô hình cũ và code mới, không fit hoặc chọn lại mô hình.
- **93.423 dòng contour** khớp chính xác label, tâm khung và F0; tất cả metric từng file cũng khớp.
- **3.873 dòng đặc trưng train** khớp periodicity, ZCR, candidate F0 và timeline. Điều này xác nhận đầu vào của các mô hình đã fit không bị đổi.
- Freeze cấu hình/mô hình, evaluation gốc và WAV/LAB được giữ nguyên. Snapshot cũ vẫn mô tả đúng mã của lần chạy cũ; không sửa SHA trong freeze để giả làm nguồn hiện tại.
- Lưu riêng snapshot 49 file nguồn hiện tại cùng SHA của bảy file test; kiểm tra cú pháp Python 3.9 trên **56 file** đạt.

Các điểm Champion và benchmark tám phương pháp đã công bố không thay đổi. Ví dụ Cepstrum ngưỡng động vẫn 16,480% trên test; 2D fit từng input vẫn 10,375% và vẫn thuộc nhóm thích nghi riêng.

Bằng chứng máy đọc và snapshot kiểm chứng mới: [outputs/final_review_2026-11-01](../outputs/final_review_2026-11-01/README.md). Khi chạy một thí nghiệm mới bằng code hiện tại, dùng thư mục đầu ra mới; pha evaluate của benchmark 21 vẫn từ chối ghép freeze cũ với source đã đổi. Phép đối chiếu sau sửa được ghi riêng, không ghi đè run cũ.
