# Các vấn đề cần xử lý

Ghi nhận ngày 28/10/2026, cập nhật ngày 30/10 sau đợt 4 trên nhánh `codex/fix-stage1-pipeline`. [Đợt 1](STAGE1_FIXES.md), [đợt 2](STAGE2_FIXES.md), [đợt 3](STAGE3_FIXES.md) sửa đầu vào, thuật toán và đánh giá. [Đợt 4](STAGE4_RESULTS.md) đã hiệu chỉnh trên train, chốt trước test, lưu đủ raw result và đối chứng 17 giao thức. Chênh lệch Cepstrum lịch sử đã truy ra: commit tạo artifact thiếu nhánh Cepstrum trong plugin detector, nên dùng ACF cho quyết định V/UV và Cepstrum trong hậu xử lý. Replay và thay duy nhất file detector đã tái lập hai mốc điểm. Các giới hạn trong bảng sau vẫn còn.

| Ưu tiên | Vấn đề | Bằng chứng / bước xử lý |
|---|---|---|
| Cao | Cepstrum chưa tổng quát tốt | Preset chọn trên train giảm lỗi train từ 15,160% xuống 11,390%, nhưng tăng lỗi test từ 21,203% lên 22,942%. Giữ lựa chọn đã chốt, cần thêm dữ liệu train và tập kiểm thử độc lập trước vòng phát triển mới. |
| Cao | Thiếu nhãn thời gian để huấn luyện/đánh giá từng khung | Tám LAB chỉ có F0mean/std/num. Đợt 3 đã chặn fit Gaussian với V=0/UV=0 và giữ Accuracy/F1 là N/A; vẫn cần nguồn nhãn thời gian để huấn luyện/đánh giá từng khung. |
| Vừa | Dữ liệu ít và test đã được quan sát trong lịch sử | Đợt 4 đã đối chứng train_all/train_active và 2D/HMM fit trên train so với thích nghi từng input. Chỉ có bốn file train và bốn file test; cần một tập mới độc lập để đo khả năng tổng quát. |
| Sau | Ngưỡng mới chưa tích hợp CLI/plugin chung | Các phương pháp không giám sát hiện nằm trong benchmark 15/16. Chưa đánh giá kết hợp đầy đủ với plugin Champion. |
| Sau | Phát hành wheel cần kiểm chứng thêm | Package discovery đã bao gồm các plugin con; chưa kiểm thử cài wheel trong môi trường hoàn toàn mới. |
| Sau | Đầu ra môn học | Chưa thấy Word, slide và bằng chứng đo T0 thủ công trong checkout. Hoàn thiện theo `HuongDan.txt`. |

## Giới hạn diễn giải

- Final Score chỉ so sánh thống kê toàn file; không phải tỷ lệ lỗi từng khung.
- Gaussian 614/180 mẫu và Accuracy/F1 trong các snapshot dùng nhãn trước đây; không trộn với LAB hiện hành.
- Tỷ lệ voiced khoảng 45,6% train/45,7% test lấy trên toàn tín hiệu, có cả silence ở phần còn lại; không phải chứng minh tỷ lệ voiced/unvoiced speech luôn 50/50.
- Các phát biểu “khử sạch 100%”, “0 khung lỗi” hoặc tỷ lệ octave error cần contour/nhãn từng khung hoặc phép đo bổ sung.
- Các benchmark nhiễu/PTDB-TUG/YIN trong ZIP là lịch sử, chưa chạy lại bằng code hiện tại.
