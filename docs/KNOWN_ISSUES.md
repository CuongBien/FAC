# Các vấn đề cần xử lý

> Cập nhật 02/11/2026: sáu đợt refactor đã merge main tại `12ec529`; **217 test đạt**. Wheel non-editable đã chạy ngoài repo/Git trên Windows Python 3.11 và 3.12; mục kiểm chứng wheel đã hoàn tất. CI Windows/Linux đã cấu hình nhưng chưa chạy remote. Xem [báo cáo tổng hợp kết quả](PROJECT_RESULTS_2026-11-02.md). Các ghi chú ngày 01/11 dưới đây là mốc lịch sử.

> Cập nhật 01/11/2026: [rà soát main](MAIN_REVIEW_FIXES.md) đã sửa Otsu và CLI với LAB thiếu F0std; đồng bộ preset/checksum/snapshot và thêm kiểm tra artifact. Bảng tích hợp tám phương pháp chưa có đủ bằng chứng đã được tách khỏi kết luận hiện hành. Các giới hạn dưới đây vẫn áp dụng.

> Cùng ngày, [benchmark 21](CHAMPION_UNSUPERVISED_RESULTS.md) đã hoàn tất tích hợp tám phương pháp với đúng đặc trưng Champion, lưu 288 kết quả/file và tách đối chứng fit từng input. Giao thức mới không được xem là tái lập các mốc cũ chưa lưu. Bộ hồi quy hiện đạt 108 test.

> Sau đó đã sửa thêm [parser LAB/BOM, quyền ghi script 19 và kiểm tra phạm vi F0](FINAL_EDGE_CASE_FIXES.md). **Hiện 120 test đạt**, 288 trường hợp đối chiếu bằng code mới vẫn khớp kết quả cũ; ba lỗi này đã được xử lý. Các giới hạn dữ liệu, CLI và đầu ra môn học trong bảng dưới vẫn còn.

Ghi nhận ngày 28/10/2026, cập nhật ngày 30/10 sau đợt 4 trên nhánh `codex/fix-stage1-pipeline`. [Đợt 1](STAGE1_FIXES.md), [đợt 2](STAGE2_FIXES.md), [đợt 3](STAGE3_FIXES.md) sửa đầu vào, thuật toán và đánh giá. [Đợt 4](STAGE4_RESULTS.md) đã hiệu chỉnh trên train, chốt trước test, lưu đủ raw result và đối chứng 17 giao thức. Chênh lệch Cepstrum lịch sử đã truy ra: commit tạo artifact thiếu nhánh Cepstrum trong plugin detector, nên dùng ACF cho quyết định V/UV và Cepstrum trong hậu xử lý. Replay và thay duy nhất file detector đã tái lập hai mốc điểm. Các giới hạn trong bảng sau vẫn còn.

| Ưu tiên | Vấn đề | Bằng chứng / bước xử lý |
|---|---|---|
| Cao | Cepstrum chưa tổng quát tốt | Preset chọn trên train giảm lỗi train từ 15,160% xuống 11,390%, nhưng tăng lỗi test từ 21,203% lên 22,942%. Giữ lựa chọn đã chốt, cần thêm dữ liệu train và tập kiểm thử độc lập trước vòng phát triển mới. |
| Cao | Thiếu nhãn thời gian để huấn luyện/đánh giá từng khung | Tám LAB chỉ có F0mean/std/num. Đợt 3 đã chặn fit Gaussian với V=0/UV=0 và giữ Accuracy/F1 là N/A; vẫn cần nguồn nhãn thời gian để huấn luyện/đánh giá từng khung. |
| Vừa | Dữ liệu ít và test đã được quan sát trong lịch sử | Đợt 4 đã đối chứng train_all/train_active và 2D/HMM fit trên train so với thích nghi từng input. Chỉ có bốn file train và bốn file test; cần một tập mới độc lập để đo khả năng tổng quát. |
| Sau | CLI chưa có lựa chọn tám phương pháp mới | Benchmark 21 đã tích hợp đủ với hậu xử lý native của Champion; chưa thêm tùy chọn tương ứng vào `predict.py`. Preset mặc định được giữ nguyên, không chọn theo thứ hạng test mới. |
| Sau | Đầu ra môn học | Chưa thấy Word, slide và bằng chứng đo T0 thủ công trong checkout. Hoàn thiện theo `HuongDan.txt`. |

## Giới hạn diễn giải

- Final Score chỉ so sánh thống kê toàn file; không phải tỷ lệ lỗi từng khung.
- Gaussian 614/180 mẫu và Accuracy/F1 trong các snapshot dùng nhãn trước đây; không trộn với LAB hiện hành.
- Tỷ lệ voiced khoảng 45,6% train/45,7% test lấy trên toàn tín hiệu, có cả silence ở phần còn lại; không phải chứng minh tỷ lệ voiced/unvoiced speech luôn 50/50.
- Các phát biểu “khử sạch 100%”, “0 khung lỗi” hoặc tỷ lệ octave error cần contour/nhãn từng khung hoặc phép đo bổ sung.
- Các benchmark nhiễu/PTDB-TUG/YIN trong ZIP là lịch sử, chưa chạy lại bằng code hiện tại.
