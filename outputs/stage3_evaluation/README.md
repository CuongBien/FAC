# Kiểm chứng sửa đợt 3 — 30/10/2026

- `validation.json`: 79 test, kiểm tra CSV, kết quả không đổi và việc từ chối fit Gaussian thiếu nhãn.
- `pipeline_after.json`: chi tiết sáu cấu hình trên cả train và test (48 cặp cấu hình/file).
- `pipeline_comparison.json`: so sánh 12 điểm tổng hợp với mốc sau đợt 1; không đổi điểm.
- `benchmark15/reports/`: JSON và CSV 15 dòng, cùng ID chiến lược và ngưỡng đầy đủ độ chính xác; tên hiển thị có cột riêng.
- `benchmark15/figures/`: ba biểu đồ tương ứng.
- `benchmark16/reports/` và `figures/`: 21 bản ghi chiến lược cùng bốn hình sau đồng bộ đánh giá; chỉ số giữ nguyên cuối đợt 2.
- `frame_comparison.{json,png}`: script 03 trên `phone_M1.wav` với khung 20/25/30 ms; Accuracy/F1 không có nhãn nên là null/N/A.
- `cli/`: CSV từng khung, PNG và summary sáu dòng trên `phone_F2.wav` với các cấu hình chung.

Các artifact này tách khỏi báo cáo lịch sử và đầu ra đợt 1/2. Chạy lại vào thư mục mới nếu muốn giữ đối chiếu. Xem [nhật ký sửa đợt 3](../../docs/STAGE3_FIXES.md) về thay đổi API mặc định, schema benchmark 15 và các giới hạn còn lại.
