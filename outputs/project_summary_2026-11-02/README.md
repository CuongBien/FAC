# Tổng hợp kết quả sau merge — 02/11/2026

Mã refactor đã gộp vào main tại `12ec529`. [Báo cáo đầy đủ](../../docs/PROJECT_RESULTS_2026-11-02.md) gồm dọn dẹp, sửa lỗi, preset, tám phương pháp Champion, 17 đối chứng raw Stage 4, benchmark sau sửa Otsu và các giới hạn.

- [aggregate_metrics.csv](aggregate_metrics.csv): 462 dòng aggregate, có family/profile và source SHA.
- [per_file_metrics.csv](per_file_metrics.csv): 1.704 kết quả từng file, không bỏ dữ liệu thiếu.
- [results.json](results.json): toàn bộ giá trị số dùng trong CSV.
- [verification.json](verification.json): archive 56/56, hai run legacy, source/test SHA, dữ liệu và checksum đầu vào.

Đọc artifact cũ; không chạy detector, fit lại hoặc sửa các run gốc. Những cấu hình/file lặp ở các family không phải mẫu độc lập mới. Code vẫn đúng bản đã đạt 217 test; báo cáo này chỉ bổ sung tài liệu/bảng dẫn xuất.
