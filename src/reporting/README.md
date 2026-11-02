# Reporting

`source → ReportView → renderer → immutable bundle → preview/apply`.

Run mới và run lịch sử đều được đọc/xác minh; việc render không chạy thuật toán. `bundle.create_report()` và `bundle.rebuild_report()` trả đường dẫn gói. `publish.preview()` chỉ tạo kế hoạch/diff; `publish.apply()` mới thay vùng marker. Sáu hàm vẽ ở `plots.py` còn được re-export từ đường import cũ.

Xem [hướng dẫn và hợp đồng](../../docs/REPORTING.md).
