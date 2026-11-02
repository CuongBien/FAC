# CLI

`main.py` chuyển lệnh `fac`/`python -m src` tới service sở hữu; không thực thi pipeline khi chỉ xem help/version. `predict.py` giữ nguyên hàm xử lý dự đoán đã chuyển từ root. `config.py` xuất spec từ checkout hoặc tài nguyên wheel, chỉ tạo file mới.

Các API cũ vẫn có wrapper. Xem [cách chạy và đóng gói](../../docs/CLI_AND_PACKAGING.md).
