# Kết quả CLI

Các lần chạy trước ngày dọn dẹp đã chuyển nguyên vẹn vào [archive](../../archive/2026-10-28-pre-cleanup/README.md).

Trong đợt dọn dẹp ngày 28/10/2026, trước sửa pipeline đợt 1, chạy lại ba Champion trên tám WAV, tách đầu ra thành:

- `train/`: bốn file TinHieuHuanLuyen, summary 12 dòng cho ba Champion.
- `test/`: bốn file TinHieuKiemThu, summary 12 dòng cho ba Champion.

Mỗi tập có bốn CSV từng khung, bốn PNG và một `evaluation_summary_metrics.csv`. Không có cột YIN. Accuracy/F1 để trống do LAB không có segments.

Cepstrum Champion của lần dọn dẹp đã khác JSON benchmark cũ trong `../reports/`; xem [KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md).

Sau sửa đợt 1, `stage1_smoke/` chứa CSV từng khung, PNG và summary sáu dòng (ba baseline, ba Champion) của `phone_F2.wav`. Điểm mới trên đầy đủ tám WAV nằm riêng ở `../reports/stage1_pipeline/`; xem [nhật ký đợt 1](../../docs/STAGE1_FIXES.md).

Tạo lần chạy khác bằng `predict.py -o outputs/predictions/<tên_lần_chạy>`. Summary chỉ bao gồm các file của lần chạy vừa thực hiện.
