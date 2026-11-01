# Kết quả CLI

Các lần chạy trước ngày dọn dẹp đã chuyển nguyên vẹn vào [archive](../../archive/2026-10-28-pre-cleanup/README.md).

Hai thư mục dưới đây đã được tạo lại ở commit `43b1de0` ngày 31/10/2026, sau bốn đợt sửa, với ba preset Champion chọn trên train:

- `train/`: bốn file TinHieuHuanLuyen, summary 12 dòng cho ba Champion.
- `test/`: bốn file TinHieuKiemThu, summary 12 dòng cho ba Champion.

Mỗi tập có bốn CSV từng khung, bốn PNG và một `evaluation_summary_metrics.csv`. Không có cột YIN. Accuracy/F1 để trống do LAB không có segments.

CSV từng khung dùng schema dài `Method,Frame_Index,Time_Sec,Label,F0_Hz`; Champion dùng khung/hop 30/10 ms. Cepstrum vẫn tăng sai số test sau hiệu chỉnh dù giảm trên train; xem [Stage 4](../stage4_validation/README.md). Sửa Otsu ngày 01/11 không thay đổi các preset này.

Sau sửa đợt 1, `stage1_smoke/` chứa CSV từng khung, PNG và summary sáu dòng (ba baseline, ba Champion) của `phone_F2.wav`. Điểm mới trên đầy đủ tám WAV nằm riêng ở `../reports/stage1_pipeline/`; xem [nhật ký đợt 1](../../docs/STAGE1_FIXES.md).

Tạo lần chạy khác bằng `predict.py -o outputs/predictions/<tên_lần_chạy>`. Summary chỉ bao gồm các file của lần chạy vừa thực hiện.
