# Bản lưu trước dọn dẹp — 28/10/2026

Nguồn: checkout tại commit `427c0a4`. Các file được sao chép/di chuyển nguyên byte; đường dẫn nguồn, lý do, kích thước và SHA-256 nằm trong [manifest.json](manifest.json).

## Nội dung

- [README cũ](original-README.md) và năm chương trong `docs/`: bản trước khi chỉnh cấu trúc tài liệu, đường dẫn ảnh và chú thích trạng thái kiểm chứng. Các liên kết tương đối trong snapshot vẫn theo vị trí nguồn ban đầu.
- [Hướng dẫn ngày mai cũ](HUONG_DAN_KIEM_THU_NGAY_MAI.md), [hướng dẫn kiểm thử thầy cũ](HUONG_DAN_KIEM_THU_THAY.md): còn lệnh YIN và cấu hình trước đây. Hướng dẫn đang dùng ở [docs/TESTING.md](../../docs/TESTING.md).
- `BTL_SpeechProcessing_F0_Submission.zip`: gói ngày 26/10/2026, chứa code/tài liệu YIN, AWGN và PTDB-TUG; không có dữ liệu WAV/LAB. ZIP được giữ cục bộ, vẫn bị bỏ qua bởi quy tắc `*.zip` của Git. SHA-256 trong manifest giúp kiểm tra khi sao lưu/chuyển máy.
- `outputs/predictions/`: toàn bộ lần chạy dự đoán cũ, gồm các CSV có cột YIN và summary chỉ của một phần dữ liệu.
- `outputs/figures/`: khảo sát frame length, baseline/Enhanced của script 04 và so sánh cấu hình plugin cũ của script 05.
- `outputs/reports/`: `compare_params.json`, `test_results*.json`, `test_summary*.csv` tương ứng.

## Cách sử dụng

Đây là dữ liệu lịch sử để truy vết; không sử dụng như kết quả chạy bằng checkout sau dọn dẹp. Không chỉnh sửa snapshot để làm khớp số liệu mới. Cần khôi phục một file thì tra `original_path` trong manifest và sao chép về đường dẫn đó.

Năm wrapper plugin và năm helper đã bỏ khỏi mã nguồn hiện hành vẫn có trong lịch sử Git ở commit nguồn; manifest liệt kê tên/path. Cache Python có thể tạo lại và đã được xóa.

Các báo cáo Gaussian và benchmark chính không được di chuyển vào đây; vẫn nằm ở `outputs/reports/` cùng ghi chú nguồn gốc trong [danh mục kết quả](../../outputs/README.md).
