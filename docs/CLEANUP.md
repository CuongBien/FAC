# Nhật ký dọn dẹp — 28/10/2026

Nguồn: commit `427c0a4`. Phạm vi đã thống nhất: lưu artifact cũ, hợp nhất hướng dẫn, bỏ code/import không dùng và kiểm tra các luồng hiện tại. Các thuật toán và tham số detector được giữ nguyên.

## Lưu trữ

- Di chuyển 50 file: ZIP nộp bài cũ, hai hướng dẫn kiểm thử, 17 file dự đoán, 25 hình và năm báo cáo cũ.
- Sao lưu README và năm chương báo cáo trước chỉnh sửa.
- Mỗi file có đường dẫn nguồn, đường dẫn lưu, lý do, kích thước và SHA-256 trong [manifest](../archive/2026-10-28-pre-cleanup/manifest.json).
- Giữ các báo cáo Gaussian, plugin sweep, benchmark 14–16 tại `outputs/reports/` để truy vết kết quả đã công bố.

## Mã nguồn

- Bỏ năm wrapper `src/plugins/{bandpass_filter,center_clipping,energy_extension,hysteresis,viterbi_tracking}.py`; script huấn luyện chuyển sang import từ `src.plugins.pre_processing`.
- Bỏ helper không còn consumer trong repo: `add_awgn_noise`, `compute_pitch_contour_metrics`, `extract_active_speech_frame_metrics`, `plot_plugin_contour_comparison`, `plot_combinations_ranking` và alias `plot_frame_analysis`.
- Cập nhật re-export, bổ sung export AMDF plot còn dùng và dọn 29 nhóm import không dùng.
- Xóa tám thư mục `__pycache__` trong mã nguồn và hai `.gitkeep` ở thư mục đã có nội dung.
- Package discovery bao gồm `src` và các package con; metadata mô tả đủ ba thuật toán.
- `.gitignore` bỏ quy tắc output dư thừa và ignore PTDB cũ, thêm build/dist; giữ quy tắc cache, môi trường và ZIP.

## Tài liệu và đầu ra

- README làm trang bắt đầu và mục lục; [TESTING.md](TESTING.md) thay hai hướng dẫn cũ.
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) theo dõi lỗi thuật toán và giới hạn đánh giá riêng.
- Bỏ bảng/ảnh đối sánh cấu hình cũ trùng lặp trong chương ACF/AMDF; snapshot gốc vẫn còn trong archive.
- Sửa đường dẫn ảnh `.gemini`, bỏ carousel đặc thù công cụ khác, sửa ký tự điều khiển LaTeX và thông tin sample rate. Thêm chú thích nguồn nhãn và trạng thái tái lập ở đầu từng chương.
- Các lần dự đoán mới tách `outputs/predictions/train/` và `test/`, mỗi tập có summary riêng. Báo cáo benchmark cũ không bị ghi đè.

## Kiểm chứng

- Kiểm tra SHA-256 của cả 56 file đã lưu: khớp manifest.
- 37 file Python còn lại phân tích cú pháp được; toàn bộ package và script import được.
- Đối chiếu AST với HEAD trước dọn: thân các hàm/class còn lại không đổi, bao gồm detector, plugin, GMM và HMM.
- Chạy bốn cấu hình ACF/AMDF baseline/Champion trên cả train/test: toàn bộ chỉ số tổng hợp và từng file khớp JSON cũ; test ACF Champion 4,22%, AMDF Champion 3,78%.
- Chạy CLI thật với ba Champion trên cả tám WAV: tạo tám CSV, tám PNG và hai summary, mỗi summary đủ 12 dòng cho bốn file × ba hệ thống.
- Kiểm tra CSV không còn cột YIN, Accuracy/F1 trống đúng với nhãn hiện hành; PNG đọc/kiểm tra được.
- Kiểm tra liên kết Markdown và ký tự điều khiển trong tài liệu hiện hành; các đường dẫn `.gemini` đã được thay.
- Đối chiếu dữ liệu WAV/LAB và báo cáo giữ lại với Git: nội dung không đổi, có tính tới chuyển đổi LF/CRLF của checkout Windows. `.venv` và `uv.lock` không bị sửa.

Chưa build/cài wheel vào môi trường mới. Các lỗi thuật toán đã biết vẫn giữ trong [KNOWN_ISSUES.md](KNOWN_ISSUES.md), không được che bằng việc thay báo cáo cũ.
