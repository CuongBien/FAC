# Kiểm chứng ba lỗi biên sau cùng — 01/11/2026

Nguồn: [nhật ký sửa](../../docs/FINAL_EDGE_CASE_FIXES.md). Đây là phép đối chiếu mô hình đã đóng băng với code hiện tại, không phải lần fit/chọn mô hình mới.

- `validation.json`: 120 test đạt; 288 trường hợp và 93.423 dòng contour khớp chính xác kết quả trong `../champion_unsupervised_2026-11-01/`; 3.873 dòng đặc trưng train khớp. Có provenance riêng cho mã gốc và mã kiểm chứng mới, SHA của mô hình/cấu hình/evaluation và các file test.
- `verification_source_snapshot.tar.gz`: 49 file nguồn của lần kiểm chứng hiện tại. Không thay snapshot hoặc SHA trong freeze của lần benchmark cũ.
- `report_preservation.json`: đối chiếu các vùng văn bản thủ công của bảy tài liệu trước/sau chạy script 19 thật; toàn bộ được giữ nguyên.

WAV/LAB, preset Champion và các kết quả benchmark gốc được giữ nguyên. Hai định dạng LAB mới có test riêng: ký pháp khoa học và UTF-8 BOM. Cấu hình F0 sai được chặn trước khi xử lý im lặng. Script 19 chỉ ghi trong marker Stage 4 và từ chối dấu thiếu/trùng/đảo thứ tự trước mọi thay đổi đầu ra.
