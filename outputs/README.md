# Danh mục kết quả

Kết quả hiện hành sau hiệu chỉnh: [stage4_validation/](stage4_validation/README.md), với cấu hình chốt trên train trước test. [Báo cáo đợt 4](../docs/STAGE4_RESULTS.md) giải thích cải thiện ACF/AMDF, suy giảm Cepstrum và lỗi pipeline lai trong artifact lịch sử.

## Báo cáo giữ nguyên từ các lần chạy trước

Các nhận xét tái lập trong bảng được kiểm tra trước sửa pipeline đợt 1; không phải điểm của code sau sửa.

| Artifact trong `reports/` | Nội dung và trạng thái |
|---|---|
| `compare_acf_amdf_cepstrum.{json,csv}` | Benchmark test sáu cấu hình. ACF/AMDF và Cepstrum Baseline tái lập khớp; Cepstrum Champion chưa khớp. |
| `huanluyen_compare_acf_amdf_cepstrum.{json,csv}` | Benchmark train; cùng giới hạn Cepstrum Champion. |
| `compare_plugins*.{json,csv}` | Các cấu hình đại diện/top groups trên train; không lưu đủ raw result 16 tổ hợp. |
| `threshold_*.json` | Phân bố và ngưỡng Gaussian lịch sử. LAB hiện hành thiếu segments để huấn luyện lại; các script vẫn có thể dùng ngưỡng đã lưu. |
| `15_unsupervised_thresholds_results.{json,csv}` | Benchmark bốn phương pháp không giám sát. JSON tái lập khớp; cột Threshold_T trong CSV hiện bị 0 do lỗi mapping. |
| `16_advanced_unsupervised_results.{json,csv}` | Snapshot lịch sử trước đợt 1/2, còn công thức GMM cũ và HMM nối qua silence. Bản sửa nằm riêng ở `stage2_thresholds/`. |

`figures/` giữ demo ba thuật toán, hình Gaussian lịch sử, ranking các nhóm cấu hình, hình benchmark 14–16 và minh họa tiền xử lý. Hình benchmark có cùng giới hạn như báo cáo tương ứng.

## Dự đoán CLI

[predictions/](predictions/README.md) tách `train/` và `test/`, mỗi thư mục có summary riêng. Hai thư mục được sinh lại trong đợt dọn dẹp, trước sửa đợt 1; không dùng để thay thế các JSON benchmark cũ. `stage1_smoke/` là lần kiểm tra CLI sau sửa trên `phone_F2.wav`, gồm sáu cấu hình.

## Kiểm chứng sửa đợt 1

`reports/stage1_pipeline/` chứa `before.json`, `after.json` (chi tiết sáu cấu hình trên train/test) và `validation.json` (tổng hợp điểm, 27 test hồi quy, 48 lượt kiểm tra pipeline/file). Ngưỡng không được hiệu chỉnh; điểm tăng ở nhiều cấu hình. Xem [nhật ký đợt 1](../docs/STAGE1_FIXES.md).

## Kiểm chứng sửa đợt 2

[stage2_thresholds/](stage2_thresholds/README.md) giữ benchmark 16 trước/sau (JSON/CSV và bốn hình mỗi phiên bản), cùng `validation.json`: 53 test, ba kiểm tra giao điểm GMM và 24 cặp thuật toán/file. [Nhật ký đợt 2](../docs/STAGE2_FIXES.md) mô tả giao thức fit và điểm thay đổi, gồm cả những trường hợp sai số tăng.

## Kiểm chứng sửa đợt 3

[stage3_evaluation/](stage3_evaluation/README.md) giữ benchmark 15/16 mới, script 03, CLI và kết quả sáu cấu hình trên train/test. 79 test đạt; 15 dòng CSV ghi đúng ngưỡng; 21 bản ghi benchmark 16 và 12 điểm tổng hợp baseline/Champion giữ nguyên các mốc trước đợt 3. CSV benchmark 15 lịch sử trong bảng trên vẫn giữ lỗi cũ để đối chiếu; bản mới đã sửa và dùng schema ID chiến lược. Xem [nhật ký đợt 3](../docs/STAGE3_FIXES.md).

## Kiểm chứng và hiệu chỉnh đợt 4

90 test đạt. `stage4_validation/` lưu toàn bộ 222 trial, freeze và checksum, snapshot source, 17 giao thức fit, raw result từng file, contour/prediction và ba hình so sánh. Các artifact trước đợt 4 giữ nguyên; CSV dự đoán mới dùng dạng dài theo từng phương pháp để hỗ trợ khung 25/30 ms cùng một lần chạy.

## Artifact đã lưu riêng

Các file dự đoán trộn phiên bản, khảo sát frame length và hình/bảng so sánh cấu hình cũ đã chuyển vào [archive](../archive/2026-10-28-pre-cleanup/README.md). Manifest lưu checksum từng file.

Chạy lại các script có thể ghi đè tên đầu ra cố định. Để tái hiện, dùng [hướng dẫn kiểm thử](../docs/TESTING.md); các lỗi chưa sửa được theo dõi tại [KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md).
