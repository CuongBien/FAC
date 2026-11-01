# Danh mục kết quả

Danh mục tự dựng: [registry.json](registry.json). `python -B -m src.experiments list` liệt kê run và `verify` kiểm chứng mà không chạy mô hình. [Refactor đợt 2](refactor_phase2/README.md) có 152 test và đối chiếu đủ 288 trường hợp khớp chính xác; [bằng chứng đợt 1](refactor_phase1/README.md) và run cũ được giữ nguyên. Xem [quản lý thí nghiệm](../docs/EXPERIMENTS.md) và [cấu trúc pipeline](../docs/PIPELINE.md).

Kết quả hiện hành sau hiệu chỉnh: [stage4_validation/](stage4_validation/README.md), với cấu hình chốt trên train trước test. [Báo cáo đợt 4](../docs/STAGE4_RESULTS.md) giải thích cải thiện ACF/AMDF, suy giảm Cepstrum và lỗi pipeline lai trong artifact lịch sử.

Rà soát main ngày 01/11/2026: [benchmark 15/16 sau sửa Otsu](main_review_2026-11-01/README.md). Bộ Stage 4 đã được sửa snapshot/checksum và kiểm tra lại; 100 test hồi quy đạt. Điểm Stage 4 đã đóng băng vẫn giữ phiên bản thuật toán của lần chạy đó.

Tiếp theo cùng ngày: [benchmark tích hợp tám phương pháp với Champion](champion_unsupervised_2026-11-01/README.md), 288 kết quả/file, 93.423 dòng contour đã kiểm tra và 24 đối chứng native khớp hoàn toàn. Hiện 108 test hồi quy đạt. Mô hình train và mô hình thích nghi từng input được lưu/đánh giá riêng; không thay preset theo điểm test.

Hoàn tất ba lỗi biên sau cùng: [final_review_2026-11-01](final_review_2026-11-01/README.md). Hiện **120 test đạt**; code mới khớp toàn bộ 288 trường hợp và 93.423 dòng contour của benchmark cũ, không fit lại hoặc sửa freeze. Script 19 đã được chạy thật và xác nhận giữ nguyên văn bản thủ công của bảy tài liệu.

## Báo cáo tại `reports/`

Các báo cáo chính đã được tạo lại ở commit `43b1de0` ngày 31/10 sau bốn đợt sửa. Riêng các mô hình Gaussian `threshold_*.json` vẫn là lịch sử. Otsu ở báo cáo 15/16 trong `reports/` còn là bản trước sửa ngày 01/11; kết quả mới nằm trong thư mục rà soát riêng.

| Artifact trong `reports/` | Nội dung và trạng thái |
|---|---|
| `compare_acf_amdf_cepstrum.{json,csv}` | Benchmark test sáu cấu hình, có cấu hình threshold/frame/hop/plugin. Champion dùng preset chọn trên train. |
| `huanluyen_compare_acf_amdf_cepstrum.{json,csv}` | Benchmark train tương ứng sáu cấu hình. |
| `compare_plugins*.{json,csv}` | Nhóm đại diện/top trên train; `*_raw.json` và `*_per_file.csv` lưu đủ 16 tổ hợp. |
| `threshold_*.json` | Phân bố và ngưỡng Gaussian lịch sử. LAB hiện hành thiếu segments để huấn luyện lại; các script vẫn có thể dùng ngưỡng đã lưu. |
| `15_unsupervised_thresholds_results.{json,csv}` | Bốn phương pháp 1D; CSV đã sửa mapping ngưỡng. Giữ Otsu trước sửa cạnh bin ngày 01/11. |
| `16_advanced_unsupervised_results.{json,csv}` | Bản sau sửa GMM và HMM reset qua silence. Giữ Otsu trước sửa ngày 01/11; không phải benchmark tích hợp Champion. |

`figures/` giữ demo ba thuật toán, hình Gaussian lịch sử, ranking các nhóm cấu hình, hình benchmark 14–16 và minh họa tiền xử lý. Hình benchmark có cùng giới hạn như báo cáo tương ứng.

## Dự đoán CLI

[predictions/](predictions/README.md) tách `train/` và `test/`, mỗi thư mục có summary 12 dòng cho ba Champion, đã tạo lại ở commit `43b1de0`. `stage1_smoke/` là lần kiểm tra CLI lịch sử sau sửa trên `phone_F2.wav`, gồm sáu cấu hình.

## Kiểm chứng sửa đợt 1

`reports/stage1_pipeline/` chứa `before.json`, `after.json` (chi tiết sáu cấu hình trên train/test) và `validation.json` (tổng hợp điểm, 27 test hồi quy, 48 lượt kiểm tra pipeline/file). Ngưỡng không được hiệu chỉnh; điểm tăng ở nhiều cấu hình. Xem [nhật ký đợt 1](../docs/STAGE1_FIXES.md).

## Kiểm chứng sửa đợt 2

[stage2_thresholds/](stage2_thresholds/README.md) giữ benchmark 16 trước/sau (JSON/CSV và bốn hình mỗi phiên bản), cùng `validation.json`: 53 test, ba kiểm tra giao điểm GMM và 24 cặp thuật toán/file. [Nhật ký đợt 2](../docs/STAGE2_FIXES.md) mô tả giao thức fit và điểm thay đổi, gồm cả những trường hợp sai số tăng.

## Kiểm chứng sửa đợt 3

[stage3_evaluation/](stage3_evaluation/README.md) giữ benchmark 15/16, script 03, CLI và kết quả sáu cấu hình tại thời điểm đợt 3. 79 test đạt; 15 dòng CSV ghi đúng ngưỡng; 21 bản ghi benchmark 16 và 12 điểm tổng hợp baseline/Champion giữ nguyên các mốc trước đợt 3. CSV dùng schema ID chiến lược. Xem [nhật ký đợt 3](../docs/STAGE3_FIXES.md).

## Kiểm chứng và hiệu chỉnh đợt 4

90 test đạt tại mốc đợt 4; hiện có 100 test đạt sau rà soát main. `stage4_validation/` lưu toàn bộ 222 trial, freeze và checksum, snapshot source, 17 giao thức fit, raw result từng file, contour/prediction và ba hình so sánh. Các thư mục stage1–stage3 giữ mốc đối chiếu riêng. CSV dự đoán dùng dạng dài theo từng phương pháp để hỗ trợ khung 25/30 ms cùng một lần chạy.

## Artifact đã lưu riêng

Các file dự đoán trộn phiên bản, khảo sát frame length và hình/bảng so sánh cấu hình cũ đã chuyển vào [archive](../archive/2026-10-28-pre-cleanup/README.md). Manifest lưu checksum từng file.

Chạy lại các script có thể ghi đè tên đầu ra cố định. Để tái hiện, dùng [hướng dẫn kiểm thử](../docs/TESTING.md); các lỗi chưa sửa được theo dõi tại [KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md).
