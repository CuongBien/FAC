# Sửa pipeline đợt 1 — 28/10/2026

Nhánh thực hiện: `codex/fix-stage1-pipeline`. Phạm vi là tính đúng và nhất quán của đầu vào, tìm F0 và hậu xử lý; chưa hiệu chỉnh lại ngưỡng hoặc thay các phương pháp tìm ngưỡng không giám sát.

## Những lỗi đã sửa

| Mục | Hành vi sau sửa |
|---|---|
| PCM 8-bit và stereo | PCM unsigned được trừ midpoint 128 trước khi chuẩn hóa; chuẩn hóa từng kênh trước khi lấy mono. PCM signed dùng thang của dtype. |
| Zero và DC | STE tính trên khung đã trừ trung bình; silence có ngưỡng năng lượng dương tối thiểu. Các hàm tìm pitch từ chối khung không có năng lượng AC, kể cả AMDF khi ngưỡng silence tương đối bằng 0. |
| Input quá ngắn | Detector yêu cầu ít nhất một khung đầy đủ; bandpass kiểm tra riêng số mẫu cần cho padding của `filtfilt`. CLI báo lỗi rõ ràng và thoát mã 2 với lỗi đầu vào này. |
| Nội suy F0 | ACF/AMDF và các ứng viên Viterbi dùng chung giới hạn lag và nội suy parabol có chặn; F0 cuối nằm trong dải cấu hình. |
| Viterbi | Dùng đúng các khung mà detector đã tiền xử lý, bao gồm bandpass/center clipping và cấu hình frame/hop/F0 thực tế. Không tự chia lại tín hiệu theo 25/10 ms. |
| Median | Lọc riêng từng đoạn voiced liên tục trong detector, Energy Extension và benchmark 16; không nối hai đoạn qua khoảng silence/unvoiced. |
| Gap filling | Khoảng UV ngắn chỉ được lấp khi có ứng viên F0 hợp lệ, periodicity đạt ngưỡng nới lỏng, đủ năng lượng và liên tục với hai đầu. Dùng F0 ứng viên thực tế, không tự tạo pitch bằng nội suy qua silence. |

Các quy tắc dùng chung nằm trong `src/core/processing.py`. Kết quả plugin mang thêm `pitch_frames` và metadata cấu hình để hậu xử lý dùng đúng dữ liệu. Gọi riêng Viterbi với kết quả thiếu `pitch_frames` sẽ nhận `ValueError` thay vì tự dựng lại khung khác.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B predict.py -i TinHieuKiemThu/phone_F2.wav -m all -o outputs/predictions/stage1_smoke
```

- 27 test hồi quy đạt, có các trường hợp PCM mono/stereo, zero/DC, file ngắn, biên F0 ở nhiều sample rate, median qua khoảng nghỉ, periodicity khi lấp gap, Viterbi với khung 30/20 ms và dải F0 riêng. Đường đi quy hoạch động được đối chiếu với vét cạn trên ví dụ nhỏ.
- 48 cặp cấu hình/file đã chạy: sáu cấu hình baseline/Champion trên bốn WAV train và bốn WAV test. Tất cả đạt: F0 hữu hạn; chỉ voiced có F0 dương; voiced trong 70–400 Hz; số voiced khớp contour; chiều dài time/label/contour khớp nhau.
- CLI trên `phone_F2.wav` xuất được CSV từng khung, PNG và summary sáu dòng vào thư mục riêng [stage1_smoke](../outputs/predictions/stage1_smoke/). Accuracy/F1 vẫn để trống vì LAB không có nhãn từng khung.

Các kiểm tra này xác nhận các lỗi nêu trên và tính nhất quán đầu ra. Chúng không chứng minh pitch đúng tại mọi khung của tiếng nói thật.

## Điểm thống kê trước và sau

Final Score (%), **thấp hơn là tốt hơn**. Cột trước là chạy code ngay trước sửa đợt 1, sau đợt dọn dẹp; không lấy số Cepstrum Champion từ JSON lịch sử vốn đã không tái lập khớp.

| Cấu hình | Train trước | Train sau | Test trước | Test sau |
|---|---:|---:|---:|---:|
| ACF Baseline | 13,23 | 13,89 | 5,55 | 5,82 |
| ACF + BP + Energy + Viterbi | 2,95 | 3,88 | 4,22 | 4,22 |
| AMDF Baseline | 7,01 | 6,76 | 4,02 | 6,36 |
| AMDF + Energy + Viterbi | 3,29 | 3,76 | 3,78 | 4,12 |
| Cepstrum Baseline | 39,81 | 44,54 | 42,95 | 48,19 |
| Cepstrum + BP + Clip + Energy + Viterbi | 6,84 | 15,16 | 13,24 | 21,20 |

**Điểm chưa cải thiện ở phần lớn cấu hình, đặc biệt Cepstrum.** Ngưỡng và hệ số plugin vẫn giữ nguyên. Các sửa đổi DC, median, điều kiện lấp gap và dữ liệu ứng viên Viterbi làm thay đổi phân bố/kết quả; cần đánh giá riêng ảnh hưởng và hiệu chỉnh trên train ở bước sau. Không thể kết luận nguyên nhân chỉ từ điểm tổng hợp, cũng chưa có căn cứ đảm bảo hiệu chỉnh sẽ khôi phục điểm cũ.

Chi tiết từng file nằm trong [before.json](../outputs/reports/stage1_pipeline/before.json), [after.json](../outputs/reports/stage1_pipeline/after.json); tóm tắt kiểm chứng ở [validation.json](../outputs/reports/stage1_pipeline/validation.json). Các báo cáo và dự đoán cũ được giữ để đối chiếu.

## Chưa thuộc đợt này

GMM/giao điểm Gaussian, HMM, mapping KMeans/CSV, chia cho tham chiếu bằng 0 khi đánh giá, script 03, đồng bộ defaults và hiệu chỉnh ngưỡng chưa được sửa. Các giới hạn dữ liệu và kết quả lịch sử vẫn theo [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
