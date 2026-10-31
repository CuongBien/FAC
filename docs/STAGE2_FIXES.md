# Sửa phân ngưỡng đợt 2 — 28/10/2026

Tiếp tục trên nhánh `codex/fix-stage1-pipeline`, chưa merge vào `main`. Phạm vi là bốn mục 8–11 của kế hoạch; giữ các kết quả cuối đợt 1 làm mốc đối chiếu. Không hiệu chỉnh tham số theo điểm test.

## 1. GMM: giao điểm, prior và hội tụ

Giao điểm giải phương trình `w0 * N(x; mu0, std0) = w1 * N(x; mu1, std1)`. Với hệ số bậc hai đang dùng, hạng logarit đúng là `log(std1/std0) + log(w0/w1)`; công thức cũ đảo dấu phần độ lệch chuẩn và bỏ phần prior.

- `prior` là trọng số component khởi tạo có mean thấp; trọng số còn lại là `1-prior`. Đây không phải luôn là prior của UV, vì AMDF có chiều V/UV ngược ACF/Cepstrum. Khi đổi thứ tự component, trọng số đi theo component.
- Giải cả trường hợp phương sai bằng nhau và khác nhau, ưu tiên nghiệm giữa hai mean. Phương trình được dịch tâm và dùng cách tính nghiệm giảm triệt tiêu số học.
- Tính EM trong miền log; chặn phương sai tối thiểu; chỉ báo hội tụ khi cả mean/std và log-likelihood ổn định. Log-likelihood trả về ứng với tham số cuối.
- Trả `threshold_status` để phân biệt nghiệm thật với dữ liệu thiếu, hai component trùng hoặc không có giao điểm. Trường hợp không có nghiệm dùng midpoint dự phòng, không coi đó là giao điểm chính xác.

Hai Gaussian khác phương sai có thể có hai giao điểm. Pipeline hiện vẫn dùng một ngưỡng với chiều quyết định theo thuật toán; đây là giới hạn của mô hình phân ngưỡng một chiều, không phải bộ phân loại Bayes đầy đủ trên cả hai đuôi.

## 2. K-Means 2D: gán nhãn trong không gian chuẩn hóa

Clustering và điểm chọn tâm voiced cùng dùng đặc trưng chuẩn hóa: `z_periodicity - z_zcr` cho ACF/Cepstrum, `-z_periodicity - z_zcr` cho AMDF. Không cộng/trừ hai đại lượng ở thang đo gốc để quyết định nhãn nữa.

- Đổi đơn vị dương của một đặc trưng không làm đảo nhãn khi dữ liệu chuẩn hóa không đổi. Chỉ đặc trưng hằng mới dùng scale dự phòng 1.
- Tâm rỗng được khởi tạo lại tại điểm xa tâm còn lại nhất. Tâm trùng/không tạo được hai nhóm trả UV cùng trạng thái `degenerate`.
- Khi điểm âm học bằng nhau, ưu tiên periodicity mạnh hơn, sau đó ZCR thấp hơn và index ổn định. Quan sát cách đều hai tâm được gán UV.
- Benchmark không tự gán tất cả active thành voiced khi chỉ có ít khung; vẫn gọi clustering để xử lý dữ liệu ít/suy biến theo cùng quy tắc.

## 3. HMM: giữ ranh giới silence

API nhận `active_mask` trên toàn bộ timeline. Forward/backward trong EM và Viterbi đều chạy riêng từng đoạn active; mỗi đoạn bắt đầu lại từ `pi`. Các emission Gaussian vẫn học chung từ các đoạn active trong cùng file, nên đây không phải fit một mô hình riêng cho mỗi đoạn.

`A = [[0.85, 0.15], [0.15, 0.85]]` và `pi` **cố định**, không được học. Code dùng xác suất log để tránh underflow. Silence trả nhãn `sil`; dữ liệu hằng không nhận voiced chỉ vì có ít mẫu. Gọi API không có mask được hiểu là một đoạn liên tục như trước.

Benchmark và hình minh họa đều truyền toàn timeline/mask. Hình ngưỡng tĩnh và HMM dùng cùng silence gate; nhãn Gaussian AMDF cũng đã được đảo đúng chiều. Các tiêu đề mới không khẳng định khử sạch lỗi từng khung.

## 4. Ngưỡng động: đúng kích thước cửa sổ

`window_frames=50` dùng đúng 50 khung ở giữa file: `[i-25, i+25)`, tương đương offset -25 đến +24. Cửa sổ lẻ cân bằng hai bên; tại biên chỉ lấy phần có dữ liệu, không padding. Cửa sổ ngắn vẫn dùng median của dữ liệu có sẵn.

Trọng số thích nghi là `alpha` được điều biến bởi STE tương đối. Đây là thích nghi bằng median/STE, chưa có bộ ước lượng SNR. Ngưỡng nền trong benchmark lấy từ GMM đã sửa.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts/16_evaluate_advanced_unsupervised.py --output-dir outputs/stage2_thresholds/recheck
```

- **53 test đạt:** 27 test đợt 1 và 26 test mới. Gồm đối chiếu mật độ Gaussian có prior, 100 bộ tham số ngẫu nhiên, đổi đơn vị K-Means, posterior/likelihood/Viterbi HMM đối chiếu bằng vét cạn, EM reset qua silence, cửa sổ chẵn/lẻ và biên.
- **24 cặp thuật toán/file** trên 8 WAV đạt các kiểm tra: K-Means không đổi nhãn khi đổi đơn vị; HMM giữ đúng nhãn silence, đoạn active và A/pi; ngưỡng động hữu hạn, khớp số khung.
- Ba GMM fit từ train có chênh lệch log-mật độ tại ngưỡng dưới `1e-12`, đều hội tụ.
- Benchmark 16 chạy lại đủ ba thuật toán, bảy chiến lược trên bốn file test; xuất JSON/CSV và bốn hình vào thư mục riêng.

## Điểm benchmark trước và sau

Final Score (%), **thấp hơn là tốt hơn**. Mốc trước được chạy ngay trước sửa đợt 2, tức sau khi đã sửa pipeline đợt 1; không dùng JSON lịch sử làm mốc này.

| Chiến lược | ACF trước → sau | AMDF trước → sau | Cepstrum trước → sau |
|---|---:|---:|---:|
| GMM 1D | 5,09 → 5,17 | 6,30 → 6,67 | 40,55 → 33,11 |
| K-Means 2D | 7,63 → 7,63 | 8,16 → 8,16 | 31,63 → 31,63 |
| HMM | 16,55 → 16,24 | 20,53 → 17,82 | 39,14 → 30,84 |
| Ngưỡng động | 5,94 → 5,93 | 7,32 → 7,63 | 41,45 → 33,77 |

Ngưỡng GMM train đổi lần lượt: ACF `0.4110 → 0.4294`, AMDF `0.4332 → 0.4239`, Cepstrum `0.0822 → 0.1024`. Sai số không giảm đồng loạt; không dùng điểm test để chọn lại tham số.

K-Means không đổi điểm trên tám WAV hiện có; lỗi thang đo được kiểm chứng bằng dữ liệu phản ví dụ và phép đổi đơn vị. HMM có cải thiện ở benchmark này nhưng LAB chỉ có thống kê toàn file, chưa đủ kết luận chất lượng từng khung.

Giao thức được giữ và ghi trong JSON: GMM/Median/Otsu fit trên **toàn bộ khung train, có silence**; 2D/HMM thích nghi trên active của **từng file test**; ngưỡng động dùng ngữ cảnh của file đầu vào. Việc thống nhất quần thể fit, đánh giá các giao thức và hiệu chỉnh trên train thuộc bước kiểm chứng sau.

Kết quả nằm tại [outputs/stage2_thresholds](../outputs/stage2_thresholds/README.md); [validation.json](../outputs/stage2_thresholds/validation.json) ghi điểm, sai số mật độ và kiểm tra từng file. Các baseline/Champion của CLI không sử dụng các mô hình này nên không được thay thế bởi điểm ở bảng trên.

## Phần tiếp theo

Đợt 3 còn: xử lý MAPE khi tham chiếu bằng 0, chặn fit Gaussian thiếu mẫu V/UV, sửa `None` trong script 03, mapping CSV benchmark 15, thống nhất cấu hình và cách gọi chỉ số. Truy vết Cepstrum Champion, hiệu chỉnh và cập nhật toàn bộ báo cáo vẫn thuộc đợt 4.
