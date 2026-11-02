# Thuật toán

## Ước lượng F0

- ACF: tìm cực đại tự tương quan chuẩn hóa trong miền độ trễ ứng với 70–400 Hz.
- AMDF: tìm cực tiểu sai khác tuyệt đối chuẩn hóa; ưu tiên đáy sớm đủ sâu để giảm lỗi chia đôi F0.
- Real Cepstrum: Hamming → FFT → log biên độ → IFFT → tìm đỉnh quefrency.
- Nội suy parabol quanh cực trị để cải thiện độ phân giải độ trễ.

## Pipeline Champion

| Thuật toán | Plugin | Ngưỡng | Frame/hop |
|---|---|---:|---:|
| ACF | Bandpass, Energy extension, Viterbi | 0,45 | 30/10 ms |
| AMDF | Energy extension, Viterbi | 0,50 | 30/10 ms |
| Cepstrum | Bandpass, Center clipping, Energy extension, Viterbi | 0,10 | 30/10 ms |

Bandpass dùng Butterworth 70–900 Hz và filtfilt. Center clipping dùng tỷ lệ 0,40. Energy extension mở rộng biên hữu thanh theo năng lượng, độ tuần hoàn và liên tục F0. Viterbi chọn đường F0 qua nhiều ứng viên và phạt bước nhảy octave. Giữ nguyên các plugin và tham số của Champion khi thay quyết định bằng unsupervised hoặc SSL. Ngưỡng guard của Energy vẫn là ngưỡng Champion.

## Quyết định không giám sát

1. Otsu: cực đại hóa phương sai giữa hai nhóm histogram.
2. K-means 1D, K=2: fit hai tâm trên độ tuần hoàn, lấy trung điểm làm ngưỡng.
3. K-means 2D, K=2: chuẩn hóa [độ tuần hoàn, ZCR], phân cụm và gán nghĩa V/UV theo tính chất âm học. Có mô hình train đóng băng và đối chứng fit từng input.
4. Ngưỡng động: T(t) = (1-a(t)) T_Otsu + a(t) median_local; trọng số phụ thuộc STE tương đối. Cửa sổ 50 khung, alpha tối đa 0,30, tách theo đoạn active. Cửa sổ có nhìn tương lai nên xử lý offline.

ACF/Cepstrum càng cao càng hữu thanh; AMDF càng thấp càng hữu thanh. Im lặng được gate bằng STE; nhãn hữu thanh chỉ chấp nhận F0 hợp lệ. Lọc median theo từng đoạn hữu thanh là thao tác làm mượt đường F0, khác với việc chọn ngưỡng phân loại.

## Pitch-Shift không model

Mỗi nhánh dùng chính bộ ước lượng của nó trên khung gốc và khung resample: ACF–ACF, AMDF–AMDF, Cepstrum–Cepstrum. Đo sai số tương đối giữa F0 sau biến đổi và alpha × F0 gốc. Điểm nhất quán Gaussian với sigma 0,12 nhân độ tuần hoàn riêng, quyết định tại 0,5; alpha mặc định 1,25.

Độ mạnh ACF là đỉnh chuẩn hóa; AMDF là 1-d_min; Cepstrum là c_positive/(c_positive+0,1), với scale lấy từ ngưỡng Champion đã chốt. Không fit scale từ test. Giữ nguyên tiền/hậu xử lý Champion. Bản ACF-only cũ vẫn giữ làm đối chứng lịch sử riêng, không mặc định đưa nhãn sang AMDF/Cepstrum.

Đây là kiểm tra tính nhất quán không cần nhãn, không huấn luyện mạng.

## Đánh giá

Fit không đọc LAB để lấy nhãn. Pha evaluate dùng thống kê LAB để tính sai số F0mean, F0std và F0num. Final Score yêu cầu đủ ba thành phần; thiếu hoặc tham chiếu bằng 0 thì thành phần phần trăm không xác định. Không có contour hoặc timestamp tham chiếu để chấm V/UV từng khung.
