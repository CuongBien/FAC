# Pitch-Shift dùng đúng thuật toán của từng nhánh

Mặc định mỗi nhánh dùng ACF–ACF, AMDF–AMDF hoặc Cepstrum–Cepstrum cho hai khung trước/sau resample. Không dùng model/Otsu/K-means. Bản ACF-only lịch sử chỉ chạy khi chỉ định --include-legacy-shared-acf, không đưa vào bảng mặc định.

Giữ alpha=1,25, sigma=0,12, gate STE=0,01 và ngưỡng điểm 0,5. Điểm = exp(-relative_error²/(2 sigma²)) × độ tuần hoàn riêng. AMDF dùng 1-d_min; Cepstrum dùng c_positive/(c_positive+0,1), với 0,1 là ngưỡng Champion đã chốt, không fit theo test. ACF gọi trực tiếp bản cũ nên khớp chính xác.

Quyết định Pitch-Shift dùng khung tín hiệu gốc như pipeline cũ. F0 đầu ra và các plugin tiền/hậu xử lý của từng Champion giữ nguyên. Final Score (%) thấp hơn tốt hơn, metric legacy-v1.

| Phương pháp | ACF train → test | AMDF train → test | Cepstrum train → test |
|---|---:|---:|---:|
| champion | 2.215 → 2.385 | 2.078 → 2.755 | 11.390 → 22.942 |
| pitch_shift_own_method | 2.960 → 2.745 | 2.227 → 3.028 | 16.815 → 18.730 |

Không chọn lại tham số dựa vào test; điểm mới có thể kém đối chứng. [CSV từng file](per_file.csv) · [Kết quả đầy đủ](results.json).
