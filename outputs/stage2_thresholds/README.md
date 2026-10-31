# Kết quả sửa phân ngưỡng đợt 2

- `before/reports/`: benchmark 16 ngay sau đợt 1, trước sửa GMM/K-Means/HMM/ngưỡng động.
- `after/reports/`: benchmark 16 sau sửa đợt 2; JSON ghi thêm GMM weights, log-likelihood, hội tụ, trạng thái nghiệm và giao thức fit.
- `before/figures/`, `after/figures/`: bốn hình tương ứng từng phiên bản. Hình trước giữ nguyên cả các giới hạn so sánh/nhãn đã được sửa ở bản sau.
- `validation.json`: 53 test hồi quy, kết quả so sánh, kiểm tra GMM và 24 cặp thuật toán/file trên tám WAV.

Mốc trước được chạy lại ngày 28/10/2026; khác các JSON lịch sử trong `outputs/reports/`. Không hiệu chỉnh tham số theo test. [Nhật ký đợt 2](../../docs/STAGE2_FIXES.md) mô tả các thay đổi, điểm số và giới hạn diễn giải.

Chạy lại vào thư mục mới để giữ bản đối chiếu:

```powershell
.\.venv\Scripts\python.exe -B scripts/16_evaluate_advanced_unsupervised.py --output-dir outputs/stage2_thresholds/recheck
```
