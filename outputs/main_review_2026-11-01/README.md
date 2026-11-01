# Benchmark sau sửa main — 01/11/2026

Nguồn: [nhật ký sửa](../../docs/MAIN_REVIEW_FIXES.md), nhánh `codex/fix-main-review` từ `5a9273f`.

- `histogram/reports/15_unsupervised_thresholds_results.json` và CSV: 4 phương pháp 1D cùng đối chứng ngưỡng lịch sử.
- `advanced/reports/16_advanced_unsupervised_results.json` và CSV: GMM, 2D, HMM, ngưỡng động cùng đối chứng.
- `histogram/figures/`, `advanced/figures/`: biểu đồ từ đúng hai lần chạy mới.
- `run_manifest.json`: source/data SHA, lệnh chạy và checksum báo cáo để truy xuất nguồn.

Otsu trả cạnh bin thay vì tâm bin, ngưỡng ACF/AMDF/Cepstrum lần lượt 0,460244 / 0,399144 / 0,193046. Final Score test lần lượt 6,45% / 7,15% / 53,68%. Các giá trị đầy đủ nằm trong JSON/CSV. Bảng Stage 4 cũ vẫn giữ mô hình và điểm trước sửa.

Train gồm bốn file, ngưỡng toàn cục fit toàn bộ khung gồm silence; test gồm bốn file, frame/hop 25/10 ms, có silence gate và median theo đoạn. Benchmark 16 ghi rõ mô hình 2D/HMM nào fit trên từng input. Hai script này không tích hợp plugin Champion và không lựa chọn lại preset. Các điểm là sai lệch thống kê toàn file; chưa có Accuracy/F1 vì LAB thiếu nhãn thời gian.

```powershell
uv run python -B scripts/15_evaluate_unsupervised_thresholds.py --output-dir outputs/main_review_recheck/histogram
uv run python -B scripts/16_evaluate_advanced_unsupervised.py --output-dir outputs/main_review_recheck/advanced
```

Dùng thư mục mới khi chạy lại để giữ bằng chứng của lần này.
