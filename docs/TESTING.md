# Kiểm thử và tái lập

Chạy từ thư mục gốc bằng môi trường `.venv` hoặc `uv run`.

```powershell
python -B -m unittest discover -s tests
python scripts/21_benchmark_champion_unsupervised.py train --output-dir outputs/new_run
python scripts/21_benchmark_champion_unsupervised.py evaluate --output-dir outputs/new_run
python scripts/21_benchmark_champion_unsupervised.py report --output-dir outputs/new_run
python -m src runs verify outputs/new_run
```

Không ghi đè run đã đóng băng. `--resume` kiểm chứng artefact của pha đã hoàn thành. Pha report tạo revision trong `reports/`; `per_file.csv` và `contours/` lưu số liệu chi tiết.

```powershell
python predict.py -i TinHieuKiemThu -m champions -o outputs/predictions
python scripts/27_evaluate_ssl_methods.py
python scripts/28_benchmark_champion_ssl.py
```

Ngưỡng động dùng Otsu làm nền; mọi kết quả phải được chạy trên phiên bản hiện tại. Champion giữ nguyên preset và kết quả gốc; ngưỡng động hiện dùng Otsu làm nền.
