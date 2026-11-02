# Kiểm chứng CLI và wheel — đợt refactor cuối

Nhánh `codex/refactor-cli-packaging`, kế thừa `28be815`. Giữ nguyên thuật toán/preset/metric, WAV/LAB, uv.lock và các kết quả lịch sử. Chỉ chuyển entrypoint, hỗ trợ nguồn trong wheel và bổ sung kiểm chứng cài đặt.

| Bằng chứng | Nội dung |
|---|---|
| [reference.json](reference.json) | 12 trường hợp CLI gốc chụp trước refactor: baseline/Champion tại 16 kHz và 44,1 kHz, metric và contour đã xuất |
| [tests.json](tests.json) | 217 unittest đạt, source/test hash |
| [validation.json](validation.json) | Wheel/sdist, build input, source snapshot, runtime và kết quả smoke Python 3.12 |
| [wheel_smoke.json](wheel_smoke.json) | 12 dự đoán khớp reference; train/evaluate/resume/report/registry/compare ngoài checkout và không có Git |
| [fac_speech_f0-0.1.0-py3-none-any.whl](fac_speech_f0-0.1.0-py3-none-any.whl) | Wheel đã cài thử non-editable trong venv sạch |
| [fac_speech_f0-0.1.0.tar.gz](fac_speech_f0-0.1.0.tar.gz) | Sdist thực sự dùng để build wheel |
| [locked-dependencies.txt](locked-dependencies.txt) | Dependency và hash xuất từ uv.lock nguyên bản; có marker cho từng runtime |
| [source_snapshot.tar.gz](source_snapshot.tar.gz) | Byte nguồn và các build input, theo SHA trong validation |
| [Python 3.11](python311/validation.json) | Lượt wheel/CLI sạch độc lập với nhánh dependency 3.11 của lock; không dùng đẳng thức float liên runtime làm tiêu chí |

Wheel có entrypoint `fac` và bảy tài nguyên hỗ trợ có SHA; không chứa dữ liệu hoặc outputs. 72 series native trên tín hiệu thử được evaluate sau khi freeze, report/rebuild giữ nguyên model/evaluation/manifest. Kiểm tra thêm editable install bằng `uv sync --locked` vào venv tạm, không đổi môi trường của người dùng.

Source snapshot giữ Git base trước commit cùng SHA của nội dung thực tế. Các runtime được kiểm tra local là **Windows Python 3.12.14 và 3.11.16**. CI Windows/Linux × Python 3.9–3.13 đã được cấu hình; chưa push/chạy remote nên không tuyên bố matrix đó đã đạt.

```powershell
uv run --locked python -B -m unittest discover -s tests
uv run --locked python -B scripts/26_verify_cli_packaging.py --offline
uv run --locked python -B scripts/26_verify_cli_packaging.py --python 3.11 --skip-reference --offline
```

Mặc định script chỉ ghi thư mục tạm; thêm `--output <thu-muc-proof-moi>` để lưu bằng chứng, không ghi đè proof cũ. Bỏ `--offline` nếu cache thiếu dependency. Không fit hoặc viết lại run lịch sử trong phép kiểm chứng này; mô hình smoke mới chỉ fit tín hiệu thử mới.

Xem [CLI/cài đặt](../../docs/CLI_AND_PACKAGING.md), [kế hoạch sáu đợt đã hoàn tất](../../docs/REFACTOR_PLAN.md).
