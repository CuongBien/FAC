# Kiểm chứng runner — refactor đợt 3

Nhánh `codex/refactor-runner`, kế thừa `codex/refactor-pipeline` tại `fe3f855`. Đợt này tách orchestration và triển khai lifecycle, không đổi preset/search rule/công thức khoa học.

| Bằng chứng | Nội dung |
|---|---|
| [tests.json](tests.json) | 170 test, gồm 18 test mới về crash/journal/OS lock, source/runtime policy và chuyển thư mục |
| [validation.json](validation.json) | Model và số liệu đối chiếu, AST, runtime, source/test SHA, resume không ghi lại và checksum bundle |
| [source_snapshot.tar.gz](source_snapshot.tar.gz) | Byte nguồn thực tế của phiên bản kiểm chứng; Git metadata ghi base commit trước commit refactor |
| [champion_run.tar.gz](champion_run.tar.gz) | Native run mới đầy đủ manifest/config/freeze/evaluation/CSV/contour/report và source/runtime snapshot; lưu nén để tránh nhân đôi hàng trăm CSV trong repo |
| [portable_snapshot.json](portable_snapshot.json) | Giải nén an toàn vào thư mục ngoài Git, import từ chính source snapshot và chạy CLI resume train/evaluate/report; xác minh 336 artifact và manifest không đổi |

Run mới fit lại ba model từ bốn WAV train, sau đó chạy evaluate/report và xác minh resume không refit hoặc đổi manifest. Tất cả giá trị model đã có trong freeze Champion cũ, 288 trường hợp, 93.423 dòng contour, 3.873 dòng đặc trưng train và các bảng report phải khớp chính xác. Hai control input-adaptive vẫn fit từng input theo giao thức riêng. Không dùng test để chọn lại Champion.

Stage 4 replay model cũ, đối chiếu 408 trường hợp protocol và 72 trường hợp pipeline. Không quét lại 222 cấu hình. AST vòng lặp khoa học, helper được di chuyển và renderer được đối chiếu với snapshot đợt 2. Các run/preset/dataset gốc giữ nguyên.

Tái lập trên cùng môi trường, dùng output mới:

```powershell
uv run python -B -m unittest discover -s tests
uv run python -B scripts/23_verify_runner_refactor.py --output outputs/refactor_phase3-recheck
```

Bundle native chứa thư mục `champion-proof/`; giải nén vào thư mục mới để xem hoặc dùng `RunService` xác minh/resume với train/test đúng manifest. Không giải nén đè lên run đã có. Registry hiện quét run dạng thư mục, chưa quét trực tiếp tar; checksum và run ID của bundle nằm trong validation.

Proof source snapshot dùng source manifest rộng cho đối chứng refactor. Native run bên trong bundle dùng source scope riêng có canonical hash, Git metadata tùy chọn, raw/canonical lock SHA và schema v1. Cả hai giữ byte nguồn đúng thời điểm, không viết lại provenance lịch sử cho khớp HEAD mới.

Xem [runner/lifecycle/resume](../../docs/RUNNER.md) và [kế hoạch còn lại](../../docs/REFACTOR_PLAN.md).
