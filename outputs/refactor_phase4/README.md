# Kiểm chứng metric — refactor đợt 4

Nhánh `codex/refactor-metrics`, kế thừa `codex/refactor-runner` tại `c5324e2`. Chỉ tách hợp đồng metric/format/tổng hợp và thêm đối sánh có metadata; không đổi detector, model, preset, dữ liệu hoặc run lịch sử.

| Bằng chứng | Nội dung |
|---|---|
| [reference.json](reference.json) | Chụp trước refactor: 89 vector metric, 72 nhóm aggregate và 48 baseline/Champion, gồm framing/threshold thực tế |
| [tests.json](tests.json) | 189 test đạt, hash nguồn test và source digest |
| [validation.json](validation.json) | Kết quả đối chiếu, AST, runtime, source/test SHA và kiểm tra metadata không tương thích |
| [metrics.json](metrics.json) | Annotation mới cho 288 kết quả Champion cũ; version/profile/context và SHA gắn với freeze/evaluation gốc |
| [source_snapshot.tar.gz](source_snapshot.tar.gz) | Byte nguồn thực tế ở thời điểm đối chiếu; Git metadata chỉ base trước commit cùng các SHA nội dung |

Reference được lưu trước khi chuyển hàm hoặc thay tổng hợp; không cập nhật expected sau refactor. Phép kiểm chứng chạy lại 48 control để xác nhận cả benchmark 14, rồi tính metric từ 93.423 dòng contour cũ cho 288 trường hợp và đối sánh 72 typed series. Không fit lại model hoặc chọn lại theo test.

Annotation dùng giao thức Champion v1 cố định, kiểm tra settings với frozen config và cấu hình control trong reference trước refactor. Nó nằm riêng, không nâng cấp run cũ tại chỗ. Native run mới tự sinh sidecar cùng manifest; native run cũ thiếu context vẫn xác minh được nhưng API đối sánh không đoán framing/fit metadata.

```powershell
uv run python -B -m unittest discover -s tests
uv run python -B scripts/24_verify_metric_refactor.py verify
```

Verify mặc định chỉ đọc. Thêm `--output <thu-muc-moi>` để lưu proof mới, giữ nguyên reference đã chụp trước refactor; không cho ghi đè proof cũ. Không chạy `capture` để thay expected khi verify thất bại. So sánh cùng môi trường dùng đẳng thức chính xác, không nới tolerance. Source snapshot/validation các đợt cũ được giữ nguyên.

Xem [hợp đồng và profile](../../docs/METRICS.md), [kế hoạch còn lại](../../docs/REFACTOR_PLAN.md).
