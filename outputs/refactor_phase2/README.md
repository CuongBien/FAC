# Kiểm chứng refactor pipeline — đợt 2

Nhánh `codex/refactor-pipeline`, kế thừa đợt 1 tại `14d58fb`. Đợt này tổ chức lại code, không hiệu chỉnh preset hoặc fit/chọn lại mô hình train theo test.

| Bằng chứng | Nội dung |
|---|---|
| [reference.json](reference.json) | 78 nhóm dữ liệu chụp trước khi sửa code: histogram train/test, raw features, baseline/Champion cho ba thuật toán và tám WAV; SHA từng mảng số/nhãn cùng manifest dataset |
| [validation.json](validation.json) | So sánh chính xác reference, 288 trường hợp, 93.423 dòng contour, 3.873 dòng train feature; runtime, AST Python 3.9, source/test SHA và xác minh hai run cũ |
| [source_snapshot.tar.gz](source_snapshot.tar.gz) | Byte nguồn thực tế của phiên bản đối chiếu; SHA snapshot và từng file được ghi trong validation |
| [tests.json](tests.json) | 152 test đạt; hash các file test của lần kiểm tra |

Mốc reference được tạo trên code đợt 1 trước khi thay detector, benchmark hoặc extractor. Không cập nhật expected sau khi refactor. Dictionary mới có thêm metadata/contract; đối chiếu reference bắt buộc mọi trường số, mảng và nhãn đã có trước đó khớp, không áp đặt các trường mới lên phiên bản cũ.

Source snapshot ghi base commit và nhánh tại lúc kiểm tra, cùng byte nguồn chưa commit để bao gồm chính thay đổi đợt 2. Đây là provenance của lần kiểm chứng, không phải snapshot runtime của run Champion lịch sử. Không sửa source/model/evaluation của run cũ để khớp HEAD mới.

Tái lập từ checkout với cùng môi trường:

```powershell
uv run python -B -m unittest discover -s tests
uv run python -B scripts/22_verify_pipeline_refactor.py verify
```

Lệnh `verify` không ghi file. Thêm `--output <thu-muc-moi>` để lưu proof bundle mới; từ chối ghi đè proof có sẵn. Chế độ `capture` dùng để thiết lập mốc ở checkout trước thay đổi, không phải cách sửa một phép đối chiếu thất bại.

288 trường hợp dùng frozen train models cũ, không fit lại các model đó. Hai control input-adaptive vẫn fit từng input theo giao thức riêng. So metric, nhãn ban đầu/cuối, candidate, timeline và F0 bằng đẳng thức chính xác trong cùng runtime. Đây không phải đánh giá chất lượng trên dataset mới hoặc bằng chứng tỷ lệ V/UV bằng 50/50.

Xem [cấu trúc pipeline](../../docs/PIPELINE.md) và [kế hoạch các đợt còn lại](../../docs/REFACTOR_PLAN.md).
