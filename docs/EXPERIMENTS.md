# Cấu trúc quản lý thí nghiệm — refactor đợt 1

Nhánh `codex/refactor-experiments`. Đợt này tách hạ tầng khỏi logic phân tích; chưa đổi pipeline, công thức metric hoặc preset. Các bước tiếp theo nằm trong [kế hoạch refactor](REFACTOR_PLAN.md).

## Phân chia trách nhiệm

```text
configs/experiments/             Spec JSON có phiên bản, không phụ thuộc default ẩn
src/experiments/
  spec.py                       ExperimentSpec bất biến, kiểm tra schema v1
  manifest.py                   Hash raw/canonical, source scope, dataset identity
  snapshots.py                  Snapshot byte nguồn chính xác và kiểm chứng archive
  serialization.py              JSON canonical mới và serializer tương thích cũ
  store.py                      Run mới, ghi atomic, khóa writer, artifact không ghi đè
  legacy.py                     Đọc/kiểm chứng format Stage 4 và Champion đã lưu
  registry.py                   Khám phá run, danh mục tương đối, không chạy mô hình
  paths.py                      Ràng buộc đường dẫn artifact nằm trong root
  __main__.py                   CLI list / verify / index
src/analysis/validation.py       Invariant, aggregate, train selection, cache cũ
outputs/registry.json            Danh mục có thể dựng lại
```

Các hàm I/O, freeze và snapshot đã rời `analysis/validation.py`. Module cũ vẫn re-export API để các import bên ngoài không bị gãy; các script 05 và 17–21 đã chuyển sang import từ module sở hữu. Cache và vòng xử lý pitch chưa được viết lại trong đợt này.

## Tra cứu và kiểm chứng kết quả

Chạy từ thư mục gốc FAC:

```powershell
uv run python -B -m src.experiments list
uv run python -B -m src.experiments list --json
uv run python -B -m src.experiments verify
uv run python -B -m src.experiments verify outputs/stage4_validation
uv run python -B -m src.experiments index
```

`list` và `verify` chỉ đọc. `verify` trả mã 1 nếu có run không hợp lệ; run hỏng vẫn được liệt kê cùng lỗi, không bị bỏ qua. `index` chỉ ghi danh mục dẫn xuất `outputs/registry.json`, không được ghi vào bên trong run được quản lý. Có thể dùng run ID từ danh mục thay đường dẫn; nếu bản sao gây trùng ID, chỉ định đường dẫn chính xác.

Với kho kết quả ở thư mục khác, dùng root tường minh:

```powershell
uv run python -B -m src.experiments --root "D:\FAC-copy" list --json
```

Registry nhận diện hai format đã có bằng cấu trúc freeze, không bằng tên thư mục: Stage 4 có 480 kết quả/file và benchmark Champion có 288. Nó kiểm tra freeze, liên kết evaluation, snapshot/runtime và các export được report manifest ghim SHA. Không import `src.analysis`, không fit hoặc gọi detector để dựng danh mục.

Trạng thái `verified` chỉ nói các liên kết/checksum của bộ bằng chứng hợp lệ. Nó không khẳng định mô hình tốt, dữ liệu mới chưa từng được xem, hoặc run cũ khớp source hiện tại. `phase` thể hiện riêng trạng thái thực thi; run mới của đợt 1 mới ở pha `prepared`.

## Quy tắc nhận diện và hash

- Nguồn mới dùng `utf8-lf-v1`: ghi cả SHA byte gốc và SHA sau chuẩn hóa LF/CRLF/CR. Không bỏ BOM, khoảng trắng, comment hoặc thay đổi code. Scope train/predict/evaluate/report phải được khai báo tường minh; sửa renderer không tự làm hỏng scope train độc lập.
- Dataset dùng `raw-sha256-v1`: tên tương đối, kích thước và SHA byte; đổi root không đổi identity, sửa WAV/LAB làm identity đổi.
- `ExperimentSpec` giữ bản JSON riêng bất biến; lấy `to_dict()` trả bản sao. Pipeline phải ghi rõ frame/hop/range/gate/median/reference threshold/pre/post plugin; fit phải ghi population, frozen/input-adaptive và strategy ID. Hiện chỉ hỗ trợ hợp đồng metric `legacy-v1`.
- Experiment ID của store dựa vào spec, dataset và scope train/predict. Report revision không đổi experiment ID; mỗi lần thực thi vẫn có run ID riêng.
- Git là metadata bổ sung của `capture_source()`, không phải điều kiện để đọc run hoặc tính fingerprint. Khi không có Git, trường này là null. Snapshot vẫn giữ byte chính xác để truy xuất nguồn.
- `.gitattributes` chỉ áp dụng cho đường mới: package experiments/config giữ LF, artifact store mới giữ byte do writer ghi. Không renormalize hoặc đổi SHA của outputs lịch sử.

Adapter legacy chỉ thử khôi phục LF/CRLF **trong bộ nhớ** khi byte khôi phục khớp chính xác SHA cũ. Nó không ghi lại file, không sửa sidecar và không chấp nhận thay đổi giá trị. API cũ `read_frozen()` vẫn kiểm tra SHA byte nghiêm ngặt như trước; adapter riêng giải quyết việc tra cứu archive qua checkout nhiều nền tảng.

## API cho run mới

```python
from src.experiments import ExperimentSpec, RunStore, RunRegistry
from src.experiments.manifest import capture_dataset, capture_source
```

`ExperimentSpec(document)` kiểm tra và giữ config; `RunStore.create(spec, datasets, source, run_id)` ghi `config.json`/`manifest.json` cho run mới. `add_json(run_id, path, value, role)` ghi artifact một lần và ghim SHA; `verify(run_id)` xác minh các artifact đã đăng ký. Ghi JSON dùng file tạm, fsync và replace; writer thứ hai bị chặn bằng lock. `RunStore` không chạy train/evaluate trong đợt 1.

Cấu hình hiện có: [champion_unsupervised.json](../configs/experiments/champion_unsupervised.json), [stage4_calibration.json](../configs/experiments/stage4_calibration.json). Từ đợt 3, runner 17/21 đã đọc/ghim spec và dùng lifecycle chung cho run mới. Hai giao thức v1 cố định được kiểm tra đầy đủ; cấu hình chưa hỗ trợ bị từ chối. Scope runtime còn bảo thủ nhưng renderer Champion đã tách riêng. Xem [runner/resume và giới hạn](RUNNER.md).

Run mới không ghi đè artifact đã tồn tại. Từ đợt 3, manifest theo dõi prepare/train/evaluate (và report lịch sử đợt 3); OS lock và journal bảo vệ việc công bố, resume kiểm tra input/SHA trước khi tiếp tục. Không tự cướp lock hoặc gắn nhãn complete cho run ghi dở. RunStore cấp thấp vẫn tạo manifest prepared chỉ có config; runner yêu cầu prepare hoàn tất trước train.

## Kiểm chứng đợt này

- **137/137 test đạt**, gồm 120 test trước và 17 test mới về spec bất biến, LF/CRLF, đổi root, source scope, thiếu Git, atomic failure, writer lock, tampering, legacy adapter và CLI.
- **288 trường hợp**, **93.423 dòng contour** và **3.873 dòng đặc trưng train** khớp chính xác kết quả đã lưu; không fit/chọn lại mô hình.
- AST của aggregate, evaluate_result, selection_key, select_trial và pitch_cache không đổi. Kiểm tra cú pháp Python 3.9 trên 67 file đạt.
- Các thư mục run cũ, WAV/LAB và preset giữ nguyên. Danh mục mới và bằng chứng refactor nằm riêng, không viết lại provenance cũ.

Nguồn kiểm chứng lịch sử của đợt 1: [outputs/refactor_phase1](../outputs/refactor_phase1/README.md). Đợt 2 có [feature contract/engine/cache](PIPELINE.md) dùng chung; `pitch_cache` được re-export từ `pipeline.cache` bằng ContextVar để tương thích caller cũ. Đợt 3 có [runner chung](RUNNER.md), 170 test tại thời điểm đó và [bộ bằng chứng riêng](../outputs/refactor_phase3/README.md).

**Đợt 4 đã hoàn tất:** [metric v1/đối sánh](METRICS.md) thuộc `src/metrics`, 189 test và [bằng chứng riêng](../outputs/refactor_phase4/README.md). Run mới có sidecar metric với context/dataset/file population; run cũ không được sửa. Format thuộc scope report, không tham gia tính metric. Đợt 5 đã có [reporting](REPORTING.md), 206 test và [bằng chứng riêng](../outputs/refactor_phase5/README.md). Báo cáo là revision riêng, không thay manifest khoa học; registry bỏ qua bản sao input trong report. Bước tiếp theo là CLI/đóng gói ở đợt 6.
