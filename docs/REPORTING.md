# Báo cáo có thể dựng lại — refactor đợt 5

`src/reporting` sở hữu view model, renderer, hình, report bundle, publisher và danh mục. Báo cáo chỉ dùng dữ liệu đã lưu và xác minh; không đọc WAV/LAB, chạy detector, fit mô hình hoặc chọn lại preset. Hai giao thức được hỗ trợ là Stage 4 và Champion v1.

## Dựng và xác minh

```powershell
uv run python -B -m src.reporting render outputs/champion_unsupervised_2026-11-01 --output outputs/reports/champion-review
uv run python -B -m src.reporting verify outputs/reports/champion-review
uv run python -B -m src.reporting render outputs/stage4_validation --output outputs/reports/stage4-review
```

Lệnh tương thích:

```powershell
uv run python -B scripts/21_benchmark_champion_unsupervised.py report --output-dir outputs/champion_unsupervised_2026-11-01 --report-dir outputs/reports/champion-review
uv run python -B scripts/19_report_validation.py --output-dir outputs/stage4_validation --report-dir outputs/reports/stage4-review
```

`--output-dir` trong script cũ vẫn chỉ run nguồn. **Script 19 giờ chỉ dựng báo cáo**, không sửa README hoặc năm chương. Không có `--report-dir`, wrapper tạo `RUN/reports/<report_id>`. Muốn giữ toàn bộ thư mục lịch sử nguyên byte, dùng destination riêng như các lệnh trên.

Run native mới giữ trạng thái `evaluated` khi tạo báo cáo; báo cáo là revision dẫn xuất, không thêm artifact vào manifest khoa học. Run `reported` của đợt 3 vẫn xác minh được bằng reader cũ. `RunService.report()` không cần thư mục train/test, source khoa học hiện hành hoặc đúng môi trường từng fit; nó kiểm tra bằng chứng đã ghim của run.

## Gói bất biến

```text
<report>/
  report_manifest.json       fac.report.v1: ID, source, renderer/runtime, artifact SHA
  view.json                  Bản trình bày bất biến: freeze/models/evaluation/extras
  inputs/                    Bản sao các đầu vào thực sự dùng và các link bằng chứng
  renderer_source.tar.gz     Mã renderer/reader đã dùng, có SHA từng file
  README.md                  Bảng, giải thích, nguồn và hình
  tables.csv                 Điểm aggregate đã lưu, kèm run/evaluation/version
  sections.json              Nội dung cho các vùng marker được quản lý
  figures/                   PNG có run/evaluation/renderer và manifest liên kết
  summary.json               Chỉ Stage 4; giữ schema và số liệu cũ
```

Native source được kiểm tra bằng run manifest, gồm sidecar metric nếu đã có. Legacy source dùng reader riêng xác minh freeze/evaluation/source snapshot/runtime và các export đã được báo cáo cũ ghim SHA. Nếu checkout đổi LF/CRLF, chỉ phục hồi byte trong bộ nhớ khi khớp SHA lịch sử. Không sửa checksum hoặc provenance tại nguồn.

Legacy chưa có sidecar được ghi rõ `context_kind: recorded-legacy`, `metrics_sha256: null`; renderer dùng aggregate đã lưu, không tự suy diễn context hoặc gắn kết quả thích nghi vào nhóm frozen. Trace Cepstrum chỉ xuất hiện khi run thật sự có bằng chứng replay.

ID dựa vào view, mã renderer chuẩn hóa EOL và môi trường dựng hình; SHA raw của mã vẫn được lưu. Đổi renderer sinh revision mới. Cùng input/destination đã có thì xác minh và tái dùng; không ghi lại hoặc thay expected. Muốn thay input phải dùng destination mới. Công bố gói bằng đổi tên thư mục tạm sau khi hoàn tất xác minh; lỗi render không để lại gói hoàn chỉnh giả. Attempt ẩn lỗi được giữ để kiểm tra.

`verify` kiểm tra đủ tập artifact, checksum, view/input đồng nhất, source link, cả hai snapshot và link Markdown nội bộ. Gói gọn chứa dữ liệu đủ dựng bảng/hình; contour, train features và input-adaptive model đầy đủ vẫn thuộc run gốc. Đây không phải bản sao đầy đủ để train lại.

## Dựng lại khi không còn run gốc

```powershell
uv run python -B -m src.reporting rebuild outputs/reports/champion-review --output outputs/reports/champion-rebuilt
```

Rebuild chỉ đọc gói đã xác minh, chạy renderer hiện tại và tạo revision riêng. Trong cùng môi trường/mã, README, view, CSV và sections phải khớp byte. PNG có thể khác giữa font/backend; không dùng SHA PNG liên máy để chứng minh kết quả khoa học. Mã nguồn renderer và runtime cũ được giữ trong gói để chọn đúng môi trường khi cần tái lập.

## Xuất bản vào tài liệu

```powershell
# Chỉ in diff, chưa ghi tài liệu
uv run python -B -m src.reporting publish outputs/reports/stage4-review --root .
# Bước ghi tường minh
uv run python -B -m src.reporting publish outputs/reports/stage4-review --root . --apply
```

Publisher kiểm tra **tất cả** target/marker trước khi ghi, đổi link theo vị trí từng tài liệu và giữ nguyên byte ngoài marker, kể cả BOM/CRLF. Không chèn marker mới hoặc ghi đè toàn file. Marker thiếu, trùng, đảo thứ tự hoặc preview đã cũ đều bị từ chối; lỗi I/O được rollback. OS lock ngăn hai publisher ghi đồng thời.

Stage 4 dùng các marker `STAGE4_README`, `STAGE4_REPORT`, `STAGE4_CURRENT` hiện có. Champion yêu cầu một cặp `<!-- CHAMPION_REPORT_START -->` / `<!-- CHAMPION_REPORT_END -->` trong `docs/CHAMPION_UNSUPERVISED_RESULTS.md`. Tài liệu Champion hiện tại chưa có cặp này; renderer vẫn dùng được, publisher sẽ từ chối cho đến khi người quản lý đặt vùng sở hữu rõ ràng. Đợt refactor này không tự thêm marker hoặc xuất bản lại các chương.

Mỗi file được thay atomically; **nhiều tài liệu không phải một transaction chịu được process crash**. Nếu tiến trình bị dừng giữa các file, dựng preview và apply lại để hoàn tất; gói báo cáo bất biến vẫn là nguồn chuẩn. Lỗi marker luôn được phát hiện trước mọi thay đổi tài liệu. Không dùng publisher để sửa run hoặc artifact.

## Danh mục

```powershell
uv run python -B -m src.reporting index --root . --output outputs/REPORT_INDEX.md
```

Danh mục dựng từ run registry và report manifest đã xác minh; liên kết từng report với run/evaluation, không chọn nguồn theo tên “mới nhất” hoặc điểm test. Từ chối đặt index vào run/gói báo cáo. Input freeze được sao vào report và attempt ẩn không bị nhận nhầm là run mới.

## Cấu trúc mã và kiểm chứng

| Module | Trách nhiệm |
|---|---|
| `source.py` | Đọc bằng chứng native/legacy đã xác minh |
| `view.py` | Dữ liệu trình bày bất biến, từ chối series trùng |
| `bundle.py` | Identity, snapshot, tạo/rebuild/verify gói |
| `champion.py`, `stage4.py` | Bảng và hình từ view |
| `plots.py` | Sáu API hình speech đã chuyển nguyên triển khai; shim ở `visualization/plotter.py` |
| `common.py` | Nguồn và chú thích hình |
| `publish.py` | Preview và thay marker |
| `catalog.py`, `__main__.py` | Danh mục và CLI reporting |

Numeric metric vẫn thuộc `src/metrics`. Source reporting thuộc scope report; sửa renderer/format không vô hiệu train/predict. Reader/store chung còn nằm trong scientific scope bảo thủ. Packaging và CLI toàn dự án tiếp tục ở đợt 6.

206 test đạt, gồm 17 test reporting mới và cập nhật hồi quy wrapper/publisher. [Bằng chứng đợt 5](../outputs/refactor_phase5/README.md) đối chiếu 105 dòng bảng và summary Stage 4 với bản chụp trước refactor, dựng lại hai gói độc lập, kiểm tra marker idempotent và byte ngoài marker, bảo toàn run cũ. Sáu hàm vẽ khớp AST với snapshot đợt 4.

```powershell
uv run python -B -m unittest discover -s tests
uv run python -B scripts/25_verify_reporting_refactor.py
```

Script 25 mặc định chỉ đọc; thêm `--output <thu-muc-proof-moi>` để lưu bằng chứng mới. Không ghi đè reference hoặc proof cũ. Việc kiểm chứng bảng cần Git để đọc ranh giới tài liệu tại base commit; render/rebuild/verify report không cần Git.
