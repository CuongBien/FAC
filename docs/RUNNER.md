# Runner chung — refactor đợt 3

Đợt 3 chuyển orchestration khỏi script 17/21 và nối chúng với `ExperimentSpec`, source/data manifest và `RunStore`. Thuật toán pitch, công thức fit/metric, thứ tự plugin, search grid và quy tắc chọn theo train được giữ nguyên. Vòng lặp khoa học và renderer được đối chiếu AST với snapshot đợt 2.

```text
scripts/17,21 hoặc python -m src.experiments run
                         ↓
                  experiments/runner.py
          spec + data/source/runtime + transactions
                    ↓               ↓
             calibration.py     champion.py
                    ↓               ↓
                pipeline + analysis
```

`transactions.py` quản lý khóa và công bố artifact; `protocol_specs.py` mô tả hai giao thức; `dependencies.py` khai báo scope nguồn. `champion_report.py` là shim gọi `src/reporting`; từ đợt 5, report revision không ghi vào manifest khoa học. Xem [reporting](REPORTING.md). Backend chỉ điều phối thí nghiệm, không chứa công thức ACF/AMDF/Cepstrum hoặc phép fit mới.

## Chạy một run mới

```powershell
uv run python -B scripts/21_benchmark_champion_unsupervised.py prepare --spec configs/experiments/champion_unsupervised.json --output-dir outputs/runs/champion-recheck
uv run python -B scripts/21_benchmark_champion_unsupervised.py train --output-dir outputs/runs/champion-recheck
uv run python -B scripts/21_benchmark_champion_unsupervised.py evaluate --output-dir outputs/runs/champion-recheck
uv run python -B scripts/21_benchmark_champion_unsupervised.py report --output-dir outputs/runs/champion-recheck
```

Pha `train` tự prepare khi thư mục run chưa tồn tại. Có thể dùng thư mục đầu ra rỗng; không dùng lại thư mục chứa bằng chứng cũ. Script 17 dùng cùng lifecycle và [stage4_calibration.json](../configs/experiments/stage4_calibration.json); các pha của nó là prepare/train/evaluate/report. Script 19 cũng gọi report bundle; publish tài liệu là bước riêng có preview.

CLI chung tương đương:

```powershell
uv run python -B -m src.experiments run champion train --spec configs/experiments/champion_unsupervised.json --train-dir TinHieuHuanLuyen --output-dir outputs/runs/champion-recheck
uv run python -B -m src.experiments run champion evaluate --train-dir TinHieuHuanLuyen --test-dir TinHieuKiemThu --output-dir outputs/runs/champion-recheck
uv run python -B -m src.experiments verify outputs/runs/champion-recheck
```

Hai JSON hiện là **giao thức v1 cố định có thể thực thi**, không phải bộ tuning mới. Runner kiểm tra đầy đủ pipeline, control, fit population, tham số và search grid; giá trị không được hỗ trợ sẽ báo lỗi, không âm thầm dùng default. Đổi `name`/`description` được phép và tạo spec identity riêng. Muốn thay giao thức khoa học cần triển khai revision mới có đối chứng. Đây là giới hạn chủ động để refactor không trở thành một lần đổi thuật toán.

## Trạng thái và artifact

| Pha hoàn tất | Trạng thái manifest | Artifact chính |
|---|---|---|
| Prepare | `prepared`, có `completed_stages.prepare` | Config, source snapshot và runtime |
| Train/freeze | `trained` | Frozen config/model cùng SHA sidecar; train features/trials |
| Evaluate | `evaluated` | Evaluation, CSV, contour và input-adaptive models |
| Report revision (từ đợt 5) | Giữ `evaluated` hoặc `reported` lịch sử | Gói riêng dưới `reports/<id>` hoặc destination chỉ định |

`RunStore.create()` cấp thấp vẫn tạo được manifest prepared chỉ có config như đợt 1; runner yêu cầu prepare hoàn tất trước train. Mỗi stage có input identity, danh sách artifact, checksum kết quả và thời điểm công bố. Verify kiểm tra checksum, stage order, train manifest, liên kết source/freeze/evaluation/report. Trạng thái verified nói về tính toàn vẹn của bằng chứng, không chứng minh chất lượng mô hình.

Run mới giữ đường file phẳng của định dạng cũ (`frozen_config.json`, `contours/`...) để các reader/report hiện hành tiếp tục dùng được. Metadata mới nằm trong `manifest.json`/`config.json`. Store metadata ghi canonical JSON atomically; các serializer scientific/compatibility vẫn giữ hợp đồng cũ. Không di chuyển hoặc viết lại run lịch sử. Registry nhận cả native run trước train và bỏ qua thư mục attempt nội bộ.

## Lỗi giữa chừng và resume

```powershell
uv run python -B scripts/21_benchmark_champion_unsupervised.py train --resume --output-dir outputs/runs/champion-recheck
uv run python -B scripts/21_benchmark_champion_unsupervised.py evaluate --resume --output-dir outputs/runs/champion-recheck
```

- Writer giữ khóa OS suốt pha. Writer thứ hai bị từ chối; khi tiến trình chết, OS tự nhả khóa. Không suy đoán PID đã chết hoặc tự xóa khóa của tiến trình khác. File `.run.lock` có thể còn trên đĩa sau khi khóa đã được nhả.
- Tính toán ghi vào `.attempts/<id>/`. Trước khi công bố, runner xác minh source/data/runtime và mọi input đã commit. Backend làm thay đổi input hoặc thiếu output bắt buộc sẽ không được gắn trạng thái hoàn tất.
- Journal `.transaction.json` ghi trước khi chuyển artifact ra đường chính. Nếu bị ngắt lúc công bố, `--resume` kiểm tra SHA và hoàn tất journal, không fit/evaluate lại. Xung đột hoặc artifact bị hỏng sẽ bị từ chối.
- Nếu lỗi xảy ra trước journal, chưa có checkpoint khoa học được công bố: lần tiếp tục bắt đầu lại **pha chưa hoàn tất** trong attempt mới. Không resume theo từng frame/trial. Attempt lỗi giữ lại để kiểm tra; registry không xem nó là một run.
- Pha đã hoàn tất chỉ được xác minh khi dùng `--resume`; không đổi manifest, không refit. Report revision được xác minh/tái dùng theo ID riêng. Không dùng resume để đổi config, dataset, code hoặc runtime.

Việc công bố nhiều file dựa vào journal và manifest được thay atomically cuối cùng; không giả định toàn thư mục file được đổi atomically. Trong khoảng bị ngắt, một số file mới có thể đã xuất hiện nhưng chưa thuộc stage hoàn tất.

## Tái lập và source scope

Prepare/train chỉ nhận train directory. Test được ghim ở input của pha evaluate sau freeze; test thay đổi thì không thể resume evaluation cũ. Identity không chứa đường dẫn data tuyệt đối; có thể chuyển run và data sang thư mục khác rồi xác minh/resume.

Nguồn dùng canonical UTF-8/LF cho scope và giữ raw SHA/snapshot để truy xuất. Lockfile cũng có raw/canonical SHA; thay xuống dòng được chấp nhận, thay nội dung bị chặn. Python/numpy/scipy/matplotlib và lock identity được kiểm tra. Git là metadata tùy chọn. Source root cung cấp phải mô tả đúng mã đang thực thi, tránh ghi provenance của một bản copy khác.

Scope train/predict/evaluate còn bảo thủ, gồm các module runtime, entrypoint và dependency lock. Renderer Champion nằm riêng, nên sửa riêng `champion_report.py` không vô hiệu train/predict. Scope report còn giữ README/config mẫu để snapshot có đủ file hỗ trợ tái lập; config thực thi vẫn được ghim độc lập trong từng run. Việc tách dependency metric/format và report revision đầy đủ thuộc đợt 4–5.

Report hiện vẫn kiểm tra nguồn và dataset của run; API render hoàn toàn độc lập dữ liệu đầu vào, nhiều report revision và publisher riêng sẽ được làm ở đợt 5. Wheel/CI trên Windows/Linux là đợt 6; kiểm tra AST Python 3.9 chưa thay cho chạy runtime trên mọi nền tảng.

## Bằng chứng

[outputs/refactor_phase3](../outputs/refactor_phase3/README.md) lưu test, source snapshot và native run nén. Có 170 test, gồm 18 test mới cho lifecycle/journal/khóa/relocation/source policy. Kiểm chứng thực tế tạo lại ba model từ train, chạy native lifecycle, đối chiếu 288 kết quả và các bảng report với run Champion cũ; Stage 4 replay model cũ mà không quét lại 222 cấu hình.

```powershell
uv run python -B -m unittest discover -s tests
uv run python -B scripts/23_verify_runner_refactor.py --output outputs/refactor_phase3-recheck
```

Script 23 từ chối ghi đè proof cũ; run mới chạy trong thư mục tạm và được lưu thành bundle nén cùng manifest/SHA. Các artifact gốc và preset không được sửa. Test đã có lịch sử quan sát; run mới không chọn lại preset theo điểm test.

Bundle đã được giải nén vào thư mục ngoài Git và chạy CLI từ chính source snapshot: resume cả train/evaluate/report thành công, 336 artifact được xác minh, manifest giữ nguyên. Kết quả bổ sung ở [portable_snapshot.json](../outputs/refactor_phase3/portable_snapshot.json).

Từ đợt 5, `RunService.report()` chỉ cần run đã evaluate; không đọc dataset/source khoa học hiện hành hoặc yêu cầu đúng runtime từng train. Gói report có renderer snapshot/runtime riêng. Run `reported` đợt 3 và journal report cũ vẫn đọc được; lần dựng mới luôn vào revision riêng.
