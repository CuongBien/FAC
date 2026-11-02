# CLI và gói cài đặt — refactor đợt 6

Lệnh chung là `fac`; `python -B -m src` tương đương. Package `src` và các wrapper cũ được giữ để tương thích. Thuật toán, preset, định dạng metric và quần thể fit không thay đổi.

## Từ checkout

```powershell
uv sync --locked
uv run --locked fac --help
uv run --locked fac predict -i TinHieuKiemThu -m champions -o outputs/predictions/new-check
```

| Lệnh | Chức năng |
|---|---|
| `fac predict` | Dự đoán bằng baseline/Champion hiện hành; CSV, hình và đánh giá LAB nếu có |
| `fac experiment <champion hoặc stage4> <prepare hoặc train hoặc evaluate hoặc report>` | Lifecycle thí nghiệm v1 |
| `fac config <champion hoặc stage4>` | In spec mẫu đầy đủ; `--output` tạo file mới |
| `fac runs --root . list` | Danh mục run; có `verify`, `index` |
| `fac report <render hoặc rebuild hoặc verify hoặc publish hoặc index>` | Gói báo cáo và xuất bản marker |
| `fac compare RUN_A SERIES_A RUN_B SERIES_B` | Đối sánh metric đã xác minh, kiểm tra giao thức |

`fac --help` và `--version` không nạp numpy/scipy/matplotlib. Mỗi lệnh con có `--help` riêng. Quy tắc argument và mã thoát được chuyển tiếp tới CLI sở hữu. Tên tám phương pháp không được thêm thành preset dự đoán mặc định; chúng vẫn là giao thức benchmark với model/spec được chốt riêng.

## Thí nghiệm mới

```powershell
uv run --locked fac config champion --output champion-local.json
uv run --locked fac experiment champion train --spec champion-local.json --train-dir TinHieuHuanLuyen --output-dir outputs/runs/champion-local
uv run --locked fac experiment champion evaluate --train-dir TinHieuHuanLuyen --test-dir TinHieuKiemThu --output-dir outputs/runs/champion-local
uv run --locked fac experiment champion report --output-dir outputs/runs/champion-local --report-dir outputs/reports/champion-local
uv run --locked fac compare outputs/runs/champion-local test/acf/gmm outputs/runs/champion-local test/acf/hmm
uv run --locked fac runs --root . verify outputs/runs/champion-local
```

Train tự prepare, chỉ đọc train. Evaluate kiểm tra đúng freeze, dữ liệu và runtime. Dùng `--resume` để xác minh pha đã xong hoặc phục hồi pha chưa công bố. Report chỉ cần run đã evaluate; không cần train/test còn ở đường cũ. Chọn thư mục mới, giữ nguyên run cũ.

Các lệnh cũ vẫn chạy: `python predict.py`, script 17/19/21, `python -m src.experiments`, `python -m src.reporting`, `python -m src.metrics`. Script 19 chỉ render gói; publish tài liệu là bước riêng. API dự đoán ở `src/cli/predict.py`; `predict.py` re-export bốn hàm cũ.

## Wheel và source distribution

```powershell
uv build
```

`dist/` có wheel `fac_speech_f0-0.1.0-py3-none-any.whl` và source archive `.tar.gz`. Wheel chứa mã package, entrypoint và bảy tài nguyên hỗ trợ; không chứa WAV/LAB, outputs, archive hoặc test. Source archive có thêm script, hướng dẫn và cấu hình để dùng đường gọi cũ; dữ liệu/kết quả vẫn được cung cấp riêng.

`build_support.py` tạo `src/_runtime/manifest.json` và blob UTF-8/LF từ README, pyproject, lock, wrapper 17/21 và hai JSON spec thực sự đang build. Không giữ bản sao thủ công trong mã nguồn. `setup.py` chỉ nối build hook với setuptools; metadata nằm trong pyproject.

Reader dùng tên logic như `uv.lock`, `scripts/21_benchmark_champion_unsupervised.py` trong cả checkout, snapshot đã giải nén và wheel. Với wheel, index chọn blob và kiểm tra SHA; không lấy nhầm README/lock của package khác trong `site-packages`. Snapshot train vẫn chứa đúng byte theo tên logic đã ghi, không phụ thuộc thư mục máy cũ hoặc Git. Runtime record đọc lock đã đóng gói; thay đổi code/data/runtime vẫn bị kiểm tra như trước.

## Cài và chạy ở thư mục khác

Trong thư mục mới, tạo venv riêng. Với bộ bằng chứng đợt 6, cài dependency đúng lock rồi cài wheel, tránh tự chọn phiên bản thư viện khác:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --require-hashes -r <proof>/locked-dependencies.txt
uv pip install --python .venv/Scripts/python.exe --no-deps <proof>/fac_speech_f0-0.1.0-py3-none-any.whl
.\.venv\Scripts\fac.exe config champion --output champion.json
.\.venv\Scripts\fac.exe predict -i <duong-dan-WAV> -o predictions
```

Trên Linux dùng `.venv/bin/python` và `.venv/bin/fac`. Input/output của lệnh được giải theo thư mục hiện hành, không theo vị trí package. Chỉ CLI compatibility trong checkout mới có default trỏ tới các thư mục dữ liệu môn học; thí nghiệm qua `fac` yêu cầu đường train/test tường minh.

Không cần `.git`, editable install hoặc `PYTHONPATH` trỏ về repo. `fac report rebuild` chỉ cần report bundle; train lại cần bộ WAV/LAB đúng manifest. Wheel gọn không thay thế gói dữ liệu thí nghiệm.

## Kiểm chứng và CI

```powershell
uv run --locked python -B -m unittest discover -s tests
uv run --locked python -B scripts/26_verify_cli_packaging.py --offline
```

Script 26 mặc định kiểm chứng trong thư mục tạm, không ghi outputs. Nó build wheel từ sdist, xuất dependency từ lock với hash, tạo venv không có package kế thừa, cài wheel không editable rồi chạy các lệnh ngoài checkout với Git không có trên PATH. Bao gồm 12 trường hợp predict/contour ở 16 kHz và 44,1 kHz, BOM/scientific notation/LAB thiếu std; train không có test, 72 series evaluate, resume, report/rebuild, registry và đối sánh từ chối sai giao thức. Trong cùng runtime, kết quả predict phải khớp reference chụp trước refactor.

Thêm `--output <thu-muc-proof-moi>` để lưu wheel/sdist/snapshot và kết quả; không ghi đè proof. `--python 3.11 --skip-reference` thử runtime khác bằng kiểm tra hành vi và hợp đồng, không áp đẳng thức float giữa hai môi trường. `--offline` chỉ dùng cache; bỏ cờ nếu dependency của runtime đó chưa được tải.

CI trong `.github/workflows/ci.yml` cấu hình Windows/Linux và Python 3.9/3.10/3.11/3.12/3.13: `uv sync --locked`, unittest và wheel smoke độc lập. Workflow chỉ chạy khi branch được push/PR; file cấu hình không có nghĩa mọi ô matrix đã được kiểm chứng tại máy này. Kết quả local thực tế và runtime được ghi trong [proof đợt 6](../outputs/refactor_phase6/README.md).

Build hook theo [giao diện mở rộng setuptools](https://setuptools.pypa.io/en/latest/userguide/extension.html); CI dùng [hướng dẫn chính thức của uv](https://docs.astral.sh/uv/guides/integration/github/). Build backend được khai báo riêng trong pyproject; uv.lock khóa dependency chạy. Proof giữ SHA của wheel/sdist, mã nguồn, build input, lock và môi trường đã kiểm tra; không khẳng định các lần build có cùng timestamp/archive SHA.
