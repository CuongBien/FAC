# Kế hoạch refactor để tái lập thí nghiệm và quản lý báo cáo

Ngày 01/11/2026. Trạng thái: **đã hoàn tất đợt 1–2**; đợt 2 trên `codex/refactor-pipeline`, kế thừa `codex/refactor-experiments`; các đợt 3–6 còn lại. Nhánh lập kế hoạch: `codex/refactor-reproducibility-plan`, xuất phát từ `main` tại `f059592` sau khi merge toàn bộ sửa lỗi. [Quản lý thí nghiệm](EXPERIMENTS.md), [cấu trúc và kiểm chứng pipeline](PIPELINE.md).

## Mốc phải giữ được

- 120/120 test đạt trước và sau merge vào main.
- Benchmark Champion có 288 trường hợp/file, 93.423 dòng contour; lần đối chiếu sau sửa cũng khớp 3.873 dòng đặc trưng train.
- Freeze/snapshot của Stage 4 và benchmark tám phương pháp vẫn hợp lệ. Các snapshot mang provenance của lúc chạy; không sửa chúng để khớp HEAD mới.
- Champion test: ACF 2,385%, AMDF 2,755%, Cepstrum 22,942%. Cepstrum ngưỡng động 16,480% và 2D fit từng input 10,375% thuộc giao thức riêng, không tự thay preset.
- WAV/LAB, uv.lock, HuongDan.txt và kết quả lịch sử được bảo toàn. Refactor không đổi thuật toán, thứ tự plugin, quần thể fit hoặc quy tắc chọn mô hình.

## Những điểm đang làm tái lập khó

| Hiện trạng | Hệ quả | Hướng xử lý |
|---|---|---|
| `source_manifest()` hash toàn bộ src/scripts/predict.py và phụ thuộc Git | Chỉ sửa renderer cũng làm pha evaluate từ chối; cài wheel hoặc chạy snapshot ngoài Git khó hơn | Manifest có phiên bản; fingerprint theo pha train/predict/evaluate/report và snapshot thực tế |
| `core.autocrlf=true`; sau checkout, bốn file chỉ khác LF/CRLF so với snapshot kiểm chứng | SHA byte đổi dù nội dung nguồn sau chuẩn hóa dòng không đổi | Tách hash byte gốc và hash nguồn chuẩn hóa có chính sách rõ ràng |
| Script 17 import hàm trích đặc trưng từ script 16 bằng importlib | Logic nghiệp vụ nằm trong entrypoint, khó tái dùng và test độc lập | Chuyển trích đặc trưng thành API trong package |
| Baseline, plugin detector, benchmark 15/16 và Champion có các đường xử lý khác nhau | Dễ lệch gate/frame/Clipping khi thêm thí nghiệm | Một engine chung, khác biệt được mô tả bằng spec và kiểm tra feature space |
| `fit_models()` trong protocols.py ghi cố định metadata 25/10 ms | Có thể dùng nhầm metadata nếu tái sử dụng cho cấu hình khác | Model luôn gắn feature spec thực tế, không suy ra từ mặc định |
| Script 17/21 lặp logic freeze/snapshot/runtime/CSV; cache dùng monkey-patch toàn cục | Khó quản lý trạng thái run, khả năng chạy đồng thời và khôi phục sau lỗi | Run store và runner chung; cache được truyền tường minh |
| Tính metric, làm tròn và format còn gắn nhau | Refactor nhỏ có thể đổi điểm hoặc thứ hạng vì làm tròn | Đóng băng hợp đồng metric cũ; format là tầng riêng |
| Script 19/21 vừa đọc dữ liệu vừa vẽ, ghi báo cáo hoặc sửa docs | Renderer khó dùng độc lập; dễ ghi đè kết quả khác thời điểm | Report bundle bất biến; xuất bản vào docs là bước riêng |
| Nhiều thư mục outputs đặt theo đợt, danh mục cập nhật thủ công | Khó biết báo cáo nào dùng run/model/code nào | Registry run và liên kết provenance tới từng bảng/hình |

Các kiểm tra LF/CRLF ở trên chỉ phát hiện thay đổi xuống dòng tại `scripts/20_verify_validation_artifacts.py`, `scripts/21_benchmark_champion_unsupervised.py`, `src/analysis/champion_unsupervised.py`, `src/core/lab_parser.py`; không phát hiện thay đổi nội dung nguồn khác sau merge. Đây không phải lý do bỏ kiểm tra source.

## Cấu trúc đích

Giữ package `src` và các đường CLI cũ trong những đợt đầu, tránh trộn việc đổi tên package với việc tách logic. Cấu trúc bên trong dự kiến:

```text
configs/                         # Experiment spec JSON, cấu hình đầy đủ
src/
  core/                          # Audio, LAB, ACF/AMDF/Cepstrum thuần tính toán
  pipeline/                      # Feature extraction, decision, post-processing
  experiments/                   # Spec, manifest, run store, runner, registry
  metrics/                       # Metric có phiên bản, aggregate, comparison
  reporting/                     # View model, bảng, hình, bundle, publish vào docs
  cli/                           # Parse args và gọi các service
scripts/                         # Wrapper tương thích tên/lệnh cũ
outputs/
  runs/<run_id>/
    manifest.json
    config.json
    source_snapshot.tar.gz
    environment.json
    models/
    features/
    predictions/
    metrics/per_file.jsonl
    metrics/aggregate.json
    reports/<report_id>/          # Markdown, CSV, PNG, manifest của báo cáo
  registry.json                  # Có thể xây lại từ các manifest
```

Đây là cấu trúc cho run mới. Không chuyển/xóa toàn bộ outputs cũ ngay; registry trước hết đăng ký chúng bằng adapter chỉ đọc. Khi cần chuyển một run, tạo bundle mới có mapping và kiểm chứng hash; giữ đường cũ hoặc liên kết tương thích.

## Các quy tắc thiết kế cần chốt

1. **Spec đầy đủ và có phiên bản.** Dùng JSON và dataclass của thư viện chuẩn trước, chưa cần thêm framework cấu hình. Spec gồm thuật toán, frame/hop/range, gate, pre/post plugin với đủ tham số, feature/ZCR source, fit population, chế độ frozen/input-adaptive, Energy guard, metric version và seed nếu có. Không dùng mặc định toàn cục làm nguồn tham số ẩn trong file model.
2. **Danh tính thí nghiệm khác danh tính lần chạy.** Experiment ID dựa vào cấu hình/dữ liệu/code cần thiết; run ID thêm mã thực thi duy nhất. Cùng một experiment có thể chạy lại để kiểm chứng, không ghi đè run cũ. Không đưa đường dẫn tuyệt đối của máy vào danh tính dữ liệu.
3. **Hash có ý nghĩa rõ ràng.** WAV và artifact nhị phân giữ SHA byte. Nguồn văn bản mới lưu cả raw SHA và canonical SHA, với phiên bản quy tắc UTF-8/LF; không bỏ qua thay đổi logic. Ghi Git commit, dirty state, snapshot/patch thực tế và dependency lock. Chốt EOL cho các đường mới; không renormalize toàn repo hoặc thay SHA legacy âm thầm.
4. **Tách fingerprint theo pha.** Thay template chỉ sinh report revision mới; thay extractor hoặc feature spec buộc train lại. Thay metric/parser làm evaluation revision mới, không sửa nhãn/số liệu đã lưu. Dependency scope được khai báo và kiểm thử, không suy đoán chỉ dựa tên file.
5. **Run bất biến và ghi an toàn.** Các pha `prepare → train/freeze → evaluate → report → verify` có trạng thái và artifact digest riêng. Ghi tạm rồi thay thế atomically, khóa writer khi cần; lỗi giữa chừng không được gắn trạng thái complete. Resume chỉ khi xác minh đầu vào và artifact đã hoàn thành. Dữ liệu test không tham gia train/selection.
6. **Báo cáo không thực thi mô hình.** Renderer chỉ đọc metric đã xác minh, không fit lại hoặc chọn cấu hình theo test. Mỗi bảng/hình có run ID, model/config/evaluation digest và renderer version. Không dùng glob “file mới nhất” để tự chọn nguồn.
7. **Metric cũ giữ nguyên.** Đặt tên hợp đồng hiện tại là legacy v1, bao gồm cách làm tròn theo file và cách tổng hợp. Dù tách tầng format, phải giữ điểm cũ chính xác. Metric raw/v2 nếu muốn thêm phải là giao thức mới, có đối chứng riêng; không lặng lẽ thay bảng cũ.
8. **Xuất bản là bước tường minh.** Sinh report bundle trong run trước; sau đó mới publish vào các marker được quản lý. Mặc định không sửa README/năm chương. Publish có preview diff, kiểm tra tất cả target trước khi ghi và giữ nguyên phần thủ công.

## Thứ tự thực hiện và điều kiện nghiệm thu

Mỗi đợt là một nhánh/PR riêng, có commit và kết quả đối chứng trước khi merge. Khi đợt trước chưa merge main, nhánh mới kế thừa đợt trước để giữ đầy đủ thay đổi; ghi rõ quan hệ nhánh. Không gộp sáu đợt thành một lần viết lại.

| Đợt | Phạm vi chính | Đầu ra cụ thể | Điều kiện hoàn thành |
|---|---|---|---|
| **1. Manifest và danh mục run** | Tách phần source/data/runtime/freeze trong `analysis/validation.py` thành `experiments/spec.py`, `manifest.py`, `store.py`, `registry.py` | Schema v1, chính sách EOL/hash; adapter chỉ đọc cho Stage 4 và benchmark 21; lệnh list/verify | Run cũ vẫn xác minh; chuyển thư mục không đổi danh tính dữ liệu; phát hiện thay đổi thật; LF/CRLF không bị nhầm với thay thuật toán theo chính sách mới |
| **2. Pipeline và feature contract** | PitchDetector, PluginPitchDetector, trích đặc trưng trong script 15/16, protocols.py và champion_unsupervised.py | FeatureSpec/FrameFeatures/PredictionResult; engine dùng chung; cache có dependency tường minh | Gate, label, timeline, candidate và F0 của 288 trường hợp khớp; train_all/train_active và raw/processed vẫn là các giao thức riêng |
| **3. Runner chung** | Rút logic khỏi script 17/21; tái dùng store/config ở đợt 1 và engine ở đợt 2 | Service train/freeze/evaluate/replay/verify; script cũ thành wrapper; run mới tự ghi đủ môi trường và trạng thái | Train không đọc test để fit; evaluate dùng đúng freeze; chạy lại ở thư mục mới; mô phỏng crash/resume và hai writer không trộn artifact |
| **4. Metric và đối sánh** | Gom error metrics, aggregate và serializer đang nằm ở nhiều script | API legacy v1 thống nhất; schema missing/N/A; comparison service kiểm tra metadata | 48 bản ghi baseline/Champion và toàn bộ 288 bộ metric khớp; không bỏ file thiếu reference; không so sánh nhầm 25/30 ms hoặc frozen/input-adaptive |
| **5. Reporting** | Tách script 19/21 và plotter thành renderer/view model/template | Report bundle + manifest; bảng/hình từ một nguồn metric; publisher chỉ sửa marker; registry tạo danh mục | Render không gọi detector/fit; chạy hai lần không làm đổi nội dung bảng; không đổi freeze/evaluation; lỗi marker không ghi dở; link bảng/hình trỏ đúng run |
| **6. CLI, đóng gói và hướng dẫn** | Tạo entrypoint chung, giữ wrapper cũ; kiểm tra wheel và môi trường sạch | CLI cho predict/experiment/report/runs; cấu hình mẫu; hướng dẫn tái lập từ checkout hoặc bundle | Lệnh cũ vẫn chạy; predict không cần Git/.git; cài wheel chạy ngoài repo; CI dùng lock và kiểm tra runtime được hỗ trợ trên Windows/Linux |

Phần triển khai CLI ở đợt 6 có thể cung cấp lựa chọn tám phương pháp bằng model/spec được chỉ định. Không tự đổi Champion mặc định theo kết quả test đã quan sát. Đổi tên package `src` chỉ cân nhắc khi interface đã ổn định và có lớp tương thích cho import/lệnh cũ.

## Hợp đồng kiểm chứng xuyên suốt

| Nội dung | Cách kiểm tra |
|---|---|
| Thuật toán không đổi | Cùng môi trường: so label/count/timeline/candidate/contour trực tiếp với mốc đã lưu; giữ đủ 288 trường hợp và 93.423 dòng |
| Model không nhầm feature space | Từ chối frame/range/preprocess/fit population/ZCR source khác; không refit ở chế độ frozen |
| Báo cáo không làm đổi kết quả | Hash freeze/evaluation trước/sau render/publish; đối chiếu vùng văn bản ngoài marker |
| Tái lập nhiều máy | Clone/cài ở đường khác, Git checkout LF/CRLF, wheel ngoài repo; kiểm tra manifest mới và adapter legacy riêng |
| Phiên bản số học | Label/count và metric legacy phải khớp; mọi tolerance cho float raw trên môi trường khác được chốt trước kiểm thử, không nới sau khi thấy lỗi |
| Hình ảnh | Metric, nhãn trục, đơn vị và nguồn dữ liệu phải đúng; PNG không bắt buộc cùng SHA giữa font/backend khác nhau |
| Môi trường | Lưu Python, numpy/scipy/matplotlib, lock digest và cấu hình thí nghiệm; AST Python 3.9 hiện có không thay cho test runtime thực |
| Đầu ra cũ | Adapter đọc được schema cũ; không sửa provenance hoặc xóa nguồn lịch sử để làm test qua |

Chỉ chạy đối chứng cần cho đợt đang làm; không quét lại 222 cấu hình train mỗi khi sửa renderer. Mốc kiểm chứng nào bị thay theo chủ đích phải mở thí nghiệm mới có version, không cập nhật expected value cho qua refactor.

## Kế hoạch riêng cho kho báo cáo

- Registry chứa run ID, đường dẫn tương đối, dataset/version, giao thức, metric version, thời điểm và trạng thái kiểm chứng. Các alias như `champion-frozen` hoặc `submission` do người quản lý chọn tường minh, không lấy run có điểm test thấp nhất.
- Báo cáo viết tay nằm ở docs; phần sinh tự động nằm trong report bundle và chỉ được nhúng vào vùng đã đánh dấu. Một báo cáo có thể dẫn nhiều run nhưng phải hiển thị nguồn từng bảng, không ghép train từ run này với test run khác.
- Config, manifest, metric gọn và tài liệu nằm trong Git. Với run mới, cân nhắc bundle nén cho contour/feature/hình lớn; nếu chuyển sang lưu ngoài Git phải có vị trí tải, SHA và kiểm thử truy xuất trước. Không đưa thêm Git LFS hoặc dịch vụ lưu trữ ngay trong đợt đầu.
- Dựng lại danh mục outputs từ registry, tránh sửa tay nhiều README sau mỗi lần chạy. Việc publish báo cáo môn học/Word/slide là công việc sau, không trộn vào refactor engine.

## Bắt đầu từ đâu

**Đợt 1–2 đã qua nghiệm thu:** schema/hash có phiên bản, hai run được đăng ký bằng adapter chỉ đọc, CLI list/verify/index hoạt động. `src/pipeline` có feature contract, engine, extractor raw/Champion và cache theo run; script 17 không import script 16; metadata model lấy từ cấu hình thực. 152 test đạt; 78 nhóm dữ liệu trước refactor, 288 trường hợp, 93.423 contour và 3.873 dòng đặc trưng train khớp chính xác. Không di chuyển outputs hoặc thay công thức/preset. Bước tiếp theo là service runner chung ở đợt 3.

Phần tối ưu Cepstrum, bổ sung tập test độc lập và nhãn theo thời gian vẫn thuộc phát triển/đánh giá thuật toán, không phải điều kiện để hoàn thành refactor tổ chức mã. Refactor này chỉ tạo nền để thực hiện các việc đó có thể tái lập và truy xuất được.
