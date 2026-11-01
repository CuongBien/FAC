# Feature contract và engine — refactor đợt 2

`src/pipeline` sở hữu việc trích xuất đặc trưng. `PitchDetector`, `PluginPitchDetector` và benchmark 15/16 dùng cùng engine; script 17 gọi API package thay vì import script 16 bằng `importlib`. `src/pipelines.py` tiếp tục là factory công khai; import/lệnh cũ và kết quả dạng dictionary được giữ.

```text
core/audio + ACF/AMDF/Cepstrum
          ↓
pipeline/contracts + engine + cache
          ↓
core/PitchDetector       plugins/PluginPitchDetector
pipeline/features       pipeline/champion
          ↓
analysis/protocols       analysis/champion_unsupervised
          ↓
scripts/15,16,17,21      predict.py
```

## Giao thức tường minh

Engine dùng các profile để giữ nguyên khác biệt có chủ đích. Không chuyển benchmark cũ thành giao thức Champion.

| Profile | Input/framing | Metric pitch | Frame plugin | ZCR |
|---|---|---|---|---|
| `baseline` | Bỏ DC toàn file; yêu cầu đủ một khung | Chỉ active; đưa frame trực tiếp vào core | Không | Frame pitch được center, chỉ active |
| `plugins` | Bỏ DC toàn file; lọc signal rồi frame; yêu cầu đủ một khung | Chỉ active; center trước frame plugin | BP/Clipping theo cấu hình | Center lại pitch frame sau plugin, chỉ active |
| `raw_all` | WAV đã load; giữ padding của `frame_signal` | Mọi frame, input trực tiếp vào core | Không | Center frame, mọi frame |
| `centered_all` | WAV đã load; giữ padding | Mọi frame, center trước core | Không | Frame đã center, mọi frame |

Script 15 dùng `raw_all`; script 16/17 dùng `centered_all`. Champion dùng `plugins`, kể cả AMDF có danh sách pre-processing rỗng. Ngưỡng silence lấy từ STE của signal trước BP; STE luôn bỏ DC từng frame. Vòng lặp giữ thứ tự frame pre-processing và decision hooks; khung silence vẫn thông báo cho decision plugin nhưng không chạy frame plugin/pitch extraction ở profile native.

## Hợp đồng dữ liệu

`FeatureSpec` bất biến, schema v1, ghi method, frame/hop thực tế, F0 range, silence ratio, profile, pre-processing cùng tham số và chính sách ZCR/metric/window. `FrameFeatures` kiểm tra mảng hữu hạn, timeline, số frame và active mask khớp energy gate. `PredictionResult` kiểm tra alignment, nhãn `v/uv/sil`, voiced count và F0 trong range; ngoài voiced phải bằng 0.

Sample rate nằm trong dữ liệu từng recording. Feature space dùng thời lượng frame/hop theo ms để model train vẫn dùng được với cả 16 kHz và 44,1 kHz theo giao thức hiện tại. Spec không chứa threshold hay quần thể fit: cùng feature space có thể đánh giá nhiều threshold. Model mới ghi riêng `train_all`/`train_active`; Champion chỉ fit `train_active`.

`fit_models()` không còn ghi cứng metadata 25/10 ms hoặc range 70–400 Hz. Nó bắt buộc feature contract đồng nhất và lấy framing/range thực tế từ đó. Prediction từ chối contract khác, gồm raw/processed, frame/hop, range, gate và ZCR. Champion kiểm tra thêm cấu hình đã freeze; Energy vẫn dùng ngưỡng Champion cố định.

Dictionary trả về thêm trường `feature_contract`; import ZCR từ `analysis.advanced_unsupervised` và các hàm trích xuất Champion từ `analysis.champion_unsupervised` vẫn được re-export để tương thích. Caller tự tạo feature dictionary để fit phải khai báo contract; không tự suy ra cấu hình từ mặc định.

Model Stage 4 cũ được adapter diễn giải theo framing đã ghi và giao thức raw lịch sử 70–400 Hz/gate 0,008. Model Champion cũ dùng spec và settings đã freeze. Adapter không sửa model, sidecar hoặc snapshot. Muốn đổi giao thức phải tạo model/run mới.

## Cache theo lần chạy

```python
from src.pipeline.cache import PitchCache
from src.pipelines import build_champion

cache = PitchCache()
detector = build_champion('acf', cache=cache)
result = detector.process_file('TinHieuKiemThu/phone_F2.wav')
print(dict(cache))  # hits/misses
```

Factory và API trích xuất nhận `cache=` tường minh; runner 17/21 dùng cache riêng cho mỗi lần chạy. Key gồm hàm, byte/shape/dtype của frame, sample rate, options và cấu hình candidate Viterbi. Giá trị trả về được copy để caller không làm bẩn cache. Decision hooks, median, Energy và Viterbi DP luôn chạy.

`analysis.validation.pitch_cache()` vẫn có để giữ caller cũ. Bridge dùng `ContextVar`, không thay hàm module/class; nested context và task chạy xen kẽ được cô lập. Không tự chia sẻ một cache giữa các run độc lập.

## Kiểm chứng và giới hạn

```powershell
uv run python -B -m unittest discover -s tests
uv run python -B scripts/22_verify_pipeline_refactor.py verify
```

Script 22 so mốc trước refactor trong [refactor_phase2](../outputs/refactor_phase2/README.md), sau đó replay 288 trường hợp từ mô hình Champion cũ và so metric, nhãn ban đầu/cuối, candidate, timeline, contour cùng feature train. So sánh cùng môi trường dùng đẳng thức chính xác, không nới tolerance. Chế độ `capture` chỉ để lập mốc trên checkout đã duyệt, từ chối ghi đè; không chạy capture để cập nhật expected khi verify thất bại.

Kết quả đợt 2: **152 test đạt**; **78 nhóm dữ liệu**, **288 trường hợp**, **93.423 dòng contour**, **3.873 dòng train feature** khớp chính xác. Dữ liệu, preset, freeze/model, evaluation và báo cáo run cũ giữ nguyên. Bằng chứng source và runtime lưu riêng trong `outputs/refactor_phase2`.

Đợt này chưa hợp nhất runner/store lifecycle hoặc tách renderer. Script 17/21 vẫn có orchestration cũ; guard nguồn rộng tiếp tục hoạt động và có thể từ chối evaluate trực tiếp run freeze trước refactor. Dùng snapshot cũ hoặc phép replay chỉ đọc của script 22; không sửa provenance để vượt guard. Đợt 3 sẽ đưa train/freeze/evaluate vào service dùng spec/store chung.
