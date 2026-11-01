# Pipeline dùng chung

| Module | Trách nhiệm |
|---|---|
| `contracts.py` | `FeatureSpec`, `FrameFeatures`, `PredictionResult`; kiểm tra feature space và alignment |
| `engine.py` | Một vòng framing → energy gate → frame plugins → pitch metric → decision hooks → median |
| `features.py` | Trích xuất raw cho benchmark 15/16/17, không phụ thuộc entrypoint |
| `champion.py` | Factory/trích xuất theo feature space Champion; không fit mô hình |
| `cache.py` | `PitchCache` truyền tường minh; bridge `pitch_cache()` cho caller cũ |

Post-processing vẫn do `PluginPitchDetector` điều phối Energy/Viterbi theo thứ tự cũ. ACF/AMDF/Cepstrum vẫn nằm trong `src/core`; các phép fit và quyết định không giám sát vẫn nằm trong `src/analysis`. Import engine không kéo theo analysis, matplotlib hoặc script benchmark.

Xem [hợp đồng, giao thức và kiểm chứng](../../docs/PIPELINE.md).
