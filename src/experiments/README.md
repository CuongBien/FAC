# Experiments

Package này sở hữu spec, manifest, snapshot, artifact store, registry và vòng đời thí nghiệm. `runner.py` dùng `transactions.py` để điều phối backend `calibration.py`/`champion.py`; công thức pitch/fit/metric vẫn ở core/pipeline/analysis. `champion_report.py` giữ renderer tương thích trong thời gian chờ refactor reporting.

API mới: `ExperimentSpec`, `RunStore`, `RunRegistry`; hash và dataset/source manifest ở `manifest.py`. Hợp đồng byte cũ nằm ở `legacy.py` và `serialization.write_json`; run mới dùng writer canonical/atomic của `store.py`.

Hướng dẫn hạ tầng: [EXPERIMENTS.md](../../docs/EXPERIMENTS.md). Lifecycle, CLI, resume và giới hạn giao thức v1: [RUNNER.md](../../docs/RUNNER.md).
