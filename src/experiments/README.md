# Experiments

Package này sở hữu spec, manifest, snapshot, artifact store và registry. Không đặt thuật toán pitch, training hoặc renderer trong package hạ tầng này.

API mới: `ExperimentSpec`, `RunStore`, `RunRegistry`; hash và dataset/source manifest ở `manifest.py`. Hợp đồng byte cũ nằm ở `legacy.py` và `serialization.write_json`; run mới dùng writer canonical/atomic của `store.py`.

Hướng dẫn, schema, CLI và giới hạn của đợt 1: [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).
