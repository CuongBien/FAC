# Experiment configs

`experiments/champion_unsupervised.json` là spec v1 mẫu có đầy đủ tham số Champion và fit hiện hành. Kiểm tra bằng `ExperimentSpec`; không áp dụng nó âm thầm vào run cũ.

Trong refactor đợt 1, runner 17/21 vẫn dùng hợp đồng hiện có. Đợt 3 mới chuyển chúng sang đọc spec và RunStore chung. Xem [quản lý thí nghiệm](../docs/EXPERIMENTS.md).
