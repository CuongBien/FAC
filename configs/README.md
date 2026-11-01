# Experiment configs

`experiments/champion_unsupervised.json` và `experiments/stage4_calibration.json` mô tả đầy đủ hai giao thức v1 hiện hành, gồm pipeline/control/fit/search settings. `ExperimentSpec` kiểm tra schema; runner kiểm tra giao thức có thể thực thi, từ chối giá trị chưa hỗ trợ thay vì bỏ qua chúng.

Script 17/21 đã dùng spec và lifecycle/RunStore chung cho run mới. Truyền `--spec <file>` ở prepare/train; spec được ghim vào run và không áp dụng lại vào bằng chứng lịch sử. Hai giao thức vẫn giữ thuật toán/tham số cũ, chưa cung cấp tuning tùy ý. Xem [runner và resume](../docs/RUNNER.md).
