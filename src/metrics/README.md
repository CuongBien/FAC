# Metrics

| Module | Trách nhiệm |
|---|---|
| `legacy.py` | Công thức từng file `legacy-v1`, gồm alias xuất dữ liệu cũ |
| `numeric.py` | Chuẩn hóa số/N/A, làm tròn, trung bình đầy đủ quần thể, ranking |
| `aggregation.py` | Profile tổng hợp có tên cho runner và benchmark cũ |
| `evaluation.py` | Adapter LAB và kiểm tra alignment/range/count |
| `formatting.py` | Chuỗi hiển thị và giá trị CSV; không tính metric |
| `contracts.py` | Hợp đồng đơn vị/N/A và `EvaluationContext` bất biến |
| `comparison.py` | `MetricSeries`, đọc series đã xác minh và đối sánh có kiểm tra metadata |
| `bundle.py` | Adapter thuần dữ liệu cho hai giao thức v1 |

I/O của sidecar nằm trong `experiments/metric_artifacts.py`. Thuật toán pitch và fit model không chuyển vào package này. Import cũ từ `analysis.evaluation`/`analysis.validation` tiếp tục được re-export.

Xem [hợp đồng số học và đối sánh](../../docs/METRICS.md).
