# Kiểm chứng refactor hạ tầng thí nghiệm — đợt 1

- `validation.json`: 137 test đạt; 288 trường hợp, 93.423 dòng contour và 3.873 dòng đặc trưng train khớp chính xác; AST logic đánh giá/chọn/cache không đổi.
- `source_snapshot.tar.gz`: byte nguồn đúng theo provenance của lần đối chiếu; SHA được ghi trong validation.
- `source_manifest_v1.json`: source scope và cả raw/canonical SHA; Git dirty state ghi rõ lần đối chiếu trên nhánh đang triển khai.
- `dataset_manifests_v1.json`: identity của WAV/LAB dựa trên đường tương đối và byte gốc.

Đây là đối chiếu refactor, không phải lần train hoặc chọn Champion mới. Hai run cũ vẫn nằm nguyên vị trí và được đăng ký trong [registry.json](../registry.json); không sửa freeze/evaluation hoặc provenance lịch sử.

Chi tiết cấu trúc mới và các phần chưa triển khai: [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).
