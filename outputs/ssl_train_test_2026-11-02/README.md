# Pitch-Shift gốc — train và test

Giữ các kết quả Pitch-Shift ACF-only ban đầu; chỉ loại dữ liệu của các phương pháp khác. Điểm giữ nguyên, tính trực tiếp theo script lúc đo.

| Phương pháp | ACF train → test | AMDF train → test | Cepstrum train → test |
|---|---:|---:|---:|
| Champion | 2.213 → 2.384 | 2.080 → 2.755 | 11.389 → 22.943 |
| Pitch-Shift ACF-only | 2.960 → 2.745 | 2.900 → 3.240 | 5.208 → 3.750 |

Pitch-Shift độc lập train → test: 6.672 → 7.214%.

[Dữ liệu đầy đủ](results.json) · [CSV từng file](per_file.csv).
