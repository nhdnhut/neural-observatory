# Neural Observatory — Báo cáo kết quả thực nghiệm

**Tên đề tài:** Xây dựng từ đầu, quan sát và nén mạng neural nhận dạng chữ số dưới ngân sách bộ nhớ.

**Nguyễn Huỳnh Đăng Nhựt — 26521522.** Phát triển từ Neural-Network-from-Scratch, commit `c3ce89b4cc22e034611287e30af1b5dda6d17bbe`.

## Kết quả chính

Mạng 784–64–32–10 FP32 đạt **97.747% ± 0.046 điểm phần trăm** trên test MNIST qua ba seed. Trọng số INT8 đạt **97.750% ± 0.050 điểm phần trăm**. Tệp giảm từ **210,730 byte** xuống **53,551 byte**, tương đương **3.935 lần** hoặc giảm **74.59%**.

Chênh lệch accuracy trung bình chỉ +0.0033 điểm phần trăm; không có cơ sở để tuyên bố nén làm tăng accuracy. Với seed 42, accuracy giảm nhẹ 97,72% → 97,70%. Phải xem cả ba seed và các ảnh bị đổi dự đoán.

## 1. Bài toán và đóng góp

Bài toán là xây dựng một mạng có thể huấn luyện, quan sát, can thiệp và nén, rồi đo kết quả theo ngân sách tệp mô hình. Đóng góp nằm ở implementation đúng, khả năng kiểm chứng và toàn bộ quy trình đo; không tuyên bố thuật toán mới hoặc SOTA.

Công cụ cho thấy từng lớp thật: ảnh 784 chiều, lớp 64 neuron, lớp 32 neuron, logits và softmax 10 lớp. Khi thay ảnh, checkpoint, precision hoặc tắt neuron, trình duyệt tính lại toàn bộ forward. Những checkpoint được ghi ở epoch thật; không mô phỏng việc training bằng animation.

## 2. Thiết kế thí nghiệm

- Dữ liệu chính thức: 60.000 training và 10.000 test. Tách 500 ảnh mỗi lớp làm validation bằng seed 2026, còn 55.000 train. Tập test không dùng chọn checkpoint.
- Kiểm tra SHA-256 bốn tệp gốc; chỉ chuẩn hóa pixel bằng 255.
- 4 cấu hình × 3 seed (42,43,44), mỗi cấu hình train 18 epoch, batch 128. Tổng cộng 12 lượt train; mỗi checkpoint được xuất FP32/INT8, thành 24 mô hình.
- SGD momentum 0,9 tự viết. LR ban đầu: 0,04 với ReLU, 0,12 với đối chứng sigmoid; cosine decay đến 10%.
- Chọn checkpoint theo validation cross-entropy nhỏ nhất. Các lựa chọn cấu hình được cố định trước khi đánh giá test.
- Đường train mỗi epoch dùng tập con cố định 5.000 ảnh để giảm chi phí log; metric train cuối dùng đủ 55.000 ảnh.
- Mạng seed 42 hiển thị trong explorer dùng epoch **14**, có **52,650** tham số.

## 3. Kết quả đầy đủ

| Kiến trúc | Trọng số | Test accuracy: mean ± SD | Tệp mô hình | Packed Python p50 |
|---|---|---:|---:|---:|
| 784–64–32–10 / ReLU | FP32 | 97.747 ± 0.046% | 210,730 B | 3.842 ms |
| 784–64–32–10 / ReLU | INT8 | 97.750 ± 0.050% | 53,551 B | 4.790 ms |
| 784–20–10 / Sigmoid | FP32 | 95.627 ± 0.143% | 63,770 B | 1.180 ms |
| 784–20–10 / Sigmoid | INT8 | 95.613 ± 0.136% | 16,279 B | 1.376 ms |
| 784–20–10 / ReLU | FP32 | 95.850 ± 0.217% | 63,767 B | 1.112 ms |
| 784–20–10 / ReLU | INT8 | 95.850 ± 0.243% | 16,276 B | 1.405 ms |
| 784–6–10 / ReLU | FP32 | 91.060 ± 0.695% | 19,246 B | 0.340 ms |
| 784–6–10 / ReLU | INT8 | 91.050 ± 0.637% | 5,047 B | 0.410 ms |

SD là độ lệch chuẩn mẫu của ba seed. Đây không phải khoảng tin cậy cho mọi loại chữ viết. Latency trong bảng là trung vị của ba p50 tương ứng ba seed. Dữ liệu từng lần đo nằm trong `experiment.json`; số liệu mỗi run nằm trong `benchmark.csv`.

Đối chứng 20 sigmoid kế thừa kích thước và hidden activation của repo, nhưng dùng bộ train/softmax/loss đúng mới. Vì các cấu hình khác nhau cả kiến trúc, activation hoặc LR, không quy toàn bộ chênh lệch accuracy cho riêng độ sâu.

## 4. Kết quả theo ngân sách

Ở ngân sách 64 KiB, mạng 64–32 INT8 vừa ngân sách (52.30 KiB), trong khi bản FP32 của nó vượt ngân sách (205.79 KiB). Có thể so với mạng 20 neuron FP32 cũng vừa ngân sách. Giao diện chọn ứng viên theo **accuracy validation trung bình**, không dùng test để chọn.

Ngân sách đang nói về **tệp mô hình**, không phải tổng RAM. `packed_weight_buffer_bytes` chỉ đo bộ đệm trọng số, không gồm Python objects, activation, allocator hoặc RAM tiến trình.

## 5. INT8: đúng phạm vi tuyên bố

Quantizer đối xứng theo từng output channel: scale = max(abs(weights))/127, round rồi clip về [-127,127]. Trọng số được đóng gói thành signed byte; scale và bias float32. Hàng toàn 0 dùng scale 1. Bộ runtime Python tính tích với activation số thực và nhân scale mỗi hàng, không dựng lại toàn bộ ma trận FP32.

Đây là **weight-only quantization**, không phải integer-only inference. Trong môi trường đo, p50 chính là 3.842 ms (FP32) và 4.790 ms (INT8). INT8 **chậm hơn** ở runtime Python này. Lợi ích được chứng minh là dung lượng, không phải tăng tốc hoặc giảm điện năng.

## 6. Quan sát và can thiệp

- Weight map lớp 1: xếp 784 trọng số theo vị trí pixel, màu đối xứng âm/dương. Không gọi nó là khái niệm neuron đã hiểu.
- Activation maps: hiện giá trị qua từng lớp cho ảnh/checkpoint đang chọn. Mỗi lớp có thang riêng.
- Ablation: đặt activation của neuron được chọn bằng 0 rồi chạy phần mạng phía sau. Có thể tắt nhiều neuron; thao tác khôi phục trả về forward ban đầu.
- Xếp hạng tác động: ablate từng neuron riêng trên 1.024 ảnh validation. Đây là can thiệp trong mô hình, phụ thuộc dữ liệu và tương tác neuron.
- Saliency: trị tuyệt đối đạo hàm của logit lớp dự đoán theo pixel. Là độ nhạy cục bộ, không phải chứng minh ngữ nghĩa.
- Ảnh minh hoạ cố ý có cả đúng, sai và đổi dự đoán sau nén; không dùng tỉ lệ của nhóm minh hoạ để suy ra accuracy.

## 7. Một giới hạn thực nghiệm rõ ràng

Với seed 42, ảnh test gốc đạt **97.72%**. Chỉ dịch 2 pixel sang phải và điền 0 đã làm accuracy còn **82.15%**. Đây là dấu hiệu nhạy với vị trí của MLP đã train không augmentation. Không dùng phép thử này để điều chỉnh mô hình trong nghiên cứu chính.

Nét vẽ bằng chuột/tay trong demo không cùng phân phối MNIST; xác suất softmax không bảo đảm độ tin cậy trên đầu vào đó.

## 8. Python thuần và quy định học phần

`python -S scripts/train_pure.py --samples 1024 --epochs 2` đã thực sự chạy, dùng thư viện chuẩn. Mạng 784–20–10 đạt 79,6875% trên 256 ảnh validation của bản nhỏ sau 2 epoch; không đánh giá test. Không so trực tiếp con số này với full benchmark vì dữ liệu, optimizer và số epoch khác.

Reference backprop, softmax, quantization, IDX và inference không có thư viện bên thứ ba. Full benchmark sử dụng NumPy cho phép toán mảng và phải được khai báo. Không có PyTorch, TensorFlow, sklearn hay autograd trong implementation mới. Nếu cấm NumPy toàn bộ, dùng đường Python thuần và tăng quy mô theo thời gian cho phép.

## 9. Kiểm tra và tái lập

Các kiểm tra gồm softmax ở logits lớn, mọi gradient của các mạng nhỏ bằng sai phân hữu hạn, đối chiếu forward/gradient reference–NumPy, bound sai số quantization, zero-row và lưu/nạp mô hình. Giao diện đã kiểm tra 82 mẫu: JavaScript khớp nhãn với Python ở cả FP32 và INT8, sai lệch xác suất lớn nhất lần lượt 1,46e-6 và 1,62e-6. Đạo hàm saliency khớp sai phân hữu hạn với sai lệch lớn nhất 5,13e-10 trên các pixel kiểm tra. Checkpoint, can thiệp neuron, vẽ/căn giữa/xoá, ngân sách và xuất CSV đều qua kiểm tra; giao diện không tràn ngang ở chiều rộng 320 và 390 px. Kết quả nằm trong `artifacts/browser_validation.json`.

Lệnh:

```bash
python main.py test
python -S scripts/train_pure.py --samples 1024 --epochs 2
python main.py train --epochs 18 --seeds 42 43 44
python main.py build
python -S scripts/infer.py --index 0
```

Môi trường: AMD EPYC 9V74 80-Core Processor; Python 3.12.14; NumPy 2.3.5; BLAS 1 thread; Linux-6.18.55-x86_64-with-glibc2.39. Độ trễ không phải kết quả đo trên laptop của người dùng. Mốc UTC trong `protocol.json` là thời gian runtime thực thi, không phải hạn nộp đồ án.

## 10. Phát triển tiếp có kiểm chứng

1. Cài integer kernel thật bằng C/C++ rồi benchmark trên máy mục tiêu.
2. Thử augmentation dịch ảnh, với protocol mới khai báo trước và tập test giữ độc lập.
3. Pruning neuron có fine-tune, đối chứng cùng ngân sách; không chỉ gán trọng số bằng 0.
4. Sau khi kiểm soát được nền tảng, mở nhánh mô hình ngôn ngữ nhỏ. Chưa gọi phiên bản hiện tại là decoding LLM.

## Nguồn

- Repo gốc: https://github.com/DangNhutNguyen/Neural-Network-from-Scratch
- MNIST mirror/IDX: https://github.com/cvdfoundation/mnist
- Nền tảng quantization: https://arxiv.org/abs/1712.05877
