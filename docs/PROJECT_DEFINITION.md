# Neural Observatory

**Tên đề tài chính thức:** Xây dựng từ đầu, quan sát và nén mạng neural nhận dạng chữ số dưới ngân sách bộ nhớ.

**Tác giả dự án gốc:** Nguyễn Huỳnh Đăng Nhựt — 26521522.

**Tên sản phẩm:** Neural Observatory.

## 1. Định nghĩa bài toán

Xây dựng mạng neural truyền thẳng bằng forward và backpropagation tự cài đặt; huấn luyện trên MNIST; ghi trạng thái mô hình qua từng epoch; cung cấp công cụ quan sát activation, gradient và ảnh hưởng của can thiệp neuron; lượng tử hóa trọng số để khảo sát đánh đổi giữa độ chính xác, dung lượng lưu mô hình và độ trễ suy luận trên CPU.

Người dùng cung cấp ảnh xám 28 × 28 hoặc chọn ảnh test, một checkpoint và chế độ trọng số. Hệ thống trả về phân phối softmax trên 10 chữ số, nhãn dự đoán, activation từng lớp, độ nhạy của logit theo pixel và thay đổi sau can thiệp. Ở cấp thực nghiệm, người dùng đặt ngân sách dung lượng tệp và xem cấu hình nào trong tập ứng viên đáp ứng ngân sách, được ưu tiên theo điểm validation.

## 2. Bốn câu hỏi có thể kiểm chứng

1. Forward và backprop có khớp đạo hàm số và bản cài đặt đối chiếu không?
2. Các cấu hình mạng đạt accuracy và dung lượng nào với cùng dữ liệu, seed và quy trình đánh giá?
3. Với cùng checkpoint, lượng tử hóa INT8 thay đổi accuracy, tệp lưu và độ trễ bao nhiêu?
4. Trạng thái lớp mạng thay đổi ra sao qua huấn luyện, khi tắt neuron hoặc khi dịch ảnh?

## 3. Đầu vào, đầu ra và ràng buộc

| Mục | Định nghĩa |
|---|---|
| Dữ liệu | MNIST: 60.000 ảnh training ban đầu, 10.000 ảnh test |
| Chia tập | 55.000 train, 5.000 validation (500 mỗi lớp), 10.000 test giữ nguyên |
| Tiền xử lý | Pixel / 255; không augmentation trong nghiên cứu chính |
| Nhiệm vụ | Phân loại 10 chữ số, không phải OCR văn bản hoặc LLM |
| Core không thư viện ngoài | Forward, stable softmax, cross-entropy, backprop, SGD, đọc IDX, INT8, serialization, inference |
| Backend train lớn | NumPy làm phép toán mảng; không autograd, PyTorch, TensorFlow hoặc sklearn |
| Tài nguyên | Ngân sách dung lượng tệp; đo thêm packed weight buffer và latency |
| Giao diện | HTML/CSS/JavaScript thuần; suy luận trực tiếp bằng trọng số thật, chạy offline |
| Tái lập | Seed, checksum, split, cấu hình, lịch sử epoch, checkpoint, số đo thô |

## 4. Phần Nhựt cần hiểu và bảo vệ được

- Layer tính `z = W a + b`; ReLU/sigmoid ở lớp ẩn, softmax ở lớp ra.
- Cross-entropy tính ổn định bằng log-sum-exp.
- Backprop dùng trọng số của cùng lượt forward: tính đủ gradient trước khi cập nhật.
- SGD momentum tự viết; NumPy không quyết định thuật toán học.
- Quantizer tính scale mỗi neuron đầu ra, làm tròn và chặn trọng số trong [-127,127].
- Trọng số lưu thành byte có dấu thực sự; bias và scale vẫn float32.
- Trình duyệt tính lại dự đoán và activation khi thay ảnh hoặc can thiệp.
- Activation bằng 0 trên một ảnh không đồng nghĩa neuron chết trên mọi dữ liệu.
- Saliency là độ nhạy đạo hàm cục bộ; weight map không chứng minh neuron hiểu nét chữ.
- Tắt neuron là can thiệp tính toán trong mô hình; không phải nhân quả về thế giới thực.

## 5. Đóng góp

Chuỗi có thể kiểm tra từ toán học đến hệ thống: implementation chuẩn → kiểm tra gradient → training có checkpoint → quan sát trạng thái thật → can thiệp neuron → nén trọng số → đánh giá theo ngân sách. Không tuyên bố phát minh backprop hay quantization mới, đạt SOTA, hoặc suy rộng MNIST sang LLM.

## 6. Kế thừa repo

Nguồn: https://github.com/DangNhutNguyen/Neural-Network-from-Scratch, commit `c3ce89b4cc22e034611287e30af1b5dda6d17bbe`.

`legacy/` giữ nguyên file gốc. Bản mới giữ quy ước trọng số `[output,input]` và ý tưởng MLP handwritten digit, sửa thứ tự cập nhật và việc thiếu test split, thay sklearn loader bằng IDX loader, mở rộng thành engine nhiều lớp. Đối chứng 20 sigmoid kế thừa kích thước và hidden activation nhưng dùng softmax/cross-entropy và quy trình mới; không phải kết quả của script gốc chưa sửa.

## 7. Nghiệm thu

- Sai phân hữu hạn cho trọng số và bias; bản Python và NumPy khớp.
- Train thật không thư viện bên thứ ba, ghi rõ tập con và số epoch.
- Nghiên cứu chính: 4 cấu hình × 3 seed × 2 định dạng, test độc lập.
- Mô hình nhị phân FP32/INT8 lưu và nạp lại được.
- Giao diện xem ảnh, checkpoint, lớp, neuron, trọng số, saliency và can thiệp.
- Số liệu lấy từ file thí nghiệm; công bố cả latency chậm hơn và lỗi ảnh bị dịch.

## 8. Quy định thư viện

Nếu cấm thư viện ngoài trên toàn đồ án, dùng `observatory/reference.py`, `dataset.py`, `packing.py`, `scripts/train_pure.py` và JavaScript. Full benchmark hiện dùng NumPy và phải khai báo. Nếu cho phép NumPy như công cụ ma trận, có thể dùng nghiên cứu chính. Colab GPU không tự tăng tốc NumPy CPU.

## 9. Chưa thực hiện

Chưa có Transformer, language model, speculative decoding, integer-only kernels, pruning sau retrain, mobile deployment, đo điện năng hoặc RAM toàn tiến trình. Không tính chúng vào kết quả đã hoàn thành.

## 10. Nguồn

- Repo gốc: https://github.com/DangNhutNguyen/Neural-Network-from-Scratch
- MNIST mirror/IDX: https://github.com/cvdfoundation/mnist
- Jacob et al., CVPR 2018: https://arxiv.org/abs/1712.05877 — tài liệu nền; bản này chỉ weight-only quantization.
