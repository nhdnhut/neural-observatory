# Kịch bản bảo vệ 6 phút

**0:00–0:45 — Bài toán.** Một mạng tự viết học gì, hoạt động ra sao và giữ được bao nhiêu accuracy dưới ngân sách dung lượng? Cho xem repo gốc và `legacy/`.

**0:45–1:30 — Tính đúng.** Forward → toàn bộ gradient → cập nhật. Chạy `python main.py test`, giải thích một phép sai phân hữu hạn. NumPy chỉ làm phép toán mảng; Python thuần đã train riêng.

**1:30–2:30 — Show layers.** Mở `artifacts/Neural_Observatory.html`, chọn ảnh test. Kéo từ epoch 0 đến checkpoint đã chọn. Giải thích 784 đầu vào, 64 neuron, 32 neuron, 10 đầu ra. Đây là activation thật.

**2:30–3:30 — Can thiệp.** Chọn neuron, đặt activation = 0, theo dõi xác suất rồi khôi phục. Một neuron có thể ảnh hưởng khác nhau trên các ảnh. Weight map không phải biểu tượng ngữ nghĩa chắc chắn.

**3:30–4:30 — Nén.** FP32 → INT8 ở cùng checkpoint. Chọn ảnh đổi dự đoán. So sánh mạng 64–32 INT8 với mạng 20 FP32 dưới 64 KiB. Công bố latency INT8 Python chậm hơn.

**4:30–5:15 — Giới hạn.** Confusion matrix, mẫu sai và dịch ảnh. Accuracy MNIST cao không đảm bảo mạnh ngoài phân phối.

**5:15–6:00 — Tái lập.** Mở `benchmark.csv`, checksum, seed, checkpoint, lệnh chạy lại. Nêu INT8 kernel thật hoặc pruning/retrain là bước tiếp theo, chưa hoàn thành.

## Câu hỏi thường gặp

**Tự viết neural network?** Forward, loss, backprop và optimizer tự viết. NumPy hỗ trợ mảng ở full benchmark; reference, quantizer và packed inference dùng Python chuẩn. Không autograd/framework.

**INT8 nhanh hơn?** Không trong packed Python runtime đã đo. Nó giảm dung lượng. Tăng tốc cần kernel/phần cứng phù hợp và phép đo mới.

**Accuracy cao chứng minh hiểu chữ số?** Chỉ là độ chính xác trên test split này. Báo cáo công khai suy giảm do dịch ảnh.

**Thuật toán mới?** Không. Đóng góp là cài đặt đúng, quan sát/can thiệp được, thực nghiệm có đối chứng và tái lập từ repo gốc.

**Cấm NumPy hoàn toàn?** Dùng `python -S scripts/train_pure.py`, tăng quy mô theo thời gian và báo cáo kết quả riêng; không trình bày full NumPy benchmark như kết quả không thư viện.
