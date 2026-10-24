# Bàn giao Neural Observatory

Mở `artifacts/Neural_Observatory.html` để xem bản trực quan offline. Đọc `docs/PROJECT_DEFINITION.md` và `docs/EXPERIMENT_REPORT.md`. Mã nguồn, 3 notebook, dataset MNIST, checkpoint, 24 mô hình và số đo đều nằm trong gói này.

Nguồn gốc: https://github.com/DangNhutNguyen/Neural-Network-from-Scratch
Base commit: c3ce89b4cc22e034611287e30af1b5dda6d17bbe
Local branch: codex/neural-observatory
Local commit: 6138becd3dc7e5d46a3c94bdde56ea2e7b07e389

Kết nối GitHub hiện chỉ có quyền đọc, nên chưa push lên remote. Patch mã nguồn ở `Neural_Observatory.patch`; các model/checkpoint/dataset được cung cấp trong ZIP, không chứa trong patch.

Nếu muốn áp dụng vào một bản clone sạch đúng base commit, tạo nhánh mới rồi dùng `git am /path/to/Neural_Observatory.patch`. Nếu repo đã thay đổi, kiểm tra diff và xử lý xung đột trước.

Notebook là hướng dẫn tái lập với cell chưa chạy; các kết quả thực thi thật được lưu riêng tại `artifacts/`.
