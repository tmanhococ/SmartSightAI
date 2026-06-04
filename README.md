---
title: SmartSight AI
emoji: 🌐
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🌐 SmartSight AI — Hugging Face Space Deployment

Bản cập nhật cấu hình cho Hugging Face Spaces chạy mô phỏng trợ lý mô tả hình ảnh hỗ trợ tiếp cận (Accessibility System) bằng giọng nói Tiếng Việt dành cho người khiếm thị.

## 🚀 Cấu hình Hugging Face Spaces

Ứng dụng này được thiết kế để triển khai trực tiếp trên Hugging Face Spaces bằng các cài đặt sau:

- **SDK:** Gradio
- **Phiên bản SDK:** 4.40.0
- **Tệp khởi chạy chính:** `app.py` (tệp wrapper ở thư mục gốc)

---

## 🛠️ Hướng dẫn Triển khai trên Hugging Face Space

1. Tạo một Space mới trên **Hugging Face**.
2. Chọn SDK là **Gradio**.
3. Kết nối Space của bạn với kho chứa mã nguồn GitHub này.
4. Thiết lập nhánh triển khai là `huggingface-deployment`.
5. Hugging Face sẽ tự động cài đặt các thư viện trong `requirements.txt`, khởi chạy `app.py` và chạy ứng dụng.

---

## 📐 Kiến trúc nhánh Hugging Face Deployment

Sự khác biệt chính của nhánh này là việc đưa tệp khởi chạy chính `app.py` ra thư mục gốc dưới dạng wrapper để tương thích hoàn toàn với cơ chế phát hiện tự động của Hugging Face Spaces, đồng thời tích hợp Metadata YAML ở phần đầu của tệp `README.md`.
