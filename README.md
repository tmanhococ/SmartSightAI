# 🌐 SmartSight AI — Hỗ Trợ Người Khiếm Thị

**SmartSight AI** là một hệ thống hỗ trợ tiếp cận (Accessibility System) chạy cục bộ trên CPU, giúp người khiếm thị nhận biết môi trường xung quanh thông qua các mô tả hình ảnh tự động và phát bằng giọng nói Tiếng Việt.

Hệ thống sử dụng mô hình ngôn ngữ lớn thị giác nhỏ gọn (Tiny VLM) **Moondream2** để phân tích hình ảnh, kết hợp với các cơ chế dịch thuật và Text-to-Speech (TTS) thông minh hỗ trợ tự động chuyển đổi sang chế độ Offline khi mất kết nối mạng.

---

## 🚀 Tính năng nổi bật

1. **Mô tả hình ảnh thông minh**: Sử dụng Moondream2 (hỗ trợ cả phiên bản 2B và 0.5B cực nhẹ) chạy cục bộ để caption/mô tả môi trường và vật thể.
2. **Hỗ trợ camera & webcam**: Chụp ảnh trực tiếp từ webcam hoặc tải lên file ảnh có sẵn từ thiết bị.
3. **Cơ chế dịch thuật thông minh (EN → VI)**:
   - *Online (Mặc định)*: Sử dụng Google Translate API cho bản dịch tự nhiên nhất.
   - *Offline (Fallback)*: Tự động chuyển hướng sang mô hình dịch thuật cục bộ Hugging Face `Helsinki-NLP/opus-mt-en-vi` khi mất kết nối Internet.
4. **Text-to-Speech phát giọng nói Tiếng Việt**:
   - *Online (Mặc định)*: Sử dụng Google TTS (`gTTS`) cho giọng đọc tự nhiên.
   - *Offline (Fallback)*: Tự động chuyển hướng sang thư viện cục bộ `pyttsx3` tìm và sử dụng giọng đọc Tiếng Việt có sẵn trên hệ thống.
5. **Dashboard giám sát hiệu năng**: Đo lường chi tiết thời gian xử lý từng giai đoạn (Preprocess, VLM, Translation, TTS) và dung lượng RAM sử dụng thời gian thực.
6. **Kiểm soát quy trình**: Tích hợp hàng đợi (Queueing) và nút "Cancel" hủy tác vụ đang chạy trên giao diện để tránh rò rỉ hoặc quá tải CPU.

---

## 🛠️ Yêu cầu hệ thống

- Python 3.9 trở lên
- RAM: Tối thiểu 4GB (Khuyến nghị 8GB+ để chạy bản 2B mượt mà)
- Hệ điều hành: Windows, macOS hoặc Linux

---

## 📦 Hướng dẫn cài đặt

1. **Clone repository:**
   ```bash
   git clone https://github.com/tmanhococ/SmartSightAI.git
   cd SmartSightAI
   ```

2. **Cài đặt các gói phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Hướng dẫn khởi chạy ứng dụng

Chạy lệnh sau từ thư mục gốc của dự án:
```bash
python src/app.py
```
Giao diện ứng dụng Gradio sẽ khởi chạy tại: `http://127.0.0.1:7860/`

---

## 📐 Kiến trúc thư mục dự án

```
SmartSightAI/
├── docs/                   # Tài liệu đặc tả và kế hoạch thiết kế
├── src/
│   ├── pipeline/
│   │   ├── preprocess.py   # Tiền xử lý ảnh (chuyển RGB, resize, kiểm tra kích thước)
│   │   ├── vision_model.py # Xử lý suy diễn với VLM Moondream2
│   │   ├── translate.py    # Module dịch thuật EN-VI với cơ chế Offline Fallback
│   │   └── tts.py          # Module chuyển văn bản thành giọng đọc Tiếng Việt (gTTS/pyttsx3)
│   ├── utils/
│   │   └── monitor.py      # Giám sát thời gian xử lý và RAM tiêu thụ
│   ├── registry.py         # Singleton Model Registry quản lý cache các mô hình
│   └── app.py              # Main Gradio web application
├── tests/                  # Bộ unit test suite của dự án (TDD)
├── requirements.txt        # Định nghĩa các package phụ thuộc
└── README.md               # Hướng dẫn sử dụng dự án
```

---

## 🧪 Chạy Kiểm thử (Unit Tests)

Bộ kiểm thử của dự án được viết hoàn toàn biệt lập (sử dụng Mocks để tránh tự động tải model nặng trong quá trình test). Để chạy kiểm thử:
```bash
pytest -v
```
Toàn bộ 29 ca kiểm thử sẽ được chạy và hiển thị kết quả thành công.
