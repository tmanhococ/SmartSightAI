# BẢN ĐẶC TẢ THIẾT KẾ HỆ THỐNG
## AI Hỗ Trợ Người Khiếm Thị — Mô Tả Hình Ảnh Bằng Tiếng Việt (Gradio Prototype)

| Thông tin | Chi tiết |
| :---- | :---- |
| **Loại dự án** | AI Hỗ trợ Tiếp cận / Tác động xã hội |
| **Đối tượng** | Người khiếm thị và người thị lực kém tại Việt Nam |
| **Công cụ chính** | Gradio, Moondream (0.5B & 2B), Helsinki-NLP, gTTS, pyttsx3 |
| **Môi trường chạy** | CPU Laptop thông thường, hỗ trợ deploy Hugging Face Spaces |
| **Ngày lập spec** | 2026-06-04 |

---

## 1. Mục tiêu dự án
Xây dựng một hệ thống hỗ trợ tiếp cận (Accessibility System) chạy trên giao diện Web (Gradio), cho phép tiếp nhận hình ảnh từ webcam hoặc file tải lên, chạy mô hình ngôn ngữ thị giác (VLM) cục bộ để mô tả hình ảnh, dịch sang Tiếng Việt và phát ra giọng đọc. 

Hệ thống hỗ trợ cả chế độ Online (độ chính xác cao, giọng đọc tự nhiên) và chế độ Offline hoàn toàn (sử dụng các mô hình chạy cục bộ làm fallback).

---

## 2. Kiến trúc Hệ thống & Luồng dữ liệu

Dự án được xây dựng dựa trên cấu trúc Pipeline tuần tự gồm 4 công đoạn (Stages):

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Stage 1:    │     │  Stage 2:    │     │  Stage 3:    │     │  Stage 4:    │
│ Preprocess   ├────>│  VLM (VLM)   ├────>│ Translation  ├────>│ Text-To-Sp.  │
│  (PIL RGB)   │     │ (Moondream)  │     │  (EN -> VI)  │     │    (TTS)     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.1. Quản lý mô hình (Model Registry)
Sử dụng Singleton Pattern thông qua một class `ModelRegistry` để quản lý việc khởi tạo và chia sẻ tài nguyên trong RAM, tránh rò rỉ bộ nhớ khi chạy đa phiên làm việc trên Gradio.

* **Moondream VLM**: Hỗ trợ chuyển đổi động giữa hai phiên bản `0.5B` và `2B` dựa trên lựa chọn của người dùng. Mô hình được cache sau khi tải lần đầu.
* **Helsinki-NLP**: Tải mô hình offline `Helsinki-NLP/opus-mt-en-vi` phục vụ chức năng dịch offline.

### 2.2. Xử lý đồng thời & Xếp hàng (Queuing & Concurrency)
Sử dụng cơ chế `queue()` tích hợp sẵn của Gradio để quản lý các request đồng thời. Khi nhiều người dùng cùng bấm "Run", các request sẽ được xếp vào hàng đợi và xử lý tuần tự trên CPU, tránh làm quá tải CPU máy chủ (đặc biệt khi chạy trên Hugging Face Spaces free tier với 2 vCPUs).

---

## 3. Thiết kế chi tiết các Module

### Module 1: Tiền xử lý ảnh (Image Preprocessing)
* **Nhiệm vụ**: Nhận đầu vào là ảnh PIL từ Webcam hoặc file upload.
* **Xử lý**: 
  * Đảm bảo định dạng ảnh là RGB (loại bỏ kênh alpha hoặc ảnh grayscale).
  * Cảnh báo nếu kích thước ảnh nhỏ hơn 224x224 px.
  * Cắt ảnh trung tâm (Center Crop) về tối đa 1280px nếu ảnh quá lớn nhằm giữ độ chi tiết khi nén xuống 384x384 px bên trong mô hình Moondream.

### Module 2: Vision-Language Model (Moondream 2)
* **Thư viện sử dụng**: Thư viện Python `moondream` chính thức.
* **Các phiên bản**:
  * **0.5B (Quantized INT4)**: Tối ưu bộ nhớ (~900 MB RAM), chạy cực nhanh trên CPU (~2-3s).
  * **2B**: Mô tả chi tiết, nhận diện ngữ cảnh phức tạp tốt hơn (~2.5 GB RAM, chạy ~5-8s trên CPU).
* **Prompt mặc định**:
  ```
  "Describe what you see in this image briefly and clearly. Focus on the main subject, people, objects, and any important context. Keep it under 3 sentences."
  ```

### Module 3: Dịch thuật (Translation - EN → VI)
Hỗ trợ 2 chế độ cấu hình:
1. **Online (Primary)**: Dùng `deep-translator` hoặc API dịch thuật của Google. Độ trễ thấp (<0.5s), dịch tự nhiên.
2. **Offline (Fallback)**: Dùng mô hình Transformer `Helsinki-NLP/opus-mt-en-vi` chạy cục bộ. Không cần internet, dung lượng ~300MB.

### Module 4: Text-To-Speech (TTS)
Hỗ trợ 2 chế độ cấu hình:
1. **Online (Primary)**: Sử dụng thư viện `gTTS` (Google Text-to-Speech) để sinh file âm thanh giọng đọc tiếng Việt tự nhiên dạng `.mp3`.
2. **Offline (Fallback)**: Sử dụng thư viện `pyttsx3` để tổng hợp giọng nói trực tiếp offline (chất lượng giọng robot cơ bản).

---

## 4. Thiết kế Giao diện người dùng (Gradio UI)

Giao diện được xây dựng bằng `gr.Blocks` với phong cách thiết kế hiện đại, hỗ trợ tối đa cho việc thử nghiệm các tham số.

### 4.1. Layout chính

* **Cột điều khiển trái (Input & Control Panel)**:
  * **Đầu vào ảnh**: `gr.Image(sources=["webcam", "upload"], type="pil", label="Đầu vào hình ảnh")`
  * **Chọn phiên bản VLM**: `gr.Radio(choices=["0.5B", "2B"], value="0.5B", label="Phiên bản Moondream")`
  * **Hàng nút bấm**:
    * Nút **Run** (màu primary) để bắt đầu pipeline.
    * Nút **Cancel** (màu stop) để ngắt ngay lập tức tiến trình tính toán của mô hình.
* **Cột kết quả phải (Outputs Panel)**:
  * Khung text tiếng Anh: `gr.Textbox(label="Mô tả Tiếng Anh (VLM Output)")`
  * Khung text tiếng Việt: `gr.Textbox(label="Mô tả Tiếng Việt (Dịch)")`
  * Trình phát âm thanh: `gr.Audio(label="Giọng đọc Tiếng Việt", autoplay=True)`
* **Bảng Cấu hình nâng cao (Accordion - Parameters & Thresholds)**:
  * Lựa chọn phương thức dịch: `gr.Dropdown(choices=["Online (Google)", "Offline (Helsinki-NLP)"])`
  * Lựa chọn phương thức TTS: `gr.Dropdown(choices=["Online (gTTS)", "Offline (pyttsx3)"])`
  * Custom Prompt cho VLM: `gr.Textbox(lines=2, label="VLM Prompt Template")`
* **Dashboard Hiệu năng (Performance Dashboard)**:
  * Sử dụng các khối `gr.Milestone` hoặc `gr.Label` hiển thị:
    * **Tổng thời gian chạy (Total Time)**
    * **Dung lượng RAM tiêu thụ (RAM Usage)** sử dụng thư viện `psutil`.
  * Vẽ bảng/biểu đồ cột đơn giản (chạy bằng code python sinh HTML hoặc dataframe) mô tả chi tiết thời gian xử lý của từng bước (Preprocess, VLM, Translation, TTS).

### 4.2. Cơ chế nút Cancel trong Gradio
Nút **Cancel** được liên kết trực tiếp với luồng chạy của nút **Run** thông qua cơ chế `cancels` của Gradio:
```python
run_event = run_btn.click(fn=run_pipeline, inputs=[...], outputs=[...])
cancel_btn.click(fn=None, cancels=[run_event])
```
Cơ chế này giúp Gradio tự động hủy generator/luồng xử lý ở backend một cách an toàn và giải phóng tài nguyên CPU ngay lập tức khi người dùng click Cancel.

---

## 5. Kế hoạch triển khai & Kiểm thử (Verification Plan)

### 5.1. Cấu trúc thư mục source code
```
src/
├── app.py                  # Mã nguồn chính chạy giao diện Gradio
├── registry.py             # ModelRegistry (Singleton quản lý mô hình)
├── pipeline/
│   ├── __init__.py
│   ├── preprocess.py       # Module 1: Xử lý kích thước/định dạng ảnh
│   ├── vision_model.py     # Module 2: Load & Inference Moondream (0.5B/2B)
│   ├── translate.py        # Module 3: Dịch tiếng Anh sang Việt
│   └── tts.py              # Module 4: Tạo giọng đọc từ text
└── utils/
    └── monitor.py          # Đo RAM tiêu thụ và đo đạc thời gian
```

### 5.2. Các kịch bản kiểm thử (Test Cases)
1. **Kiểm thử Webcam & Upload**: Kiểm tra xem Gradio có nhận ảnh chuẩn từ cả webcam và file JPG/PNG không.
2. **Kiểm thử Đổi phiên bản VLM**: Chọn đổi từ 0.5B sang 2B, kiểm tra RAM tăng lên và thời gian xử lý thay đổi tương ứng.
3. **Kiểm thử Offline Fallback**: Ngắt kết nối mạng Internet, kiểm tra hệ thống có tự động chuyển sang dịch bằng Helsinki-NLP và đọc bằng pyttsx3 không.
4. **Kiểm thử Cancel**: Bấm "Run" với mô hình 2B (thời gian chạy ~6s), sau đó bấm "Cancel" ở giây thứ 2. Kiểm tra xem tiến trình có dừng ngay lập tức và giao diện sẵn sàng cho lượt chạy tiếp theo không.
5. **Kiểm thử Đồng thời (Concurrency)**: Mở 2 tab trình duyệt cùng lúc bấm Run. Kiểm tra xem Gradio queue có xếp hàng tuần tự và không gây quá tải CPU hay crash app không.
