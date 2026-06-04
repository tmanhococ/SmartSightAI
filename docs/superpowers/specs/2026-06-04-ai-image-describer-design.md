# BẢN ĐẶC TẢ THIẾT KẾ HỆ THỐNG (CẬP NHẬT 2 PHIÊN BẢN MOONDREAM2)
## AI Hỗ Trợ Người Khiếm Thị — Mô Tả Hình Ảnh Bằng Tiếng Việt (Gradio Prototype)

| Thông tin | Chi tiết |
| :---- | :---- |
| **Loại dự án** | AI Hỗ trợ Tiếp cận / Tác động xã hội |
| **Đối tượng** | Người khiếm thị và người thị lực kém tại Việt Nam |
| **Công cụ chính** | Gradio, Transformers, Moondream2 (0.5B & 2B), Helsinki-NLP, deep-translator, gTTS, pyttsx3 |
| **Môi trường chạy** | CPU Laptop thông thường, hỗ trợ deploy Hugging Face Spaces |
| **Ngày lập spec** | 2026-06-04 |

---

## 1. Mục tiêu dự án
Xây dựng một hệ thống hỗ trợ tiếp cận (Accessibility System) chạy trên giao diện Web (Gradio), cho phép tiếp nhận hình ảnh từ webcam hoặc file tải lên, chạy mô hình ngôn ngữ thị giác (VLM) cục bộ để mô tả hình ảnh, dịch sang Tiếng Việt và phát ra giọng đọc. 

Hệ thống hỗ trợ cả chế độ Online (độ chính xác cao, giọng đọc tự nhiên) và chế độ Offline hoàn toàn (sử dụng các mô hình dịch thuật và TTS chạy cục bộ làm fallback).

---

## 2. Kiến trúc Hệ thống & Luồng dữ liệu

Dự án được xây dựng dựa trên cấu trúc Pipeline tuần tự gồm 4 công đoạn (Stages):

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Stage 1:    │     │  Stage 2:    │     │  Stage 3:    │     │  Stage 4:    │
│ Preprocess   ├────>│  VLM (VLM)   ├────>│ Translation  ├────>│ Text-To-Sp.  │
│  (PIL RGB)   │     │(Moondream2)  │     │  (EN -> VI)  │     │    (TTS)     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.1. Quản lý mô hình & Công cụ Dịch thuật (Registry)
Sử dụng Singleton Pattern thông qua class `ModelRegistry` để quản lý việc khởi tạo và chia sẻ tài nguyên trong RAM, tránh rò rỉ bộ nhớ hoặc rò rỉ connection pool.

* **Sử dụng thư viện `transformers` ổn định**: 
  * Cả hai mô hình VLM đều được load qua lớp `transformers` tiêu chuẩn.
  * **Moondream2 (2B)**: Sử dụng ID `vikhyatk/moondream2` với `revision="2025-01-09"`.
  * **Moondream2 (0.5B)**: Sử dụng ID cộng đồng được chuẩn hóa sang Transformers là `andito/moondream-05`.
* **Helsinki-NLP**: Tải mô hình offline `Helsinki-NLP/opus-mt-en-vi` phục vụ dịch offline.
* **Translator & TTS Instance**: Khởi tạo duy nhất một luồng kết nối dịch thuật toàn cục để tránh leak connection HTTP.

### 2.2. Xử lý đồng thời & Xếp hàng (Queuing & Concurrency)
Sử dụng cơ chế `queue()` tích hợp sẵn của Gradio để quản lý các request đồng thời. Khi nhiều người dùng cùng bấm "Run", các request sẽ được xếp vào hàng đợi và xử lý tuần tự trên CPU, tránh làm quá tải CPU máy chủ (đặc biệt khi chạy trên Hugging Face Spaces free tier với 2 vCPUs).

---

## 3. Thiết kế chi tiết các Module & Xử lý lỗi (Error Handling)

### Module 1: Tiền xử lý ảnh (Image Preprocessing)
* **Xử lý**: 
  * Đảm bảo định dạng ảnh là RGB (loại bỏ kênh alpha hoặc chuyển ảnh grayscale sang RGB).
  * Cắt ảnh trung tâm (Center Crop) về tối đa 1280px nếu ảnh quá lớn.
* **Xử lý lỗi**:
  * Nếu file ảnh tải lên bị lỗi (corrupted) hoặc không thể đọc bởi PIL, hệ thống sẽ ném ra ngoại lệ rõ ràng, báo lỗi trên giao diện Gradio thông qua `gr.Warning` và dừng tiến trình thay vì crash backend.

### Module 2: Vision-Language Model (VLM Inference)
* **Các mô hình hỗ trợ**:
  * **Moondream2 (2B) [Mặc định]**: Chất lượng cao (~2.5 GB RAM), chạy ~3-6s trên CPU.
  * **Moondream2 (0.5B)**: Phiên bản siêu nhẹ (~900 MB RAM), chạy cực nhanh ~1-3s trên CPU.
* **Xử lý khả năng tương thích ngược của API (API Compatibility)**:
  * Do Moondream2 giữa các phiên bản có sự thay đổi API, code inference sẽ tự động phát hiện phương thức gọi:
    * Ưu tiên API mới: `model.query(image, prompt)["answer"]`.
    * Fallback nếu mô hình cũ không hỗ trợ: `model.answer_question(model.encode_image(image), prompt, processor)`.
* **Xử lý lỗi**:
  * Hàm được bọc trong block `try-except`. Nếu gặp lỗi tràn bộ nhớ (OOM) hoặc timeout, tiến trình sẽ báo lỗi cụ thể lên màn hình và giải phóng bộ nhớ.

### Module 3: Dịch thuật (Translation - EN → VI)
* **Online**: Sử dụng thư viện `deep-translator` (`GoogleTranslator`) - wrapper API ổn định hơn và ít bị rate-limit.
* **Offline**: Sử dụng mô hình `Helsinki-NLP/opus-mt-en-vi`.
* **Xử lý lỗi & Tự động Fallback**:
  * Nếu chọn chế độ **Online** nhưng gặp lỗi mạng hoặc bị Google block request, hệ thống sẽ **tự động chuyển sang dịch Offline** bằng Helsinki-NLP, đồng thời hiển thị thông báo cảnh báo màu vàng nhẹ trên UI để người dùng biết.

### Module 4: Text-To-Speech (TTS)
* **Quy định ghi file**: Do triển khai trên môi trường giới hạn như Hugging Face Spaces có filesystem read-only (chỉ `/tmp` cho phép ghi), mọi file âm thanh tạo ra phải được ghi vào thư mục tạm của hệ thống (`tempfile.gettempdir()`).
* **Online**: Sử dụng thư viện `gTTS` để tạo file giọng nói tự nhiên dạng `.mp3`.
* **Offline**: Sử dụng thư viện `pyttsx3` để tổng hợp giọng nói offline trực tiếp.
* **Xử lý lỗi & Tự động Fallback**:
  * Nếu chọn chế độ **Online** nhưng lỗi kết nối mạng tới Google TTS, hệ thống sẽ **tự động chuyển sang pyttsx3** để đọc offline.

---

## 4. Thiết kế Giao diện người dùng (Gradio UI)

### 4.1. Layout chính

* **Cột điều khiển trái (Input & Control Panel)**:
  * **Đầu vào ảnh**: `gr.Image(sources=["webcam", "upload"], type="pil", label="Đầu vào hình ảnh")`
  * **Chọn mô hình VLM**: `gr.Radio(choices=["Moondream2 (2B)", "Moondream2 (0.5B)"], value="Moondream2 (2B)", label="Mô hình VLM")`
  * **Hàng nút bấm**:
    * Nút **Run** (màu primary) để bắt đầu pipeline.
    * Nút **Cancel** (màu stop) để ngắt tiến trình.
* **Cột kết quả phải (Outputs Panel)**:
  * Khung mô tả tiếng Anh: `gr.Textbox(label="Mô tả Tiếng Anh (VLM Output)")`
  * Khung mô tả tiếng Việt: `gr.Textbox(label="Mô tả Tiếng Việt (Dịch)")`
  * Trình phát âm thanh: `gr.Audio(label="Giọng đọc Tiếng Việt", autoplay=True)`
* **Bảng Cấu hình nâng cao (Accordion - Parameters & Thresholds)**:
  * Lựa chọn phương thức dịch: `gr.Dropdown(choices=["Auto-Detect (Online)", "Offline (Helsinki-NLP)"], value="Auto-Detect (Online)")`
  * Lựa chọn phương thức TTS: `gr.Dropdown(choices=["Auto-Detect (Online)", "Offline (pyttsx3)"], value="Auto-Detect (Online)")`
  * Custom Prompt cho VLM: `gr.Textbox(lines=2, label="VLM Prompt Template")`
* **Dashboard Hiệu năng (Performance Dashboard)**:
  * Sử dụng một `gr.Group` làm container gom nhóm các thành phần:
    * **Tổng thời gian chạy (Total Time)**
    * **Dung lượng RAM tiêu thụ (RAM Usage)** sử dụng thư viện `psutil`.
  * Sử dụng một widget `gr.HTML` để vẽ biểu đồ thanh tiến trình bằng HTML/CSS đơn giản hiển thị chi tiết thời gian xử lý của từng bước (Preprocess, VLM, Translation, TTS).

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
├── __init__.py             # Đảm bảo import module chính xác
├── app.py                  # Mã nguồn chính chạy giao diện Gradio
├── registry.py             # ModelRegistry (Singleton quản lý mô hình & connections)
├── pipeline/
│   ├── __init__.py
│   ├── preprocess.py       # Module 1: Xử lý kích thước/định dạng ảnh
│   ├── vision_model.py     # Module 2: Load & Inference Moondream2 2B & 0.5B
│   ├── translate.py        # Module 3: Dịch tiếng Anh sang Việt (với fallback)
│   └── tts.py              # Module 4: Tạo giọng đọc từ text (với fallback)
└── utils/
    ├── __init__.py
    └── monitor.py          # Đo RAM tiêu thụ và đo đạc thời gian
```

### 5.2. Các kịch bản kiểm thử (Test Cases)
1. **Kiểm thử Webcam & Upload**: Kiểm tra xem Gradio có nhận ảnh chuẩn từ cả webcam và file JPG/PNG không.
2. **Kiểm thử Đổi phiên bản VLM**: Chọn đổi từ Moondream2 (0.5B) sang Moondream2 (2B), kiểm tra RAM tăng lên và thời gian xử lý thay đổi tương ứng.
3. **Kiểm thử Offline Fallback**: Ngắt kết nối mạng Internet, kiểm tra hệ thống có tự động chuyển sang dịch bằng Helsinki-NLP và đọc bằng pyttsx3 không.
4. **Kiểm thử Cancel**: Bấm "Run" với Moondream2 (2B) (thời gian chạy ~5s), sau đó bấm "Cancel" ở giây thứ 2. Kiểm tra xem tiến trình có dừng ngay lập tức và giao diện sẵn sàng cho lượt chạy tiếp theo không.
5. **Kiểm thử Đồng thời (Concurrency)**: Mở 2 tab trình duyệt cùng lúc bấm Run. Kiểm tra xem Gradio queue có xếp hàng tuần tự và không gây quá tải CPU hay crash app không.
