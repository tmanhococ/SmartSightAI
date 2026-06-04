# BẢN ĐẶC TẢ THIẾT KẾ HỆ THỐNG (CẬP NHẬT)
## AI Hỗ Trợ Người Khiếm Thị — Mô Tả Hình Ảnh Bằng Tiếng Việt (Gradio Prototype)

| Thông tin | Chi tiết |
| :---- | :---- |
| **Loại dự án** | AI Hỗ trợ Tiếp cận / Tác động xã hội |
| **Đối tượng** | Người khiếm thị và người thị lực kém tại Việt Nam |
| **Công cụ chính** | Gradio, Transformers, Moondream2, BLIP-Base, Helsinki-NLP, gTTS, pyttsx3 |
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
│  (PIL RGB)   │     │(Moondream/   │     │  (EN -> VI)  │     │    (TTS)     │
│              │     │  BLIP-Base)  │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.1. Quản lý mô hình (Model Registry)
Sử dụng Singleton Pattern thông qua class `ModelRegistry` để quản lý việc khởi tạo và chia sẻ tài nguyên trong RAM, tránh rò rỉ bộ nhớ khi chạy đa phiên làm việc trên Gradio.

* **Sử dụng thư viện `transformers` ổn định**: 
  * Cả hai mô hình VLM đều được load qua lớp `transformers` tiêu chuẩn thay vì thư viện `moondream` chính thức nhằm tránh breaking changes trong cuộc thi.
  * **Moondream2 (1.6B/2B)**: Sử dụng ID `vikhyatk/moondream2` với `trust_remote_code=True`.
  * **BLIP-Base (220M)**: Thay thế cho tùy chọn "0.5B" để so sánh và làm baseline siêu nhẹ chạy trên CPU. Sử dụng ID `Salesforce/blip-image-captioning-base`.
* **Helsinki-NLP**: Tải mô hình offline `Helsinki-NLP/opus-mt-en-vi` phục vụ dịch offline.

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
  * **BLIP-Base (220M)**: Rất nhẹ (~450 MB RAM), tốc độ tính toán cực nhanh (<1s trên CPU).
  * **Moondream2 (1.6B/2B)**: Chất lượng cao (~2.5 GB RAM), chạy ~3-6s trên CPU.
* **Xử lý lỗi**:
  * Do quá trình chạy VLM trên CPU tốn nhiều tài nguyên nhất, hàm được bọc trong block `try-except`. Nếu gặp lỗi tràn bộ nhớ (OOM) hoặc timeout, tiến trình sẽ báo lỗi cụ thể lên màn hình và giải phóng bộ nhớ.

### Module 3: Dịch thuật (Translation - EN → VI)
* **Online**: Sử dụng thư viện `googletrans==4.0.0rc1` (wrapper API ổn định cho demo).
* **Offline**: Sử dụng mô hình `Helsinki-NLP/opus-mt-en-vi`.
* **Xử lý lỗi & Tự động Fallback**:
  * Nếu chọn chế độ **Online** nhưng gặp lỗi mạng hoặc bị Google block request, hệ thống sẽ **tự động chuyển sang dịch Offline** bằng Helsinki-NLP, đồng thời hiển thị thông báo cảnh báo màu vàng nhẹ trên UI để người dùng biết.

### Module 4: Text-To-Speech (TTS)
* **Online**: Sử dụng thư viện `gTTS` để tạo file giọng nói tự nhiên dạng `.mp3`.
* **Offline**: Sử dụng thư viện `pyttsx3` để tổng hợp giọng nói offline trực tiếp.
* **Xử lý lỗi & Tự động Fallback**:
  * Nếu chọn chế độ **Online** nhưng lỗi kết nối mạng tới Google TTS, hệ thống sẽ **tự động chuyển sang pyttsx3** để đọc offline, đảm bảo luôn luôn có đầu ra giọng nói cho người khiếm thị.

---

## 4. Thiết kế Giao diện người dùng (Gradio UI)

### 4.1. Layout chính

* **Cột điều khiển trái (Input & Control Panel)**:
  * **Đầu vào ảnh**: `gr.Image(sources=["webcam", "upload"], type="pil", label="Đầu vào hình ảnh")`
  * **Chọn mô hình VLM**: `gr.Radio(choices=["BLIP-Base (220M)", "Moondream2 (2B)"], value="BLIP-Base (220M)", label="Mô hình VLM")`
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
  * Custom Prompt cho VLM: `gr.Textbox(lines=2, label="VLM Prompt Template (chỉ áp dụng cho Moondream2)")`
* **Dashboard Hiệu năng (Performance Dashboard)**:
  * Sử dụng các widget `gr.Textbox` để hiển thị:
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
├── app.py                  # Mã nguồn chính chạy giao diện Gradio
├── registry.py             # ModelRegistry (Singleton quản lý mô hình)
├── pipeline/
│   ├── __init__.py
│   ├── preprocess.py       # Module 1: Xử lý kích thước/định dạng ảnh
│   ├── vision_model.py     # Module 2: Load & Inference Moondream2 & BLIP-Base
│   ├── translate.py        # Module 3: Dịch tiếng Anh sang Việt (với fallback)
│   └── tts.py              # Module 4: Tạo giọng đọc từ text (với fallback)
└── utils/
    └── monitor.py          # Đo RAM tiêu thụ và đo đạc thời gian
```

### 5.2. Các kịch bản kiểm thử (Test Cases)
1. **Kiểm thử Webcam & Upload**: Kiểm tra xem Gradio có nhận ảnh chuẩn từ cả webcam và file JPG/PNG không.
2. **Kiểm thử Đổi phiên bản VLM**: Chọn đổi từ BLIP-Base sang Moondream2, kiểm tra RAM tăng lên và thời gian xử lý thay đổi tương ứng.
3. **Kiểm thử Offline Fallback**: Ngắt kết nối mạng Internet, kiểm tra hệ thống có tự động chuyển sang dịch bằng Helsinki-NLP và đọc bằng pyttsx3 không.
4. **Kiểm thử Cancel**: Bấm "Run" với Moondream2 (thời gian chạy ~5s), sau đó bấm "Cancel" ở giây thứ 2. Kiểm tra xem tiến trình có dừng ngay lập tức và giao diện sẵn sàng cho lượt chạy tiếp theo không.
5. **Kiểm thử Đồng thời (Concurrency)**: Mở 2 tab trình duyệt cùng lúc bấm Run. Kiểm tra xem Gradio queue có xếp hàng tuần tự và không gây quá tải CPU hay crash app không.
