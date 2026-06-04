# BẢN ĐẶC TẢ THIẾT KẾ HỆ THỐNG

**AI Hỗ Trợ Người Khiếm Thị — Mô Tả Hình Ảnh Bằng Tiếng Việt**

| Thông tin | Chi tiết |
| :---- | :---- |
| **Loại dự án** | Chuyển đổi số / AI — Tác động xã hội |
| **Đối tượng** | Người khiếm thị và người thị lực kém tại Việt Nam |
| **Mục tiêu demo** | Prototype Web App — trình bày trên màn hình lớn |
| **Thời gian triển khai** | 7 ngày |
| **Thành phần nộp** | GitHub repo + Web Demo (Streamlit) + Bản đặc tả |

---

# 1. Mô tả bài toán

## 1.1. Bối cảnh

Người khiếm thị Việt Nam hiện đang phụ thuộc hoàn toàn vào sự hỗ trợ của người khác để tiếp cận thông tin thị giác từ môi trường xung quanh — đọc biển hiệu, nhận biết vật dụng, hiểu nội dung tài liệu. Các giải pháp hiện có như **Google Lookout** hay **Microsoft Seeing AI** đều ưu tiên tiếng Anh, yêu cầu kết nối Internet liên tục, và không tối ưu cho ngữ cảnh Việt Nam.

Sự ra đời của các Vision-Language Model (VLM) nhỏ gọn như **Moondream 2** (0.5B–2B tham số) mở ra cơ hội triển khai trí tuệ nhân tạo ngay trên thiết bị phổ thông — không cần GPU, không cần Cloud đắt tiền — với khả năng mô tả hình ảnh chuyên sâu và chính xác.

## 1.2. Định nghĩa bài toán

> **Xây dựng hệ thống hỗ trợ tiếp cận (Accessibility System) chạy cục bộ trên CPU, có khả năng tiếp nhận hình ảnh từ camera hoặc file ảnh, tự động phân tích nội dung và phát ra mô tả bằng giọng nói tiếng Việt, giúp người khiếm thị hiểu được môi trường xung quanh mà không cần nhờ người khác.**

Về mặt kỹ thuật, đây là bài toán kết hợp chuỗi pipeline:

```
Image → [Moondream 2] → English Text → [Translation] → Vietnamese Text → [TTS] → Audio
```

## 1.3. Đặc điểm bài toán

- **Đầu vào:** Hình ảnh JPEG/PNG chụp từ webcam hoặc tải lên từ giao diện
- **Đầu ra:** Giọng đọc tiếng Việt mô tả nội dung ảnh + hiển thị text trên giao diện
- **Ràng buộc:** Chạy được trên CPU laptop thông thường, không yêu cầu GPU
- **Mục tiêu latency:** Tổng thời gian từ chụp ảnh đến phát audio < 7 giây

## 1.4. Phạm vi (Scope) — Phiên bản demo

| Tính năng | Trạng thái |
| :---- | :---- |
| Mô tả môi trường xung quanh từ ảnh tĩnh | ✅ Có trong demo |
| Chụp ảnh trực tiếp từ webcam | ✅ Có trong demo |
| Phát âm thanh tiếng Việt | ✅ Có trong demo |
| Giao diện Web (Streamlit) | ✅ Có trong demo |
| Đọc text trong ảnh (OCR) | 🔜 Roadmap v2 |
| Triển khai Mobile (Android/iOS) | 🔜 Roadmap v3 |
| Hỗ trợ TalkBack / VoiceOver | 🔜 Roadmap v3 |

---

# 2. Yêu cầu hệ thống

## 2.1. Chức năng bắt buộc (Functional Requirements)

1. Nhận đầu vào là hình ảnh từ webcam hoặc file upload
2. Phân tích nội dung ảnh và sinh mô tả bằng tiếng Anh (Moondream 2)
3. Dịch mô tả sang tiếng Việt tự nhiên, đúng ngữ cảnh
4. Chuyển văn bản tiếng Việt thành giọng đọc (TTS)
5. Hiển thị kết quả text + phát audio trên giao diện web
6. Chạy hoàn toàn offline (ngoại trừ module dịch — có fallback)

## 2.2. Yêu cầu phi chức năng (Non-functional Requirements)

- **Latency:** < 7 giây / ảnh trên CPU laptop thông thường
- **Dung lượng model:** < 1 GB (dùng bản quantized INT4)
- **RAM tối thiểu:** 4 GB (model 0.5B INT4 cần ~816 MB)
- **Tương thích:** Python 3.9+, Windows/Linux/macOS

## 2.3. Giao diện demo

Giao diện Streamlit với các thành phần:
- Nút lớn **"Chụp ảnh"** (hỗ trợ phím tắt `Space`)
- Vùng upload file ảnh thay thế
- Khung hiển thị ảnh vừa chụp
- Khung text hiển thị mô tả tiếng Việt
- Audio player tự động phát sau khi xử lý xong
- Thanh trạng thái hiển thị tiến trình xử lý

---

# 3. Hướng tiếp cận — Option C (Prototype → Roadmap)

## 3.1. Lý do chọn Option C

Sau quá trình phân tích kỹ thuật và yêu cầu thực tế:

| Tiêu chí | Option A (Mobile App) | Option B (Web App thuần) | **Option C (Web + Roadmap)** |
| :---- | :---- | :---- | :---- |
| Khả thi trong 7 ngày | ❌ Khó | ✅ Có | ✅ **Có** |
| Ấn tượng khi demo màn hình lớn | ⚠️ Khó show | ✅ Tốt | ✅ **Tốt** |
| Người khiếm thị tự dùng ngay | ✅ Được | ❌ Không | ⚠️ **Prototype** |
| Điểm tác động xã hội | ✅ Cao | ⚠️ Trung bình | ✅ **Cao (nếu pitch đúng)** |

**Option C** là cân bằng tối ưu: xây dựng **prototype kỹ thuật hoàn chỉnh** trên web, đồng thời trình bày rõ **roadmap triển khai thực tế** cho người dùng cuối — điều này thuyết phục ban giám khảo hơn việc chỉ có một app mobile sơ sài hoặc chỉ có demo lab không thực tế.

## 3.2. Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    GIAO DIỆN STREAMLIT                       │
│  [Nút Chụp / Upload]  →  [Hiển thị ảnh]  →  [Text + Audio] │
└────────────────────┬────────────────────────────────────────┘
                     │ PIL.Image object
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              MODULE 1: TIỀN XỬ LÝ ẢNH                       │
│  - Chuyển BGR → RGB (nếu từ webcam OpenCV)                  │
│  - Kiểm tra kích thước tối thiểu (>= 224px)                 │
│  - Convert sang RGB mode (loại RGBA, grayscale)             │
│  - Center crop nếu ảnh > 1280px                             │
└────────────────────┬────────────────────────────────────────┘
                     │ PIL.Image (RGB, hợp lệ)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            MODULE 2: VISION-LANGUAGE MODEL                   │
│  Model: Moondream 0.5B INT4 (CPU inference)                 │
│  Prompt: "Describe this image briefly and clearly           │
│           for a visually impaired person."                  │
│  Output: English description string                         │
│  Latency: ~3–5 giây trên CPU laptop                        │
└────────────────────┬────────────────────────────────────────┘
                     │ English text
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              MODULE 3: DỊCH THUẬT (EN → VI)                  │
│  Primary:  Google Translate API (cần internet)              │
│  Fallback: Helsinki-NLP/opus-mt-en-vi (offline, ~300MB)     │
│  Latency: < 1 giây                                          │
└────────────────────┬────────────────────────────────────────┘
                     │ Vietnamese text
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              MODULE 4: TEXT-TO-SPEECH (TTS)                  │
│  Primary:  gTTS — Google Text-to-Speech (cần internet)      │
│  Fallback: pyttsx3 (offline, giọng cơ bản)                  │
│  Output: .mp3 file → tự động phát trên giao diện            │
│  Latency: < 2 giây                                          │
└────────────────────────────────────────────────────────────┘
```

**Tổng latency ước tính:** 5–8 giây / ảnh trên CPU laptop thông thường

## 3.3. Giải thích chi tiết từng Module

### Module 1 — Tiền xử lý ảnh

Moondream 2 dùng **SigLIP** làm visual encoder — kiến trúc Vision Transformer hoạt động theo cơ chế patch. Model tự động resize ảnh về 384×384 nội bộ, nhưng cần đảm bảo:

- Ảnh đủ sáng, không mờ, kích thước tối thiểu 224×224 px
- Format đúng RGB (không phải RGBA hay grayscale)
- Không quá lớn (>1280px) vì sau khi resize xuống 384px, chi tiết nhỏ sẽ mất

```python
def preprocess_image(image_input):
    img = Image.open(image_input).convert("RGB")
    w, h = img.size
    if w < 224 or h < 224:
        st.warning("⚠️ Ảnh quá nhỏ, chất lượng mô tả có thể kém")
    if w > 1280 or h > 1280:
        img = center_crop(img, 1280)
    return img
```

### Module 2 — Vision-Language Model (Moondream 2)

**Tại sao chọn Moondream 2:**
- Tiny VLM mã nguồn mở, miễn phí thương mại
- Bản 0.5B INT4 chỉ cần ~816 MB RAM, chạy được trên CPU
- Latency ~3–5 giây/ảnh trên Raspberry Pi 5 (laptop CPU nhanh hơn)
- API đơn giản: chỉ cần `.caption(image)` hoặc `.query(image, prompt)`

**Prompt được tối ưu:**
```
"Describe what you see in this image briefly and clearly.
 Focus on the main subject, people, objects, and any
 important context. Keep it under 3 sentences."
```

**Lưu ý kỹ thuật:**
- Moondream không xử lý batch — chỉ 1 ảnh / lần inference
- Dùng `revision` cố định để đảm bảo reproducibility
- Load model 1 lần duy nhất khi khởi động app (cache với `@st.cache_resource`)

### Module 3 — Dịch thuật EN → VI

**Vấn đề:** Moondream chỉ sinh output tiếng Anh. Dịch sai ngữ cảnh = toàn bộ pipeline hỏng.

**Chiến lược 2 tầng:**

| Tình huống | Giải pháp | Chất lượng |
| :---- | :---- | :---- |
| Có internet | Google Translate API (free tier) | ⭐⭐⭐⭐⭐ |
| Không internet | `Helsinki-NLP/opus-mt-en-vi` (HuggingFace) | ⭐⭐⭐ |

**Quan trọng:** Cần kiểm tra chất lượng dịch thủ công với các prompt Moondream thường sinh ra — đặc biệt với từ ngữ mô tả không gian, đồ vật trong nhà.

### Module 4 — Text-to-Speech Tiếng Việt

**Tiêu chí chọn TTS:** Giọng đọc tự nhiên, rõ ràng, không bị ngắt câu bất thường.

| Giải pháp | Ưu điểm | Nhược điểm |
| :---- | :---- | :---- |
| gTTS (Google) | Giọng tự nhiên, miễn phí | Cần internet |
| Viettel TTS API | Giọng VN chuẩn nhất | Cần đăng ký API key |
| pyttsx3 | Hoàn toàn offline | Giọng robot, kém tự nhiên |
| Zalo TTS | Giọng VN tốt | Giới hạn request |

**Lựa chọn cho demo:** gTTS (primary) + pyttsx3 (offline fallback)

## 3.4. Kế hoạch 7 ngày

| Ngày | Công việc | Deliverable |
| :---- | :---- | :---- |
| **Ngày 1** | Setup môi trường, cài Moondream 0.5B INT4, chạy inference thử với ảnh mẫu | Notebook test inference OK |
| **Ngày 2** | Xây Module 1 (tiền xử lý) + Module 2 (Moondream), test với nhiều loại ảnh | Script pipeline cơ bản |
| **Ngày 3** | Tích hợp Module 3 (dịch) + Module 4 (TTS), test end-to-end | Pipeline hoàn chỉnh CLI |
| **Ngày 4** | Xây giao diện Streamlit, tích hợp webcam | Web app v1 chạy được |
| **Ngày 5** | Tối ưu latency, xử lý edge cases, test nhiều tình huống thực tế | Web app v2 ổn định |
| **Ngày 6** | Thu thập ảnh demo thực tế (phòng học, đường phố, menu nhà hàng...), chuẩn bị slide | Demo set + Slide thuyết trình |
| **Ngày 7** | Buffer — sửa bug, rehearsal demo, hoàn thiện tài liệu | Final demo ready |

## 3.5. Cấu trúc thư mục dự án

```
project/
├── app.py                  # Streamlit main app
├── pipeline/
│   ├── __init__.py
│   ├── preprocess.py       # Module 1: tiền xử lý ảnh
│   ├── vision_model.py     # Module 2: Moondream inference
│   ├── translate.py        # Module 3: EN→VI translation
│   └── tts.py              # Module 4: Text-to-Speech
├── utils/
│   ├── camera.py           # Webcam capture helper
│   └── audio.py            # Audio playback helper
├── assets/
│   └── demo_images/        # Ảnh mẫu cho demo
├── requirements.txt
└── README.md
```

---

# 4. Đánh giá ưu/nhược điểm

## 4.1. Ưu điểm

- **Zero Cloud dependency** cho module chính (Moondream chạy local hoàn toàn)
- **Mã nguồn mở** — toàn bộ tech stack miễn phí, không có vendor lock-in
- **Latency chấp nhận được** — 5–8 giây phù hợp với use case không cần real-time
- **Dễ demo** — giao diện web đơn giản, trực quan cho khán giả không chuyên kỹ thuật
- **Tính mới ở Việt Nam** — chưa có sản phẩm tương tự tối ưu cho tiếng Việt

## 4.2. Nhược điểm & Hạn chế hiện tại

| Hạn chế | Mức độ ảnh hưởng | Giải pháp đề xuất |
| :---- | :---- | :---- |
| Moondream 0.5B đôi khi mô tả thiếu ngữ cảnh phức tạp | Trung bình | Nâng lên 2B nếu hardware cho phép |
| Dịch thuật EN→VI đôi khi mất sắc thái | Cao | Fine-tune prompt + post-processing từ điển |
| gTTS cần internet | Thấp (có fallback) | Tích hợp Viettel TTS API offline |
| Chưa hỗ trợ TalkBack/VoiceOver natively | Cao với user thật | Roadmap Mobile v3 |
| Không xử lý text trong ảnh (biển hiệu, menu) | Trung bình | Roadmap OCR v2 |

---

# 5. Roadmap phát triển

## v1.0 — Demo hiện tại (Tuần 1)

- Web prototype trên Streamlit
- Pipeline: Moondream → Dịch → TTS tiếng Việt
- Chạy trên laptop CPU

## v2.0 — Mở rộng tính năng (Tháng 1–2)

- Thêm OCR: nhận dạng và đọc text trong ảnh (biển hiệu, menu, hóa đơn)
- Thêm chế độ "Nhận dạng tiền Việt Nam" (bài toán cực thực tế cho người khiếm thị)
- Tích hợp Viettel TTS / Zalo TTS cho giọng đọc tự nhiên hơn
- Tối ưu latency xuống < 3 giây

## v3.0 — Mobile & Accessibility (Tháng 3–6)

- Chuyển sang React Native hoặc Flutter app
- Tích hợp đầy đủ với TalkBack (Android) và VoiceOver (iOS)
- Gesture: double tap = chụp ảnh, không cần nhìn màn hình
- Thử nghiệm thực tế với người dùng khiếm thị tại Hội người mù Việt Nam

---

# 6. Kết quả benchmark sơ bộ

*(Đo trên laptop CPU Intel Core i5, RAM 8GB)*

| Metric | Moondream 0.5B INT4 | Moondream 2B INT4 |
| :---- | :---- | :---- |
| Thời gian inference | ~3–4 giây | ~6–8 giây |
| RAM sử dụng | ~900 MB | ~2.5 GB |
| Chất lượng mô tả đơn giản | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Chất lượng mô tả phức tạp | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Phù hợp demo | ✅ Recommended | ⚠️ Nếu RAM đủ |

---

# 7. Tài liệu tham khảo

| Tài liệu | Link |
| :---- | :---- |
| Moondream 2 — GitHub chính thức | https://github.com/vikhyat/moondream |
| Moondream 2 — HuggingFace model card | https://huggingface.co/vikhyatk/moondream2 |
| Moondream 0.5B announcement | https://moondream.ai/blog/introducing-moondream-0-5b |
| SigLIP paper (visual encoder) | https://arxiv.org/abs/2303.15343 |
| LLaVA paper (VLM architecture reference) | https://arxiv.org/abs/2304.08485 |
| Helsinki-NLP EN→VI model | https://huggingface.co/Helsinki-NLP/opus-mt-en-vi |
| Survey on Small VLMs for Edge (2025) | https://arxiv.org/abs/2504.09724 |
