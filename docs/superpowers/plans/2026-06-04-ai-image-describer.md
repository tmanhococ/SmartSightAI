# AI Image Describer Implementation Plan (Cập nhật 2 phiên bản Moondream2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng ứng dụng Gradio mô tả hình ảnh bằng tiếng Việt chạy cục bộ, hỗ trợ webcam, đổi phiên bản VLM (Moondream2 2B vs Moondream2 0.5B), dịch thuật (sử dụng deep-translator) và phát audio tiếng Việt (lưu file trong thư mục /tmp) với cơ chế tự động chuyển đổi sang chế độ offline khi mất mạng.

**Architecture:** Sử dụng kiến trúc Pipeline chia thành các module chức năng độc lập (preprocess, vision_model, translate, tts). Quản lý mô hình qua class ModelRegistry dạng Singleton, đo hiệu năng bằng psutil, giao diện Gradio xử lý queue và nút Cancel bản địa.

**Tech Stack:** Python 3.9+, Gradio, PyTorch, Transformers, deep-translator, gTTS, pyttsx3, psutil, pytest, Pillow.

---

### Task 1: Environment, Requirements, and Packages Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/pipeline/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write requirements configuration**

Create: `requirements.txt`
```text
gradio>=4.0.0
transformers>=4.40.0
torch>=2.0.0
pillow>=9.0.0
deep-translator>=1.11.0
gTTS>=2.3.0
pyttsx3>=2.90
psutil>=5.9.0
sentencepiece>=0.1.99
sacremoses>=0.0.53
pytest>=7.0.0
```

- [ ] **Step 2: Create empty package initialization files**

Create empty files at:
- `src/__init__.py`
- `src/pipeline/__init__.py`
- `src/utils/__init__.py`
- `tests/__init__.py`

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Cài đặt thành công toàn bộ các thư viện và không có xung đột package.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/__init__.py src/pipeline/__init__.py src/utils/__init__.py tests/__init__.py
git commit -m "chore: setup environment and python packages"
```

---

### Task 2: Performance Monitoring Utility

**Files:**
- Create: `src/utils/monitor.py`
- Test: `tests/test_monitor.py`

- [ ] **Step 1: Write the failing test**

Create: `tests/test_monitor.py`
```python
import time
from src.utils.monitor import ExecutionMonitor

def test_execution_monitor():
    monitor = ExecutionMonitor()
    assert monitor.get_ram_usage() > 0.0
    
    with monitor.track("test_stage"):
        time.sleep(0.1)
        
    durations = monitor.get_durations()
    assert "test_stage" in durations
    assert durations["test_stage"] >= 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor.py -v`
Expected: FAIL due to missing files/modules.

- [ ] **Step 3: Write minimal implementation**

Create: `src/utils/monitor.py`
```python
import psutil
import time
from contextlib import contextmanager

class ExecutionMonitor:
    def __init__(self):
        self.durations = {}
        
    def get_ram_usage(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # MB
        
    @contextmanager
    def track(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[stage_name] = time.perf_counter() - start
            
    def get_durations(self) -> dict:
        return self.durations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monitor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/monitor.py tests/test_monitor.py
git commit -m "feat: add performance monitoring utility"
```

---

### Task 3: Image Preprocessing

**Files:**
- Create: `src/pipeline/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing test**

Create: `tests/test_preprocess.py`
```python
import pytest
from PIL import Image
from src.pipeline.preprocess import preprocess_image

def test_preprocess_image_valid():
    img = Image.new("RGBA", (1500, 1000), color="red")
    processed = preprocess_image(img)
    assert processed.mode == "RGB"
    assert processed.size[0] <= 1280
    assert processed.size[1] <= 1280

def test_preprocess_image_too_small():
    img = Image.new("L", (100, 100))
    with pytest.raises(ValueError, match="Image is too small"):
        preprocess_image(img)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create: `src/pipeline/preprocess.py`
```python
from PIL import Image

def preprocess_image(image: Image.Image) -> Image.Image:
    if image is None:
        raise ValueError("No image provided")
        
    w, h = image.size
    if w < 224 or h < 224:
        raise ValueError(f"Image is too small ({w}x{h}). Minimum size is 224x224 px.")
        
    # Convert RGBA or L to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Center crop / Resize if too large (> 1280px on any dimension)
    max_size = 1280
    if w > max_size or h > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    return image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat: implement image preprocessing module"
```

---

### Task 4: Translation Module with Fallback

**Files:**
- Create: `src/pipeline/translate.py`
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing test**

Create: `tests/test_translate.py`
```python
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.translate import TranslatorModule

def test_translate_offline():
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    mock_model.generate.return_value = [[4, 5, 6]]
    mock_tokenizer.decode.return_value = "Xin chào"
    
    translator = TranslatorModule(offline_model=mock_model, offline_tokenizer=mock_tokenizer)
    result, is_offline = translator.translate("Hello", mode="Offline (Helsinki-NLP)")
    
    assert result == "Xin chào"
    assert is_offline is True

@patch("src.pipeline.translate.GoogleTranslator.translate")
def test_translate_online_success(mock_google_translate):
    mock_google_translate.return_value = "Xin chào"
    
    translator = TranslatorModule()
    result, is_offline = translator.translate("Hello", mode="Auto-Detect (Online)")
    
    assert result == "Xin chào"
    assert is_offline is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_translate.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create: `src/pipeline/translate.py`
```python
import logging
from deep_translator import GoogleTranslator

class TranslatorModule:
    def __init__(self, offline_model=None, offline_tokenizer=None):
        self.offline_model = offline_model
        self.offline_tokenizer = offline_tokenizer
        self.google_translator = GoogleTranslator(source="auto", target="vi")
        
    def translate(self, text: str, mode: str = "Auto-Detect (Online)") -> tuple[str, bool]:
        if not text.strip():
            return "", False
            
        if mode == "Offline (Helsinki-NLP)":
            return self._translate_offline(text), True
            
        # Try Online translation
        try:
            translated_text = self.google_translator.translate(text)
            return translated_text, False
        except Exception as e:
            logging.warning(f"Online translation failed: {e}. Falling back to offline translation.")
            if self.offline_model and self.offline_tokenizer:
                return self._translate_offline(text), True
            else:
                return f"[Lỗi mạng - Không dịch được] {text}", False
                
    def _translate_offline(self, text: str) -> str:
        if not self.offline_model or not self.offline_tokenizer:
            return f"[Chưa load model offline] {text}"
            
        inputs = self.offline_tokenizer(text, return_tensors="pt")
        generated_ids = self.offline_model.generate(**inputs)
        translated_text = self.offline_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return translated_text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_translate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/translate.py tests/test_translate.py
git commit -m "feat: implement translator module with deep-translator"
```

---

### Task 5: TTS Module with Fallback

**Files:**
- Create: `src/pipeline/tts.py`
- Test: `tests/test_tts.py`

- [ ] **Step 1: Write the failing test**

Create: `tests/test_tts.py`
```python
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.tts import TTSModule

def test_tts_offline():
    # Mock pyttsx3 engine in the correct import scope
    with patch("src.pipeline.tts.pyttsx3.init") as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        tts = TTSModule()
        temp_dir = tempfile.gettempdir()
        target_path = os.path.join(temp_dir, "test_offline.mp3")
        
        output_path = tts.generate_speech("Xin chào", mode="Offline (pyttsx3)", filename=target_path)
        
        assert output_path == target_path
        mock_engine.save_to_file.assert_called_once_with("Xin chào", target_path)
        mock_engine.runAndWait.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create: `src/pipeline/tts.py`
```python
import os
import tempfile
import logging
from gtts import gTTS
import pyttsx3

class TTSModule:
    def __init__(self):
        self.offline_engine = None
        
    def _init_offline(self):
        if self.offline_engine is None:
            try:
                self.offline_engine = pyttsx3.init()
                voices = self.offline_engine.getProperty("voices")
                for voice in voices:
                    if voice.languages and any("vi" in lang for lang in voice.languages):
                        self.offline_engine.setProperty("voice", voice.id)
                        break
                    elif "vietnam" in voice.name.lower():
                        self.offline_engine.setProperty("voice", voice.id)
                        break
            except Exception as e:
                logging.error(f"Failed to initialize pyttsx3 offline TTS: {e}")
                
    def generate_speech(self, text: str, mode: str = "Auto-Detect (Online)", filename: str = None) -> str:
        if not text.strip():
            return None
            
        if filename is None:
            filename = os.path.join(tempfile.gettempdir(), "output.mp3")
            
        # Clean existing file to avoid permission or handle lock issues
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
                
        if mode == "Offline (pyttsx3)":
            self._generate_offline(text, filename)
            return filename
            
        # Try Online gTTS
        try:
            tts = gTTS(text=text, lang="vi", slow=False)
            tts.save(filename)
            return filename
        except Exception as e:
            logging.warning(f"Online gTTS failed: {e}. Falling back to pyttsx3.")
            try:
                self._generate_offline(text, filename)
                return filename
            except Exception as e_off:
                logging.error(f"Offline TTS failed: {e_off}")
                return None
            
    def _generate_offline(self, text: str, filename: str):
        self._init_offline()
        if self.offline_engine:
            self.offline_engine.save_to_file(text, filename)
            self.offline_engine.runAndWait()
        else:
            raise RuntimeError("Offline TTS engine pyttsx3 is not available.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/tts.py tests/test_tts.py
git commit -m "feat: implement TTS module using temp directory and proper patch scope"
```

---

### Task 6: Singleton Model Registry

**Files:**
- Create: `src/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test with fixture isolation**

Create: `tests/test_registry.py`
```python
import pytest
from src.registry import ModelRegistry

@pytest.fixture(autouse=True)
def reset_singleton():
    ModelRegistry._instance = None
    yield
    ModelRegistry._instance = None

def test_singleton_pattern():
    reg1 = ModelRegistry()
    reg2 = ModelRegistry()
    assert reg1 is reg2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create: `src/registry.py`
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
from src.pipeline.translate import TranslatorModule

class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.vlm_models = {"Moondream2 (2B)": None, "Moondream2 (0.5B)": None}
            cls._instance.vlm_processors = {"Moondream2 (2B)": None, "Moondream2 (0.5B)": None}
            cls._instance.translation_model = None
            cls._instance.translation_tokenizer = None
            cls._instance.translator_instance = None
        return cls._instance
        
    def get_vlm(self, version: str) -> tuple:
        if version not in self.vlm_models:
            raise ValueError(f"Unknown VLM model version: {version}")
            
        if self.vlm_models[version] is None:
            if version == "Moondream2 (0.5B)":
                model_id = "andito/moondream-05" # Community transformers 0.5B model
                processor = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
            else: # Moondream2 (2B)
                model_id = "vikhyatk/moondream2"
                revision = "2025-01-09" # API 2025 compatible
                processor = AutoTokenizer.from_pretrained(model_id, revision=revision)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    revision=revision, 
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
            self.vlm_models[version] = model
            self.vlm_processors[version] = processor
            
        return self.vlm_models[version], self.vlm_processors[version]
        
    def get_translator_module(self, mode: str) -> TranslatorModule:
        if self.translator_instance is None:
            model_id = "Helsinki-NLP/opus-mt-en-vi"
            self.translation_tokenizer = MarianTokenizer.from_pretrained(model_id)
            self.translation_model = MarianMTModel.from_pretrained(model_id)
            self.translator_instance = TranslatorModule(self.translation_model, self.translation_tokenizer)
        return self.translator_instance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/registry.py tests/test_registry.py
git commit -m "feat: implement singleton model registry with 2B and 0.5B Moondream2 support"
```

---

### Task 7: VLM Inference Module

**Files:**
- Create: `src/pipeline/vision_model.py`
- Test: `tests/test_vision_model.py`

- [ ] **Step 1: Write the failing test**

Create: `tests/test_vision_model.py`
```python
import pytest
from unittest.mock import MagicMock
from PIL import Image
from src.pipeline.vision_model import run_vlm_inference

def test_run_vlm_moondream_new_api_mock():
    mock_model = MagicMock()
    # Mock the new query method
    mock_model.query.return_value = {"answer": "a photo of a laptop"}
    
    img = Image.new("RGB", (224, 224))
    result = run_vlm_inference(img, "Moondream2 (2B)", mock_model, MagicMock(), prompt="What is this?")
    assert result == "a photo of a laptop"
    mock_model.query.assert_called_once_with(img, "What is this?")

def test_run_vlm_moondream_old_api_fallback_mock():
    mock_model = MagicMock()
    # Delete query method to trigger fallback
    del mock_model.query
    mock_model.encode_image.return_value = "encoded"
    mock_model.answer_question.return_value = "a photo of a laptop from fallback"
    
    img = Image.new("RGB", (224, 224))
    result = run_vlm_inference(img, "Moondream2 (0.5B)", mock_model, "processor", prompt="What is this?")
    assert result == "a photo of a laptop from fallback"
    mock_model.encode_image.assert_called_once_with(img)
    mock_model.answer_question.assert_called_once_with("encoded", "What is this?", "processor")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vision_model.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create: `src/pipeline/vision_model.py`
```python
from PIL import Image

def run_vlm_inference(image: Image.Image, version: str, model, processor, prompt: str = "") -> str:
    if version not in ["Moondream2 (2B)", "Moondream2 (0.5B)"]:
        raise ValueError(f"Unsupported model version: {version}")
        
    if not prompt.strip():
        prompt = "Describe what you see in this image briefly and clearly. Focus on the main subject, people, objects, and any important context. Keep it under 3 sentences."
        
    # Check for the newer API (model.query) or fallback to older API (encode_image / answer_question)
    if hasattr(model, "query"):
        response = model.query(image, prompt)
        return response["answer"]
    else:
        # Fallback to older Moondream version API
        enc_image = model.encode_image(image)
        return model.answer_question(enc_image, prompt, processor)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vision_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/vision_model.py tests/test_vision_model.py
git commit -m "feat: implement VLM inference with fallback compat for query and answer_question APIs"
```

---

### Task 8: Gradio User Interface

**Files:**
- Create: `src/app.py`

- [ ] **Step 1: Write full UI application with Moondream2 2B and 0.5B versions**

Create: `src/app.py`
```python
import os
import tempfile
import time
import gradio as gr
from PIL import Image
from src.registry import ModelRegistry
from src.pipeline.preprocess import preprocess_image
from src.pipeline.vision_model import run_vlm_inference
from src.pipeline.tts import TTSModule
from src.utils.monitor import ExecutionMonitor

# Global registry and TTS
registry = ModelRegistry()
tts_module = TTSModule()

def get_performance_html(durations: dict) -> str:
    total = sum(durations.values())
    if total == 0:
        return "<p>Chưa có dữ liệu hiệu năng.</p>"
        
    html = "<div style='font-family: monospace; background: #1e1e1e; padding: 10px; border-radius: 5px; color: #fff;'>"
    html += "<h4 style='margin-top:0; color:#58a6ff;'>Timing Breakdown:</h4>"
    for stage, duration in durations.items():
        pct = (duration / total) * 100 if total > 0 else 0
        bar_count = int(pct / 5)
        bar = "█" * bar_count + "░" * (20 - bar_count)
        html += f"<div style='margin-bottom: 5px;'><b>{stage.capitalize()}:</b> {duration:.3f}s <span style='color: #8b949e;'>[{bar}]</span> {pct:.1f}%</div>"
    html += "</div>"
    return html

def run_pipeline(image, vlm_version, translate_mode, tts_mode, custom_prompt):
    monitor = ExecutionMonitor()
    
    if image is None:
        raise gr.Error("Vui lòng chụp ảnh hoặc tải ảnh lên trước!")
        
    # Preprocessing
    with monitor.track("preprocess"):
        try:
            img = preprocess_image(image)
        except Exception as e:
            raise gr.Error(f"Lỗi xử lý ảnh: {str(e)}")
            
    # Load VLM and Inference
    with monitor.track("vlm_inference"):
        try:
            vlm_model, vlm_processor = registry.get_vlm(vlm_version)
            eng_desc = run_vlm_inference(img, vlm_version, vlm_model, vlm_processor, custom_prompt)
        except Exception as e:
            raise gr.Error(f"Lỗi VLM Inference: {str(e)}")
            
    # Load Translator and Translate
    with monitor.track("translation"):
        try:
            # Load TranslatorModule using Singleton registry
            translator = registry.get_translator_module(translate_mode)
            vi_desc, is_offline_trans = translator.translate(eng_desc, translate_mode)
        except Exception as e:
            vi_desc = f"[Lỗi dịch] {eng_desc}"
            is_offline_trans = False
            gr.Warning(f"Dịch thuật thất bại: {str(e)}")
            
    # TTS with temp directory destination
    with monitor.track("tts"):
        try:
            temp_path = os.path.join(tempfile.gettempdir(), "output.mp3")
            audio_path = tts_module.generate_speech(vi_desc, tts_mode, filename=temp_path)
        except Exception as e:
            audio_path = None
            gr.Warning(f"Không thể tạo giọng đọc: {str(e)}")
            
    total_time = sum(monitor.get_durations().values())
    ram_usage = monitor.get_ram_usage()
    
    timing_html = get_performance_html(monitor.get_durations())
    
    # Prepend translation warning if fallback happened
    if "Auto-Detect" in translate_mode and is_offline_trans:
        gr.Warning("Mất kết nối Internet - Tự động chuyển sang dịch Offline (Helsinki-NLP)")
        
    return (
        eng_desc,
        vi_desc,
        audio_path,
        f"{total_time:.3f} s",
        f"{ram_usage:.1f} MB",
        timing_html
    )

# Gradio Interface build
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.HTML("<h1 style='text-align: center; color: #1f6feb;'>🌐 SmartSight AI — Hỗ Trợ Người Khiếm Thị</h1>")
    gr.HTML("<p style='text-align: center;'>Hệ thống mô tả hình ảnh tự động bằng giọng nói Tiếng Việt</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(sources=["webcam", "upload"], type="pil", label="Đầu vào hình ảnh")
            vlm_version = gr.Radio(
                choices=["Moondream2 (2B)", "Moondream2 (0.5B)"], 
                value="Moondream2 (2B)", 
                label="Mô hình VLM"
            )
            with gr.Row():
                run_btn = gr.Button("Run Pipeline", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")
                
            with gr.Accordion("Parameters & Thresholds (Cấu hình nâng cao)", open=False):
                translate_mode = gr.Dropdown(
                    choices=["Auto-Detect (Online)", "Offline (Helsinki-NLP)"], 
                    value="Auto-Detect (Online)", 
                    label="Chế độ dịch"
                )
                tts_mode = gr.Dropdown(
                    choices=["Auto-Detect (Online)", "Offline (pyttsx3)"], 
                    value="Auto-Detect (Online)", 
                    label="Chế độ TTS"
                )
                custom_prompt = gr.Textbox(
                    lines=2, 
                    label="VLM Prompt Template",
                    placeholder="Mặc định: Describe what you see..."
                )
                
        with gr.Column(scale=1):
            eng_out = gr.Textbox(label="Mô tả Tiếng Anh (VLM Output)", interactive=False)
            vi_out = gr.Textbox(label="Mô tả Tiếng Việt (Dịch)", interactive=False)
            audio_out = gr.Audio(label="Giọng đọc Tiếng Việt", autoplay=True, interactive=False)
            
            with gr.Group():
                gr.Markdown("### 📊 Performance Dashboard")
                with gr.Row():
                    total_time_lbl = gr.Textbox(label="TOTAL TIME", value="0.000 s", interactive=False)
                    ram_usage_lbl = gr.Textbox(label="RAM USAGE", value="0.0 MB", interactive=False)
                timing_chart = gr.HTML(value="<p>Chưa chạy xử lý.</p>")

    # Wire event handler with cancel support
    run_event = run_btn.click(
        fn=run_pipeline, 
        inputs=[input_image, vlm_version, translate_mode, tts_mode, custom_prompt],
        outputs=[eng_out, vi_out, audio_out, total_time_lbl, ram_usage_lbl, timing_chart]
    )
    
    # Cancel clicks terminate the run_event
    cancel_btn.click(fn=None, cancels=[run_event])

if __name__ == "__main__":
    # Pre-cache registry setup on startup with default Moondream2 (2B)
    print("Warm-starting ModelRegistry with Moondream2 (2B)...")
    registry.get_vlm("Moondream2 (2B)")
    
    # Enable queuing to serialise concurrency on CPU
    demo.queue().launch()
```

- [ ] **Step 2: Run all verification tests**

Run: `pytest -v`
Expected: ALL tests pass.

- [ ] **Step 3: Run the web application locally**

Run: `python src/app.py`
Expected: Webserver khởi động tại http://127.0.0.1:7860/ và load thành công mô hình Moondream2 (2B).

- [ ] **Step 4: Commit UI changes**

```bash
git add src/app.py
git commit -m "feat: add main Gradio web application with native queue, cancel, temp directory outputs, gr.Group container, and warm-start Moondream2 2B"
```
