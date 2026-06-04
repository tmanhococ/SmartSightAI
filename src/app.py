import os
import sys
import tempfile
import time
import uuid
import logging
import gradio as gr
from PIL import Image
from src.registry import ModelRegistry
from src.pipeline.preprocess import preprocess_image
from src.pipeline.vision_model import run_vlm_inference
from src.pipeline.tts import TTSModule
from src.utils.monitor import ExecutionMonitor

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("smartsight")

logger.info("=" * 60)
logger.info("SmartSight AI — startup")
logger.info(f"  Python  : {sys.version.split()[0]}")
logger.info(f"  Gradio  : {gr.__version__}")
logger.info(f"  Platform: {sys.platform}")
logger.info("=" * 60)

# ─── Global singletons ────────────────────────────────────────────────────────
logger.info("Initialising ModelRegistry and TTSModule singletons...")
try:
    registry = ModelRegistry()
    tts_module = TTSModule()
    logger.info("Singletons ready (models will lazy-load on first Run).")
except Exception:
    logger.critical("Failed to initialise singletons!", exc_info=True)
    raise


# ─── Helpers ──────────────────────────────────────────────────────────────────
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


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(image, vlm_version, translate_mode, tts_mode, custom_prompt):
    logger.info(
        "run_pipeline called | vlm=%s | translate=%s | tts=%s",
        vlm_version, translate_mode, tts_mode,
    )
    monitor = ExecutionMonitor()

    if image is None:
        logger.warning("run_pipeline: no image provided")
        raise gr.Error("Vui lòng chụp ảnh hoặc tải ảnh lên trước!")

    # Preprocessing
    with monitor.track("preprocess"):
        try:
            img = preprocess_image(image)
            logger.info("Preprocess OK: %s → %s", image.size if hasattr(image, 'size') else '?', img.size)
        except Exception as e:
            logger.error("Preprocess failed: %s", e, exc_info=True)
            raise gr.Error(f"Lỗi xử lý ảnh: {str(e)}")

    # Load VLM and Inference
    with monitor.track("vlm_inference"):
        try:
            logger.info("Loading VLM: %s", vlm_version)
            vlm_model, vlm_processor = registry.get_vlm(vlm_version)
            eng_desc = run_vlm_inference(img, vlm_version, vlm_model, vlm_processor, custom_prompt)
            logger.info("VLM inference OK (%d chars)", len(eng_desc))
        except Exception as e:
            logger.error("VLM inference failed: %s", e, exc_info=True)
            raise gr.Error(f"Lỗi VLM Inference: {str(e)}")

    # Translate
    with monitor.track("translation"):
        try:
            translator = registry.get_translator_module(translate_mode)
            vi_desc, is_offline_trans = translator.translate(eng_desc, translate_mode)
            logger.info("Translation OK (offline=%s)", is_offline_trans)
        except Exception as e:
            logger.warning("Translation failed: %s", e, exc_info=True)
            vi_desc = f"[Lỗi dịch] {eng_desc}"
            is_offline_trans = False
            gr.Warning(f"Dịch thuật thất bại: {str(e)}")

    # TTS
    with monitor.track("tts"):
        try:
            temp_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex}.mp3")
            audio_path = tts_module.generate_speech(vi_desc, tts_mode, filename=temp_path)
            logger.info("TTS OK: %s", audio_path)
        except Exception as e:
            logger.warning("TTS failed: %s", e, exc_info=True)
            audio_path = None
            gr.Warning(f"Không thể tạo giọng đọc: {str(e)}")

    total_time = sum(monitor.get_durations().values())
    ram_usage = monitor.get_ram_usage()
    timing_html = get_performance_html(monitor.get_durations())

    logger.info("Pipeline complete: %.3fs | RAM %.1fMB", total_time, ram_usage)

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


# ─── Gradio UI ────────────────────────────────────────────────────────────────
logger.info("Building Gradio Blocks UI...")
try:
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

        run_event = run_btn.click(
            fn=run_pipeline,
            inputs=[input_image, vlm_version, translate_mode, tts_mode, custom_prompt],
            outputs=[eng_out, vi_out, audio_out, total_time_lbl, ram_usage_lbl, timing_chart]
        )
        cancel_btn.click(fn=None, cancels=[run_event])

    logger.info("Gradio Blocks UI built successfully.")

except Exception:
    logger.critical("Failed to build Gradio UI!", exc_info=True)
    raise


if __name__ == "__main__":
    # For local development only — HF Spaces uses the root app.py instead.
    # Model loads lazily on first Run click (no warm-start blocking the UI).
    demo.queue().launch()
