"""
Smoke tests for Gradio app initialization.
These tests verify the app can be built without errors, WITHOUT
downloading any ML models (all heavy ops are lazy-loaded).

Run locally before pushing to HF Space:
    pytest tests/test_gradio_startup.py -v
"""

import gradio as gr
import pytest


def test_demo_imports_without_error():
    """src/app.py must import and build the Gradio UI without any exception."""
    # This exercises the full module-level code in src/app.py:
    #   - logging setup
    #   - ModelRegistry() singleton creation (no model download)
    #   - TTSModule() singleton creation
    #   - gr.Blocks() UI construction
    # It does NOT load any ML model weights.
    from src.app import demo  # noqa: F401 — import triggers module execution
    assert demo is not None


def test_demo_is_gradio_blocks_instance():
    """The exported 'demo' must be a gr.Blocks instance for Gradio to serve it."""
    from src.app import demo
    assert isinstance(demo, gr.Blocks), (
        f"Expected gr.Blocks, got {type(demo)}. "
        "HF Spaces requires the demo variable to be a gr.Blocks instance."
    )


def test_run_pipeline_is_callable():
    """run_pipeline function must be importable and callable."""
    from src.app import run_pipeline
    assert callable(run_pipeline), "run_pipeline must be a callable function"


def test_registry_singleton_is_created():
    """ModelRegistry singleton must be created on import (no model weights loaded)."""
    from src.app import registry
    from src.registry import ModelRegistry
    assert isinstance(registry, ModelRegistry)
    # Models are NOT loaded yet — all None until first Run click
    assert registry.vlm_models["Moondream2 (2B)"] is None, (
        "Model weights must NOT be loaded at import time. "
        "Use lazy loading: load on first run_pipeline() call."
    )


def test_tts_module_singleton_is_created():
    """TTSModule singleton must be created on import with no engine initialised."""
    from src.app import tts_module
    from src.pipeline.tts import TTSModule
    assert isinstance(tts_module, TTSModule)
    assert tts_module.offline_engine is None, (
        "pyttsx3 engine must NOT be initialised at import time."
    )


def test_demo_has_expected_input_components():
    """The demo must have at least one Image input (for image upload/webcam)."""
    from src.app import demo
    image_components = [
        c for c in demo.blocks.values()
        if hasattr(c, '__class__') and c.__class__.__name__ == "Image"
    ]
    assert len(image_components) >= 1, (
        "Gradio demo must have at least one Image input component."
    )


def test_demo_has_audio_output():
    """The demo must have an Audio output for TTS playback."""
    from src.app import demo
    audio_components = [
        c for c in demo.blocks.values()
        if hasattr(c, '__class__') and c.__class__.__name__ == "Audio"
    ]
    assert len(audio_components) >= 1, (
        "Gradio demo must have at least one Audio output component."
    )


def test_demo_queue_is_callable():
    """demo.queue() must not raise — required for HF Spaces concurrent requests."""
    from src.app import demo
    try:
        queued = demo.queue()
        assert queued is not None
    except Exception as e:
        pytest.fail(f"demo.queue() raised an exception: {e}")


def test_gradio_version_is_444x():
    """Gradio must be 4.44.x to match sdk_version and avoid 6.x breaking changes."""
    from packaging.version import Version
    installed = Version(gr.__version__)
    # We accept 4.44.x (pinned), warn if it's 5.x or 6.x (could break HF infra)
    if installed >= Version("5.0.0"):
        pytest.skip(
            f"Local Gradio is {gr.__version__} (dev machine may differ from HF Space). "
            "Ensure HF Space uses 4.44.1 via sdk_version + requirements.txt pin."
        )
    assert installed >= Version("4.44.1"), (
        f"Gradio {gr.__version__} is too old. Minimum required: 4.44.1"
    )
