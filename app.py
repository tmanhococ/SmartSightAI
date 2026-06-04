from src.app import demo, registry

if __name__ == "__main__":
    # Pre-cache registry setup on startup with default Moondream2 (2B)
    print("Warm-starting ModelRegistry with Moondream2 (2B)...")
    registry.get_vlm("Moondream2 (2B)")

    demo.queue().launch(
        # Required for Hugging Face Spaces containers:
        server_name="0.0.0.0",   # bind all interfaces, not just localhost
        server_port=7860,         # standard HF Spaces port
        show_api=False,           # disable API schema generation (prevents TypeError
                                  # in gradio_client/utils.py get_type() on gradio 4.40.x)
    )
