from src.app import demo

if __name__ == "__main__":
    # Model loads lazily on first "Run" click via ModelRegistry singleton.
    # This lets the UI become available in ~5-10s instead of waiting ~40s
    # to download the 3.85GB Moondream2 model before Gradio even starts.
    demo.queue().launch(
        # Required for Hugging Face Spaces containers:
        server_name="0.0.0.0",   # bind all interfaces, not just localhost
        server_port=7860,         # standard HF Spaces port
        show_api=False,           # disable API schema generation (prevents TypeError
                                  # in gradio_client/utils.py get_type() on gradio 4.40.x)
    )
