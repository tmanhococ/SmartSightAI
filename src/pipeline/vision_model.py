import torch
import torch.nn as nn
from PIL import Image


def _ensure_float32(model: nn.Module) -> nn.Module:
    """Convert all model weights to float32 to prevent Half/Float dtype mismatch on CPU."""
    try:
        device = next(model.parameters()).device
        if device.type == "cpu":
            # Walk the full module tree to ensure every sub-module is float32
            inner = getattr(model, "model", model)
            inner.float()
    except StopIteration:
        pass  # model has no parameters (unlikely), safe to ignore
    return model


def run_vlm_inference(image: Image.Image, version: str, model, processor, prompt: str = "") -> str:
    if version not in ["Moondream2 (2B)", "Moondream2 (0.5B)"]:
        raise ValueError(f"Unsupported model version: {version}")

    if not prompt or not prompt.strip():
        prompt = (
            "Describe what you see in this image briefly and clearly. "
            "Focus on the main subject, people, objects, and any important context. "
            "Keep it under 3 sentences."
        )

    # Guarantee float32 on CPU before every inference call.
    # This guards against HF-Space runtime re-loading weights in half precision.
    _ensure_float32(model)

    # Check for the newer API (model.query) or fallback to older API
    if hasattr(model, "query"):
        # Wrap in no-op autocast (CPU) to prevent any implicit half-precision cast
        with torch.autocast(device_type="cpu", enabled=False):
            response = model.query(image, prompt)
        return response["answer"]
    else:
        # Fallback to older Moondream version API
        with torch.autocast(device_type="cpu", enabled=False):
            enc_image = model.encode_image(image)
            return model.answer_question(enc_image, prompt, processor)
