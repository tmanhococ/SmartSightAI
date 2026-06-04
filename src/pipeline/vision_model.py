import torch
import torch.nn as nn
from PIL import Image


def _ensure_model_float32(model: nn.Module) -> None:
    """
    Force ALL parameters and buffers of the model (and any wrapped inner model)
    to float32.  This is necessary because Moondream2's cached source files
    (vision.py / moondream.py) create the `all_crops` image tensor with the
    same dtype as the model weights.  If any sub-module is still float16 the
    `F.linear(x, w.weight, w.bias)` call will raise a Half/Float mismatch.
    """
    # HfMoondream wraps MoondreamModel as .model — convert both levels.
    for obj in (model, getattr(model, "model", None)):
        if obj is not None and isinstance(obj, nn.Module):
            obj.float()  # converts parameters AND buffers in-place


def run_vlm_inference(image: Image.Image, version: str, model, processor, prompt: str = "") -> str:
    if version not in ["Moondream2 (2B)", "Moondream2 (0.5B)"]:
        raise ValueError(f"Unsupported model version: {version}")

    if not prompt or not prompt.strip():
        prompt = (
            "Describe what you see in this image briefly and clearly. "
            "Focus on the main subject, people, objects, and any important context. "
            "Keep it under 3 sentences."
        )

    # ── CPU float32 safety ──────────────────────────────────────────────────
    # Root cause of the "Half and Float" error on HF Space CPU:
    #   • Moondream2's vision.py calls `prepare_crops` which creates the
    #     image tensor (`all_crops`) in torch.get_default_dtype().
    #   • If the default dtype is float16 (or any sub-module still holds
    #     float16 weights), `F.linear(x, w.weight, w.bias)` crashes because
    #     x and w.weight have different dtypes.
    #
    # Fix: (1) force the global default dtype to float32 for the duration of
    # this call so that ALL new tensors are created as float32, and (2) also
    # force every model weight/buffer to float32.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        _ensure_model_float32(model)

        # Check for the newer API (model.query) or fallback to older API
        if hasattr(model, "query"):
            response = model.query(image, prompt)
            return response["answer"]
        else:
            # Fallback to older Moondream version API
            enc_image = model.encode_image(image)
            return model.answer_question(enc_image, prompt, processor)
    finally:
        # Always restore the previous default dtype, even on exception.
        torch.set_default_dtype(prev_dtype)
