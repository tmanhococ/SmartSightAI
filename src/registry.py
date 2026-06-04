import sys
import types
import logging
import threading
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
from src.pipeline.translate import TranslatorModule

logger = logging.getLogger("smartsight")


def _patch_moondream_for_cpu(model: nn.Module) -> nn.Module:
    """
    Patch Moondream2 (revision 2025-01-09) for correct float32 CPU inference.

    The upstream cached files have two hardcoded float16 issues that break CPU:

      [Bug 1] vision.py line 45:
          all_crops = torch.from_numpy(...).to(device=device, dtype=torch.float16)
          → image crop tensor is Half, but model weights are Float32 after .float()
          → RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float

      [Bug 2] moondream.py encode_image():
          kv_cache = torch.zeros(..., dtype=torch.float16)
          → kv_cache is Half, but text model activations are Float32
          → Potential dtype mismatch in prefill/decode_one_token ops

    These files are cached by HuggingFace Hub and cannot be edited at deploy time.
    We fix both bugs by monkey-patching the relevant functions after model load.

    On GPU (CUDA / MPS) the model is left completely untouched — float16 is fine there.
    """
    # HfMoondream wraps MoondreamModel as .model
    inner: nn.Module = getattr(model, "model", model)

    try:
        device = next(inner.parameters()).device
    except StopIteration:
        device = next(model.parameters()).device

    if device.type != "cpu":
        logger.info("GPU detected: skipping CPU float32 patch for Moondream2.")
        return model

    logger.info("CPU detected: applying Moondream2 float32 patches for HF Space.")

    # ── Step 1: Convert all nn.Parameters and registered buffers to float32 ────
    inner.float()
    logger.info("Step 1 done: model weights/buffers converted to float32.")

    # ── Step 2: Patch ops["vision_encoder"] ─────────────────────────────────────
    # Bug 1 fix: vision.py hardcodes .to(dtype=torch.float16) for image crops.
    # We wrap the vision_encoder op to convert the crop tensor to float32 first.
    _orig_vision_encoder = inner.ops["vision_encoder"]

    def _float32_vision_encoder(crops: torch.Tensor, w, cfg):
        return _orig_vision_encoder(crops.to(torch.float32), w, cfg)

    inner.ops["vision_encoder"] = _float32_vision_encoder
    logger.info("Step 2 done: vision_encoder patched — image crops cast float16→float32.")

    # ── Step 3: Patch encode_image to use float32 kv_cache ──────────────────────
    # Bug 2 fix: encode_image hardcodes dtype=torch.float16 for kv_cache.
    # We rewrite encode_image to use float32 for the kv_cache tensor.
    moondream_mod = sys.modules.get(type(inner).__module__)
    if moondream_mod is None:
        logger.warning("Step 3 skipped: could not locate moondream module in sys.modules.")
    else:
        _EncodedImage = getattr(moondream_mod, "EncodedImage", None)
        _text_encoder = getattr(moondream_mod, "text_encoder", None)

        if _EncodedImage is None or _text_encoder is None:
            logger.warning(
                "Step 3 skipped: EncodedImage or text_encoder not found in moondream module."
            )
        else:
            def _patched_encode_image(self, image):
                # Pass-through if image is already encoded
                if isinstance(image, _EncodedImage):
                    return image
                if not hasattr(image, "size"):
                    raise ValueError("image must be a PIL Image or EncodedImage")

                # Use float32 instead of the hardcoded float16
                kv_cache = torch.zeros(
                    self.config.text.n_layers,
                    2,   # k, v
                    1,   # batch size
                    self.config.text.n_heads,
                    self.config.text.max_context,
                    self.config.text.dim // self.config.text.n_heads,
                    device=self.device,
                    dtype=torch.float32,  # ← patched from float16
                )

                with torch.no_grad():
                    img_emb = self._run_vision_encoder(image)
                    bos_emb = _text_encoder(
                        torch.tensor(
                            [[self.config.tokenizer.bos_id]], device=self.device
                        ),
                        self.text,
                    )
                    inputs_embeds = torch.cat([bos_emb, img_emb[None]], dim=1)
                    self.ops["prefill"](
                        inputs_embeds, kv_cache, 0, self.text, self.config.text
                    )

                return _EncodedImage(pos=inputs_embeds.size(1), kv_cache=kv_cache)

            inner.encode_image = types.MethodType(_patched_encode_image, inner)
            logger.info(
                "Step 3 done: encode_image patched — kv_cache dtype float16→float32."
            )

    logger.info("Moondream2 CPU patches applied. Ready for float32 inference.")
    return model


class ModelRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
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
                model_id = "andito/moondream05"  # Community transformers 0.5B model
                processor = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )
            else:  # Moondream2 (2B)
                model_id = "vikhyatk/moondream2"
                revision = "2025-01-09"  # API 2025 compatible
                processor = AutoTokenizer.from_pretrained(model_id, revision=revision)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # load weights as float32
                )
                model = _patch_moondream_for_cpu(model)
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
