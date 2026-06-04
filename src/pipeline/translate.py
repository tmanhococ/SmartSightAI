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
        # Move tensors to same device as the model (handles CPU/GPU HF Spaces)
        if hasattr(self.offline_model, "device") and hasattr(inputs, "to"):
            inputs = inputs.to(self.offline_model.device)
        generated_ids = self.offline_model.generate(**inputs)
        translated_text = self.offline_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return translated_text
