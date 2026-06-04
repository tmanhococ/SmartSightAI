import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
from src.pipeline.translate import TranslatorModule

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
                model_id = "andito/moondream05" # Community transformers 0.5B model
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
