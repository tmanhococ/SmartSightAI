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
