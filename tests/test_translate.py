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

def test_translate_empty_or_whitespace():
    translator = TranslatorModule()
    assert translator.translate("") == ("", False)
    assert translator.translate("   ") == ("", False)

@patch("src.pipeline.translate.GoogleTranslator.translate")
def test_translate_online_failure_with_offline_fallback(mock_google_translate):
    mock_google_translate.side_effect = Exception("Network Error")
    
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    mock_model.generate.return_value = [[4, 5, 6]]
    mock_tokenizer.decode.return_value = "Xin chào"
    
    translator = TranslatorModule(offline_model=mock_model, offline_tokenizer=mock_tokenizer)
    result, is_offline = translator.translate("Hello", mode="Auto-Detect (Online)")
    
    assert result == "Xin chào"
    assert is_offline is True

@patch("src.pipeline.translate.GoogleTranslator.translate")
def test_translate_online_failure_no_offline_fallback(mock_google_translate):
    mock_google_translate.side_effect = Exception("Network Error")
    
    translator = TranslatorModule()
    result, is_offline = translator.translate("Hello", mode="Auto-Detect (Online)")
    
    assert result == "[Lỗi mạng - Không dịch được] Hello"
    assert is_offline is False

