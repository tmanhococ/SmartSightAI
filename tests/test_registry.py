import pytest
from unittest.mock import patch, MagicMock
from src.registry import ModelRegistry

@pytest.fixture(autouse=True)
def reset_singleton():
    ModelRegistry._instance = None
    yield
    ModelRegistry._instance = None

def test_singleton_pattern():
    reg1 = ModelRegistry()
    reg2 = ModelRegistry()
    assert reg1 is reg2

@patch("src.registry.AutoModelForCausalLM.from_pretrained")
@patch("src.registry.AutoTokenizer.from_pretrained")
def test_get_vlm_2b(mock_tokenizer_init, mock_model_init):
    mock_model_init.return_value = "mock_model_2b"
    mock_tokenizer_init.return_value = "mock_tokenizer_2b"
    
    registry = ModelRegistry()
    model, tokenizer = registry.get_vlm("Moondream2 (2B)")
    assert model == "mock_model_2b"
    assert tokenizer == "mock_tokenizer_2b"
    
    # Assert cached
    model2, tokenizer2 = registry.get_vlm("Moondream2 (2B)")
    assert model2 is model
    assert tokenizer2 is tokenizer
    mock_model_init.assert_called_once()
    mock_tokenizer_init.assert_called_once()

@patch("src.registry.AutoModelForCausalLM.from_pretrained")
@patch("src.registry.AutoTokenizer.from_pretrained")
def test_get_vlm_05b(mock_tokenizer_init, mock_model_init):
    mock_model_init.return_value = "mock_model_05b"
    mock_tokenizer_init.return_value = "mock_tokenizer_05b"
    
    registry = ModelRegistry()
    model, tokenizer = registry.get_vlm("Moondream2 (0.5B)")
    assert model == "mock_model_05b"
    assert tokenizer == "mock_tokenizer_05b"
    
    # Assert cached
    model2, tokenizer2 = registry.get_vlm("Moondream2 (0.5B)")
    assert model2 is model
    assert tokenizer2 is tokenizer
    mock_model_init.assert_called_once()
    mock_tokenizer_init.assert_called_once()

def test_get_vlm_invalid():
    registry = ModelRegistry()
    with pytest.raises(ValueError) as exc_info:
        registry.get_vlm("InvalidModel")
    assert "Unknown VLM model version" in str(exc_info.value)

@patch("src.registry.MarianMTModel.from_pretrained")
@patch("src.registry.MarianTokenizer.from_pretrained")
def test_get_translator_module(mock_tokenizer_init, mock_model_init):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model_init.return_value = mock_model
    mock_tokenizer_init.return_value = mock_tokenizer
    
    registry = ModelRegistry()
    translator = registry.get_translator_module("Offline (Helsinki-NLP)")
    
    assert translator.offline_model == mock_model
    assert translator.offline_tokenizer == mock_tokenizer
    
    # Assert cached
    translator2 = registry.get_translator_module("Offline (Helsinki-NLP)")
    assert translator2 is translator
    mock_model_init.assert_called_once()
    mock_tokenizer_init.assert_called_once()
