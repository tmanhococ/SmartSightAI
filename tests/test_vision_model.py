import pytest
from unittest.mock import MagicMock
from PIL import Image
from src.pipeline.vision_model import run_vlm_inference

def test_run_vlm_moondream_new_api_mock():
    mock_model = MagicMock()
    # Mock the new query method
    mock_model.query.return_value = {"answer": "a photo of a laptop"}
    
    img = Image.new("RGB", (224, 224))
    result = run_vlm_inference(img, "Moondream2 (2B)", mock_model, MagicMock(), prompt="What is this?")
    assert result == "a photo of a laptop"
    mock_model.query.assert_called_once_with(img, "What is this?")

def test_run_vlm_moondream_old_api_fallback_mock():
    mock_model = MagicMock()
    # Delete query method to trigger fallback
    del mock_model.query
    mock_model.encode_image.return_value = "encoded"
    mock_model.answer_question.return_value = "a photo of a laptop from fallback"
    
    img = Image.new("RGB", (224, 224))
    result = run_vlm_inference(img, "Moondream2 (0.5B)", mock_model, "processor", prompt="What is this?")
    assert result == "a photo of a laptop from fallback"
    mock_model.encode_image.assert_called_once_with(img)
    mock_model.answer_question.assert_called_once_with("encoded", "What is this?", "processor")

def test_run_vlm_invalid_version():
    mock_model = MagicMock()
    img = Image.new("RGB", (224, 224))
    with pytest.raises(ValueError, match="Unsupported model version: InvalidModel"):
        run_vlm_inference(img, "InvalidModel", mock_model, MagicMock())

def test_run_vlm_default_prompt():
    mock_model = MagicMock()
    mock_model.query.return_value = {"answer": "a photo of a laptop"}
    img = Image.new("RGB", (224, 224))
    
    # Test with empty prompt
    run_vlm_inference(img, "Moondream2 (2B)", mock_model, MagicMock(), prompt="")
    default_prompt = (
        "Describe what you see in this image briefly and clearly. "
        "Focus on the main subject, people, objects, and any important context. "
        "Keep it under 3 sentences."
    )
    mock_model.query.assert_called_with(img, default_prompt)
    
    # Test with whitespace prompt
    run_vlm_inference(img, "Moondream2 (2B)", mock_model, MagicMock(), prompt="   ")
    mock_model.query.assert_called_with(img, default_prompt)

