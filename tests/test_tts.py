import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.tts import TTSModule

def test_tts_offline():
    # Mock pyttsx3 engine in the correct import scope
    with patch("src.pipeline.tts.pyttsx3.init") as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        tts = TTSModule()
        temp_dir = tempfile.gettempdir()
        target_path = os.path.join(temp_dir, "test_offline.mp3")
        
        output_path = tts.generate_speech("Xin chào", mode="Offline (pyttsx3)", filename=target_path)
        
        assert output_path == target_path
        mock_engine.save_to_file.assert_called_once_with("Xin chào", target_path)
        mock_engine.runAndWait.assert_called_once()
