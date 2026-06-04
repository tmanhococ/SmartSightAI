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

def test_tts_none_or_empty_text():
    tts = TTSModule()
    assert tts.generate_speech(None) is None
    assert tts.generate_speech("") is None
    assert tts.generate_speech("   ") is None

def test_tts_file_deletion():
    tts = TTSModule()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name
    
    with open(temp_path, "w") as f:
        f.write("existing content")
    
    assert os.path.exists(temp_path)
    
    with patch("src.pipeline.tts.gTTS") as mock_gtts:
        mock_gtts_instance = MagicMock()
        mock_gtts.return_value = mock_gtts_instance
        
        output_path = tts.generate_speech("Hello", mode="Auto-Detect (Online)", filename=temp_path)
        assert output_path == temp_path
        # The existing file is removed inside generate_speech before gTTS saves
        # When mock saves, the file will be handled, but during execution we clean it.
        # Let's double check that the file is not raising exceptions.

def test_tts_default_filename():
    tts = TTSModule()
    with patch("src.pipeline.tts.gTTS") as mock_gtts:
        mock_gtts_instance = MagicMock()
        mock_gtts.return_value = mock_gtts_instance
        
        output_path = tts.generate_speech("Xin chào", mode="Auto-Detect (Online)")
        assert output_path is not None
        assert output_path.endswith("output.mp3")
        assert tempfile.gettempdir() in output_path

def test_tts_online_success():
    tts = TTSModule()
    with patch("src.pipeline.tts.gTTS") as mock_gtts:
        mock_gtts_instance = MagicMock()
        mock_gtts.return_value = mock_gtts_instance
        
        temp_path = os.path.join(tempfile.gettempdir(), "test_online.mp3")
        output_path = tts.generate_speech("Xin chào", mode="Auto-Detect (Online)", filename=temp_path)
        
        assert output_path == temp_path
        mock_gtts.assert_called_once_with(text="Xin chào", lang="vi", slow=False)
        mock_gtts_instance.save.assert_called_once_with(temp_path)

def test_tts_online_failure_fallback():
    tts = TTSModule()
    with patch("src.pipeline.tts.gTTS") as mock_gtts, \
         patch("src.pipeline.tts.pyttsx3.init") as mock_init:
        mock_gtts.side_effect = Exception("Network error")
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        temp_path = os.path.join(tempfile.gettempdir(), "test_fallback.mp3")
        output_path = tts.generate_speech("Xin chào", mode="Auto-Detect (Online)", filename=temp_path)
        
        assert output_path == temp_path
        mock_engine.save_to_file.assert_called_once_with("Xin chào", temp_path)
        mock_engine.runAndWait.assert_called_once()

def test_tts_offline_failure():
    tts = TTSModule()
    with patch("src.pipeline.tts.pyttsx3.init") as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        mock_engine.save_to_file.side_effect = Exception("pyttsx3 failed")
        
        temp_path = os.path.join(tempfile.gettempdir(), "test_fail.mp3")
        output_path = tts.generate_speech("Xin chào", mode="Offline (pyttsx3)", filename=temp_path)
        
        assert output_path is None

def test_tts_offline_init_failure():
    tts = TTSModule()
    with patch("src.pipeline.tts.pyttsx3.init") as mock_init:
        mock_init.side_effect = Exception("Init failed")
        temp_path = os.path.join(tempfile.gettempdir(), "test_init_fail.mp3")
        output_path = tts.generate_speech("Xin chào", mode="Offline (pyttsx3)", filename=temp_path)
        
        assert output_path is None

def test_tts_vietnamese_voice_selection():
    tts = TTSModule()
    with patch("src.pipeline.tts.pyttsx3.init") as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        mock_voice1 = MagicMock()
        mock_voice1.id = "english_voice"
        mock_voice1.languages = ["en-US"]
        mock_voice1.name = "English Voice"
        
        mock_voice2 = MagicMock()
        mock_voice2.id = "vietnamese_voice"
        mock_voice2.languages = ["vi-VN"]
        mock_voice2.name = "Vietnamese Voice"
        
        mock_engine.getProperty.return_value = [mock_voice1, mock_voice2]
        
        tts._init_offline()
        
        mock_engine.setProperty.assert_called_with("voice", "vietnamese_voice")
