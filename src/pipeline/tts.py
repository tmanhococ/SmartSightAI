import os
import tempfile
import logging
from gtts import gTTS
import pyttsx3

class TTSModule:
    def __init__(self):
        self.offline_engine = None
        
    def _init_offline(self):
        if self.offline_engine is None:
            try:
                self.offline_engine = pyttsx3.init()
                voices = self.offline_engine.getProperty("voices")
                for voice in voices:
                    if voice.languages and any("vi" in lang for lang in voice.languages):
                        self.offline_engine.setProperty("voice", voice.id)
                        break
                    elif "vietnam" in voice.name.lower():
                        self.offline_engine.setProperty("voice", voice.id)
                        break
            except Exception as e:
                logging.error(f"Failed to initialize pyttsx3 offline TTS: {e}")
                
    def generate_speech(self, text: str, mode: str = "Auto-Detect (Online)", filename: str = None) -> str:
        if not text or not text.strip():
            return None
            
        if filename is None:
            filename = os.path.join(tempfile.gettempdir(), "output.mp3")
            
        # Clean existing file to avoid permission or handle lock issues
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
                
        if mode == "Offline (pyttsx3)":
            try:
                self._generate_offline(text, filename)
                return filename
            except Exception as e:
                logging.error(f"Offline TTS failed: {e}")
                return None
            
        # Try Online gTTS
        try:
            tts = gTTS(text=text, lang="vi", slow=False)
            tts.save(filename)
            return filename
        except Exception as e:
            logging.warning(f"Online gTTS failed: {e}. Falling back to pyttsx3.")
            try:
                self._generate_offline(text, filename)
                return filename
            except Exception as e_off:
                logging.error(f"Offline TTS failed: {e_off}")
                return None
            
    def _generate_offline(self, text: str, filename: str):
        self._init_offline()
        if self.offline_engine:
            self.offline_engine.save_to_file(text, filename)
            self.offline_engine.runAndWait()
        else:
            raise RuntimeError("Offline TTS engine pyttsx3 is not available.")
