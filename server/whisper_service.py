"""
TrueVision — Server Whisper Service

Server-side faster-whisper transcription with CUDA support
and automatic fallback to CPU.
"""

import logging
import os
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy load
WhisperModel = None

class ServerWhisperService:
    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        self.model = None
        self.lock = threading.Lock()
        
        # Determine actual device if auto
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                self.compute_type = "int8"
                logger.warning("CUDA not available. Falling back to CPU/int8 for Whisper.")

    def load_model(self):
        """Loads the model synchronously."""
        global WhisperModel
        with self.lock:
            if self.model is not None:
                return
                
            try:
                logger.info(f"Loading Server Whisper model ({self.model_size}) on {self.device}...")
                if WhisperModel is None:
                    from faster_whisper import WhisperModel
                
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                logger.info("Server Whisper model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Server Whisper model: {e}")
                
                # If CUDA failed, try CPU fallback
                if self.device == "cuda":
                    logger.info("Retrying with CPU fallback...")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    try:
                        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                        logger.info("CPU fallback successful.")
                    except Exception as e2:
                        logger.error(f"CPU fallback also failed: {e2}")

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Full transcription."""
        if self.model is None:
            self.load_model()
        if self.model is None:
            return None
            
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            text = " ".join([segment.text for segment in segments]).strip()
            return text if text else None
        except Exception as e:
            logger.error(f"Server transcription failed: {e}")
            return None

    def transcribe_live(self, audio_path: str, force_language: str = None) -> Tuple[Optional[str], Optional[str], float]:
        """
        Live transcription for captions. Also performs language detection for translation.
        Returns: (text, detected_language_code, language_probability)
        """
        if self.model is None:
            self.load_model()
        if self.model is None:
            return None, None, 0.0
            
        try:
            kwargs = {
                "beam_size": 1,
                "vad_filter": False,
                "condition_on_previous_text": False,
                "without_timestamps": True
            }
            if force_language:
                kwargs["language"] = force_language
                
            segments, info = self.model.transcribe(audio_path, **kwargs)
            text = " ".join([segment.text for segment in segments]).strip()
            
            return text if text else None, info.language, info.language_probability
        except Exception as e:
            logger.error(f"Server live transcription failed: {e}")
            return None, None, 0.0

    def translate(self, audio_path: str, source_language: str) -> Optional[str]:
        """Uses Whisper's translate task to translate non-English audio to English."""
        if self.model is None:
            self.load_model()
        if self.model is None:
            return None
            
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                task="translate",
                language=source_language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True
            )
            text = " ".join([segment.text for segment in segments]).strip()
            return text if text else None
        except Exception as e:
            logger.error(f"Server translation failed: {e}")
            return None
