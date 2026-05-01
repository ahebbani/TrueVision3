"""
TrueVision — Local Whisper Transcriber

Wraps faster-whisper (CTranslate2) for on-device transcription
on the Raspberry Pi.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy load faster_whisper to avoid blocking import
WhisperModel = None

class LocalTranscriber:
    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._load_thread = None

    def start_loading(self):
        """Start loading the model in a background thread."""
        with self.lock:
            if self.model is not None or (self._load_thread and self._load_thread.is_alive()):
                return
            self._load_thread = threading.Thread(target=self._load_model, daemon=True)
            self._load_thread.start()

    def _load_model(self):
        global WhisperModel
        try:
            logger.info(f"Loading local Whisper model ({self.model_size}) on {self.device}...")
            if WhisperModel is None:
                from faster_whisper import WhisperModel
            
            model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            with self.lock:
                self.model = model
            logger.info("Local Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")

    def _ensure_model(self, wait: bool = False):
        with self.lock:
            if self.model is not None:
                return True
        
        if self._load_thread and self._load_thread.is_alive():
            if wait:
                logger.info("Waiting for Whisper model to finish loading...")
                self._load_thread.join()
                with self.lock:
                    return self.model is not None
            else:
                return False # Still loading, don't block
                
        # If thread is NOT alive and model is None, it failed or hasn't started.
        if wait:
            self._load_model()
            return self.model is not None
        else:
            self.start_loading()
            return False

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Full transcription optimized for accuracy."""
        if not self._ensure_model(wait=True):
            return None
            
        with self.inference_lock:
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
                logger.error(f"Local transcription failed: {e}")
                return None

    def transcribe_live(self, audio_path: str, language: str = "en") -> Optional[str]:
        """Fast transcription optimized for live captioning (no VAD, beam_size=1)."""
        if not self._ensure_model(wait=False):
            return None
            
        with self.inference_lock:
            try:
                segments, _ = self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    without_timestamps=True
                )
                text = " ".join([segment.text for segment in segments]).strip()
                return text if text else None
            except Exception as e:
                logger.error(f"Local live transcription failed: {e}")
                return None
