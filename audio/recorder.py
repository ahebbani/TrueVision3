"""
TrueVision — ESP32 Serial Recorder

Drop-in recorder that wraps the serial receiver for session-based
WAV recording.
"""

import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf
import numpy as np

from audio.serial_receiver import SerialReceiver

logger = logging.getLogger(__name__)


class ESP32SerialRecorder:
    def __init__(self, serial_port: str, baud_rate: int = 921600):
        self.receiver = SerialReceiver.get_instance(serial_port, baud_rate)
        self.is_recording = False
        self.start_time = 0
        self.session_dir = ""
        self.session_prefix = ""
        self.sample_rate = self.receiver.sample_rate

    def start(self, directory: str = "recordings", prefix: str = "session"):
        if self.is_recording:
            logger.warning("Recorder already running. Stopping previous session.")
            self.stop()
            
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.session_dir = directory
        self.session_prefix = prefix
        
        self.receiver.clear_buffer()
        self.start_time = time.time()
        self.is_recording = True
        logger.info(f"Started recording session '{prefix}'")
        
        # We don't return a path yet, because we wait until stop() or flush() to write

    def stop(self) -> Optional[str]:
        if not self.is_recording:
            return None
            
        self.is_recording = False
        duration = time.time() - self.start_time
        
        # Require at least 0.5s of audio to avoid Whisper errors
        if duration < 0.5:
            logger.warning(f"Recording too short ({duration:.2f}s). Discarding.")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_prefix}_{timestamp}.wav"
        filepath = os.path.join(self.session_dir, filename)
        
        audio_data = self.receiver.get_all_audio()
        
        if len(audio_data) == 0:
            logger.warning("No audio data received during recording session.")
            return None
            
        try:
            sf.write(filepath, audio_data, self.sample_rate)
            logger.info(f"Saved recording: {filepath} ({len(audio_data)/self.sample_rate:.2f}s)")
            return filepath
        except Exception as e:
            logger.error(f"Failed to write WAV file {filepath}: {e}")
            return None

    def flush_to_wav(self, seconds: float) -> Optional[str]:
        """Writes the last N seconds of the current session to a temporary WAV file.
           Useful for live captioning without stopping the recording."""
        if not self.is_recording:
            return None
            
        audio_data = self.receiver.get_last_n_seconds(seconds)
        if len(audio_data) < self.sample_rate * 0.5: # min 0.5s
            return None
            
        try:
            fd, filepath = tempfile.mkstemp(suffix=".wav", prefix="live_")
            os.close(fd)
            sf.write(filepath, audio_data, self.sample_rate)
            return filepath
        except Exception as e:
            logger.error(f"Failed to flush live audio: {e}")
            return None
