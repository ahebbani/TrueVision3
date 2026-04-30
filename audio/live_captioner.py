"""
TrueVision — Live Captioner

Background worker that periodically transcribes a sliding audio window
and produces rolling captions for the HUD.
"""

import logging
import os
import threading
import time
from typing import Callable, Optional

from audio.recorder import ESP32SerialRecorder
from audio.transcriber import LocalTranscriber

logger = logging.getLogger(__name__)


class LiveCaptioner:
    def __init__(self, recorder: ESP32SerialRecorder, transcriber: LocalTranscriber, interval: float = 0.7, window: float = 2.0):
        self.recorder = recorder
        self.transcriber = transcriber
        self.interval = interval
        self.window = window
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.on_caption: Optional[Callable[[str], None]] = None
        
        # State
        self.caption_history = []
        self.max_history_words = 30
        self.generation = 0
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        
        # Ensure model is loading/loaded
        self.transcriber.start_loading()
            
        self.running = True
        self.generation += 1
        self.caption_history.clear()
        
        self.thread = threading.Thread(target=self._worker_loop, args=(self.generation,), daemon=True)
        self.thread.start()
        logger.info("Started Live Captioner")

    def stop(self):
        self.running = False
        with self.lock:
            self.generation += 1
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Stopped Live Captioner")

    def clear(self):
        with self.lock:
            self.caption_history.clear()
            if self.on_caption:
                self.on_caption("")

    def _worker_loop(self, my_generation: int):
        while self.running:
            with self.lock:
                if self.generation != my_generation:
                    break
                    
            start_t = time.time()
            
            # Flush recent audio
            temp_wav = self.recorder.flush_to_wav(self.window)
            if temp_wav:
                # Transcribe
                text = self.transcriber.transcribe_live(temp_wav)
                
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
                    
                if text and self.running:
                    with self.lock:
                        if self.generation == my_generation:
                            self._update_history(text)
                            if self.on_caption:
                                # We join the history.
                                # But wait, live transcribe might return overlapping text.
                                # For local fallback, we just append if it's new. 
                                # A simple heuristic: check if it's already at the end.
                                self.on_caption(self.get_current_text())
                                
            elapsed = time.time() - start_t
            sleep_time = max(0.1, self.interval - elapsed)
            time.sleep(sleep_time)

    def _update_history(self, new_text: str):
        # A very basic dedup logic for local captioning.
        # Server-side handles this much better.
        if not new_text:
            return
            
        words = new_text.split()
        if not words:
            return
            
        # Check if this text overlaps with the end of history
        overlap = False
        if len(self.caption_history) > 0:
            last_word = self.caption_history[-1]
            if words[0].lower().strip('.,!?') == last_word.lower().strip('.,!?'):
                overlap = True
                
        if not overlap:
            self.caption_history.extend(words)
            if len(self.caption_history) > self.max_history_words:
                self.caption_history = self.caption_history[-self.max_history_words:]

    def get_current_text(self) -> str:
        with self.lock:
            return " ".join(self.caption_history)
