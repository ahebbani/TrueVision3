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
        if not new_text:
            return
            
        words = new_text.split()
        if not words:
            return
            
        if not self.caption_history:
            self.caption_history.extend(words)
        else:
            # Find the largest overlap between the end of history and the start of words
            max_overlap = 0
            max_k = min(len(self.caption_history), len(words))
            
            for k in range(1, max_k + 1):
                hist_suffix = [w.lower().strip('.,!?') for w in self.caption_history[-k:]]
                new_prefix = [w.lower().strip('.,!?') for w in words[:k]]
                if hist_suffix == new_prefix:
                    max_overlap = k
                    
            # Append non-overlapping part
            self.caption_history.extend(words[max_overlap:])
            
        if len(self.caption_history) > self.max_history_words:
            self.caption_history = self.caption_history[-self.max_history_words:]

    def get_current_text(self) -> str:
        with self.lock:
            return " ".join(self.caption_history)
