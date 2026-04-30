"""
TrueVision — HUD Toast System

Manages confirmation toasts that appear center-screen and fade
after ~2 seconds.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Toast:
    text: str
    duration: float = 2.0
    created_at: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.age > self.duration

    @property
    def alpha(self) -> float:
        """Returns a value 0.0 to 1.0 for fading out."""
        # Fade out during the last 0.5 seconds
        fade_time = 0.5
        time_left = self.duration - self.age
        if time_left < 0:
            return 0.0
        if time_left > fade_time:
            return 1.0
        return time_left / fade_time


class ToastManager:
    def __init__(self):
        self.toasts: List[Toast] = []
        self.lock = threading.Lock()

    def show(self, text: str, duration: float = 2.0):
        with self.lock:
            self.toasts.append(Toast(text=text, duration=duration))

    def get_active_toasts(self) -> List[Toast]:
        with self.lock:
            # Clean up expired
            self.toasts = [t for t in self.toasts if not t.is_expired]
            return list(self.toasts) # Return a copy
