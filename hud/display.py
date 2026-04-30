"""
TrueVision — HUD Display Manager

Main HUD renderer that composes all zones into a single full-screen
OpenCV window with a black background. On the OLED AR optic, black
pixels are transparent — all elements float over the real world.
"""

import logging
import platform
import time
from typing import Dict, List

import cv2
import numpy as np

from hud.toasts import ToastManager
from hud import zones

logger = logging.getLogger(__name__)


class DisplayManager:
    def __init__(self, resolution=(1280, 720)):
        self.width, self.height = resolution
        self.window_name = "TrueVision HUD"
        self.toast_manager = ToastManager()
        self.is_running = False

    def start(self):
        if self.is_running:
            return
            
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Try to make it fullscreen
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
            
        self.is_running = True
        logger.info("HUD Display started")

    def stop(self):
        self.is_running = False
        cv2.destroyWindow(self.window_name)
        logger.info("HUD Display stopped")

    def render_frame(
        self,
        mode: str,
        server_available: bool,
        faces_info: List[dict],
        caption_text: str,
        source_language: str,
        reminders: List[str]
    ) -> bool:
        """
        Renders a single frame and displays it.
        Returns False if the user pressed 'q' to quit.
        """
        if not self.is_running:
            return False
            
        # 1. Create purely black base frame (transparent on OLED)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 2. Render Global Zones (always visible)
        zones.render_clock_date(frame)
        zones.render_system_status(frame, server_available, mode, self.width)
        zones.render_reminders(frame, reminders, self.height)
        zones.render_toasts(frame, self.toast_manager, self.width, self.height)
        
        # 3. Render Mode-Specific Zones
        if mode in ("face", "both"):
            # Map camera coordinates to HUD coordinates
            # For this prototype, we assume the camera frame (e.g. 640x480) 
            # is scaled and centered on the HUD (e.g. 1280x720)
            # To keep it simple, we just pass the raw faces_info and let the zone renderer draw it.
            # In a real AR system, complex projection math is needed here.
            zones.render_face_recognition(frame, faces_info)
            
        if mode in ("audio", "both"):
            zones.render_live_captions(frame, caption_text, source_language, self.width, self.height)
            
        # 4. Display
        cv2.imshow(self.window_name, frame)
        
        # 5. Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
            
        return True

    def show_toast(self, text: str, duration: float = 2.0):
        self.toast_manager.show(text, duration)
