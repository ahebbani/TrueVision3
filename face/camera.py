"""
TrueVision — Camera Backend

Provides a unified camera interface that auto-detects the platform
and uses the appropriate backend:
  - Picamera2 (libcamera) on Raspberry Pi
  - OpenCV VideoCapture on desktop/Mac
"""

import logging
import platform
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.resolution = resolution
        self.framerate = framerate
        self.backend = None
        self.cap = None
        self.is_running = False
        
        self._detect_platform()

    def _detect_platform(self):
        machine = platform.machine()
        sys = platform.system()
        
        if sys == "Linux" and machine in ("armv7l", "aarch64"):
            logger.info("Detected Raspberry Pi. Using Picamera2 backend.")
            self.backend = "picamera2"
        else:
            logger.info(f"Detected {sys} {machine}. Using OpenCV backend.")
            self.backend = "opencv"

    def start(self) -> bool:
        if self.is_running:
            return True
            
        if self.backend == "picamera2":
            return self._start_picamera()
        else:
            return self._start_opencv()

    def _start_picamera(self) -> bool:
        try:
            from picamera2 import Picamera2, controls
            self.cap = Picamera2()
            config = self.cap.create_preview_configuration(
                main={"size": self.resolution, "format": "BGR888"}
            )
            self.cap.configure(config)
            self.cap.set_controls({"FrameRate": self.framerate})
            self.cap.start()
            self.is_running = True
            logger.info("Picamera2 started successfully.")
            return True
        except ImportError:
            logger.error("Picamera2 not installed. Falling back to OpenCV.")
            self.backend = "opencv"
            return self._start_opencv()
        except Exception as e:
            logger.error(f"Failed to start Picamera2: {e}")
            return False

    def _start_opencv(self) -> bool:
        # OpenCV handles AVFoundation on Mac, V4L2 on Linux
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open OpenCV VideoCapture.")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
        
        self.is_running = True
        logger.info("OpenCV camera started successfully.")
        return True

    def read(self) -> Optional[np.ndarray]:
        if not self.is_running:
            return None
            
        if self.backend == "picamera2":
            try:
                # Get the BGR array directly
                return self.cap.capture_array()
            except Exception as e:
                logger.error(f"Picamera2 read error: {e}")
                return None
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def stop(self):
        if not self.is_running:
            return
            
        if self.backend == "picamera2" and self.cap:
            try:
                self.cap.stop()
                self.cap.close()
            except Exception:
                pass
        elif self.backend == "opencv" and self.cap:
            self.cap.release()
            
        self.is_running = False
        self.cap = None
        logger.info("Camera stopped.")
