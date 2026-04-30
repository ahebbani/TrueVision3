"""
TrueVision — Face Detector

dlib-based face detection with two modes:
  - HOG (fast, CPU-friendly) — default on ARM/Pi
  - CNN (more accurate) — default on x86 with GPU
"""

import logging
import platform
from pathlib import Path
from typing import List

import cv2
import dlib

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, mode: str = "auto", models_dir: str = "models"):
        self.mode = mode
        self.models_dir = Path(models_dir)
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if self.mode == "auto":
            if platform.machine() in ("armv7l", "aarch64"):
                self.mode = "hog"
            else:
                self.mode = "cnn"

        if self.mode == "hog":
            logger.info("Initializing dlib HOG face detector...")
            self.detector = dlib.get_frontal_face_detector()
        else:
            logger.info("Initializing dlib CNN face detector...")
            cnn_model_path = self.models_dir / "mmod_human_face_detector.dat"
            if not cnn_model_path.exists():
                logger.warning(f"CNN model missing at {cnn_model_path}. Falling back to HOG.")
                self.mode = "hog"
                self.detector = dlib.get_frontal_face_detector()
            else:
                self.detector = dlib.cnn_face_detection_model_v1(str(cnn_model_path))

    def detect(self, rgb_frame) -> List[dlib.rectangle]:
        """
        Detect faces in an RGB frame.
        Returns a list of dlib.rectangle objects.
        """
        if self.detector is None:
            return []

        # Convert to grayscale for faster detection if using HOG
        if self.mode == "hog":
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            dets = self.detector(gray, 1) # 1 upsample
        else:
            # CNN expects RGB
            dets = self.detector(rgb_frame, 1)

        rects = []
        for det in dets:
            if self.mode == "cnn":
                # CNN returns mmod_rectangles, we extract the underlying rect
                rects.append(det.rect)
            else:
                rects.append(det)

        return rects
