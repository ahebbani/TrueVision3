"""
TrueVision — Face Recognizer

Uses dlib's 128-dimensional face embedding model for face recognition.
Matches live embeddings against stored templates using L2 distance.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(self, models_dir: str = "models", match_threshold: float = 0.6):
        self.models_dir = Path(models_dir)
        self.match_threshold = match_threshold
        
        self.shape_predictor = None
        self.face_rec_model = None
        
        self._load_models()

    def _load_models(self):
        shape_path = self.models_dir / "shape_predictor_68_face_landmarks.dat"
        rec_path = self.models_dir / "dlib_face_recognition_resnet_model_v1.dat"
        
        if not shape_path.exists() or not rec_path.exists():
            logger.error("dlib models missing. Please run setup script.")
            return
            
        logger.info("Loading dlib shape predictor...")
        self.shape_predictor = dlib.shape_predictor(str(shape_path))
        
        logger.info("Loading dlib face recognition model...")
        self.face_rec_model = dlib.face_recognition_model_v1(str(rec_path))

    def compute_embedding(self, rgb_frame, face_rect: dlib.rectangle) -> Optional[np.ndarray]:
        """
        Compute the 128-dimensional embedding for a given face rectangle.
        """
        if self.shape_predictor is None or self.face_rec_model is None:
            return None
            
        try:
            # 1. Get facial landmarks
            shape = self.shape_predictor(rgb_frame, face_rect)
            
            # 2. Compute the 128D descriptor
            # Using 1 jitter for speed. Higher values are more accurate but slower.
            descriptor = self.face_rec_model.compute_face_descriptor(rgb_frame, shape, 1)
            
            return np.array(descriptor, dtype=np.float64)
        except Exception as e:
            logger.error(f"Failed to compute embedding: {e}")
            return None

    def match(self, live_embedding: np.ndarray, known_templates: Dict[int, List[np.ndarray]]) -> Optional[Tuple[int, float]]:
        """
        Matches a live embedding against all known templates.
        known_templates: {person_id: [embedding1, embedding2, ...]}
        Returns (best_person_id, distance) or None if no match below threshold.
        """
        best_id = None
        best_dist = float('inf')
        
        for person_id, templates in known_templates.items():
            if not templates:
                continue
                
            # L2 distance against all templates for this person
            distances = np.linalg.norm(templates - live_embedding, axis=1)
            min_dist = float(np.min(distances))
            
            if min_dist < best_dist:
                best_dist = min_dist
                best_id = person_id
                
        if best_dist <= self.match_threshold:
            return best_id, best_dist
            
        return None

    @staticmethod
    def calculate_quality(gray_frame, face_rect: dlib.rectangle) -> float:
        """
        Calculates a quality score for the face crop using Laplacian variance (blur detection).
        Higher is better.
        """
        # Clamp rect to frame bounds
        h, w = gray_frame.shape
        x1 = max(0, face_rect.left())
        y1 = max(0, face_rect.top())
        x2 = min(w, face_rect.right())
        y2 = min(h, face_rect.bottom())
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0 # Too small
            
        face_crop = gray_frame[y1:y2, x1:x2]
        
        # Laplacian variance measures edge sharpness
        variance = cv2.Laplacian(face_crop, cv2.CV_64F).var()
        return float(variance)
