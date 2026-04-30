"""
TrueVision — Adaptive Embedding Template Management

Continuously collects new face embeddings for recognized people
to improve accuracy over time, with bootstrap and steady-state phases.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from database.db import Database

logger = logging.getLogger(__name__)


class TemplateManager:
    def __init__(self, db: Database):
        self.db = db
        # {person_id: [embedding1, embedding2, ...]}
        self.templates_cache: Dict[int, List[np.ndarray]] = {}
        # {person_id: last_collection_time}
        self.last_collection: Dict[int, float] = {}
        
        # Phase settings
        self.MAX_TEMPLATES_PER_PERSON = 30
        
        # Bootstrap phase (< 5 templates)
        self.BOOTSTRAP_THRESHOLD = 5
        self.BOOTSTRAP_QUALITY = 60.0
        self.BOOTSTRAP_COOLDOWN = 0.75
        self.BOOTSTRAP_DIVERSITY_L2 = 0.08
        
        # Steady-state phase (>= 5 templates)
        self.STEADY_QUALITY = 120.0
        self.STEADY_COOLDOWN = 5.0
        self.STEADY_DIVERSITY_L2 = 0.20

    def load_all_templates(self):
        """Loads all templates from DB into memory."""
        self.templates_cache.clear()
        faces = self.db.get_faces()
        
        total_templates = 0
        for face_id, _, _, _ in faces:
            db_embs = self.db.get_embeddings(face_id)
            if db_embs:
                self.templates_cache[face_id] = [emb for _, emb, _ in db_embs]
                total_templates += len(db_embs)
            else:
                self.templates_cache[face_id] = []
                
        logger.info(f"Loaded {total_templates} templates for {len(faces)} faces.")

    def get_all_templates(self) -> Dict[int, List[np.ndarray]]:
        return self.templates_cache

    def evaluate_and_collect(self, person_id: int, embedding: np.ndarray, quality: float) -> bool:
        """
        Evaluates a live embedding for collection.
        Returns True if the template was collected and saved.
        """
        now = time.time()
        last_time = self.last_collection.get(person_id, 0)
        
        current_templates = self.templates_cache.get(person_id, [])
        num_templates = len(current_templates)
        
        # 1. Determine Phase Thresholds
        if num_templates < self.BOOTSTRAP_THRESHOLD:
            # Bootstrap
            req_quality = self.BOOTSTRAP_QUALITY
            req_cooldown = self.BOOTSTRAP_COOLDOWN
            req_diversity = self.BOOTSTRAP_DIVERSITY_L2 if num_templates >= 3 else 0.0 # Force first 3
        else:
            # Steady-state
            req_quality = self.STEADY_QUALITY
            req_cooldown = self.STEADY_COOLDOWN
            req_diversity = self.STEADY_DIVERSITY_L2
            
        # 2. Check Cooldown
        if (now - last_time) < req_cooldown:
            return False
            
        # 3. Check Quality
        if quality < req_quality:
            return False
            
        # 4. Check Diversity (L2 distance to nearest existing template)
        if current_templates and req_diversity > 0.0:
            distances = np.linalg.norm(current_templates - embedding, axis=1)
            min_dist = float(np.min(distances))
            if min_dist < req_diversity:
                return False # Too similar to an existing template
                
        # 5. Collect!
        self._add_template(person_id, embedding, quality)
        self.last_collection[person_id] = now
        
        # 6. Prune if needed
        if len(self.templates_cache[person_id]) > self.MAX_TEMPLATES_PER_PERSON:
            self._prune_templates(person_id)
            
        return True

    def _add_template(self, person_id: int, embedding: np.ndarray, quality: float):
        # Save to DB
        self.db.add_embedding(person_id, embedding, quality)
        
        # Update Cache
        if person_id not in self.templates_cache:
            self.templates_cache[person_id] = []
        self.templates_cache[person_id].append(embedding)
        
        logger.debug(f"Collected new template for person {person_id}. Total: {len(self.templates_cache[person_id])}")

    def _prune_templates(self, person_id: int):
        """Removes the lowest quality templates to keep the count within max limits."""
        db_embs = self.db.get_embeddings(person_id) # Returns (id, emb, quality)
        if len(db_embs) <= self.MAX_TEMPLATES_PER_PERSON:
            return
            
        # Sort by quality ascending
        db_embs.sort(key=lambda x: x[2])
        
        # IDs to delete
        num_to_delete = len(db_embs) - self.MAX_TEMPLATES_PER_PERSON
        to_delete_ids = [x[0] for x in db_embs[:num_to_delete]]
        
        self.db.delete_embeddings(to_delete_ids)
        
        # Reload cache for this person
        new_embs = self.db.get_embeddings(person_id)
        self.templates_cache[person_id] = [emb for _, emb, _ in new_embs]
        
        logger.info(f"Pruned {num_to_delete} templates for person {person_id}.")
