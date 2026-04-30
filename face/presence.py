"""
TrueVision — Presence Tracker

Per-person state machine tracking whether a person is present or absent,
with a grace period to smooth transient detection misses.
"""

import logging
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PresenceTracker:
    def __init__(self, absence_grace_period: float = 2.0):
        self.absence_grace_period = absence_grace_period
        
        # {person_id: state ('present' or 'absent')}
        self.states: Dict[int, str] = {}
        
        # {person_id: last_seen_timestamp}
        self.last_seen: Dict[int, float] = {}
        
        # Callbacks
        self.on_present: Optional[Callable[[int], None]] = None
        self.on_absent: Optional[Callable[[int], None]] = None

    def update(self, detected_person_ids: List[int]):
        """
        Called every frame with the list of recognized person IDs.
        Handles state transitions and fires callbacks.
        """
        now = time.time()
        
        # 1. Update last_seen for currently detected people
        for person_id in detected_person_ids:
            self.last_seen[person_id] = now
            
            # Transition to PRESENT
            if self.states.get(person_id) != "present":
                self.states[person_id] = "present"
                logger.info(f"Person {person_id} transitioned to PRESENT")
                if self.on_present:
                    try:
                        self.on_present(person_id)
                    except Exception as e:
                        logger.error(f"Error in on_present callback: {e}")

        # 2. Check for absences (grace period expiration)
        # We need a list of keys to avoid mutating dict while iterating
        for person_id in list(self.states.keys()):
            if self.states[person_id] == "present":
                last_time = self.last_seen.get(person_id, 0)
                if (now - last_time) > self.absence_grace_period:
                    # Transition to ABSENT
                    self.states[person_id] = "absent"
                    logger.info(f"Person {person_id} transitioned to ABSENT")
                    if self.on_absent:
                        try:
                            self.on_absent(person_id)
                        except Exception as e:
                            logger.error(f"Error in on_absent callback: {e}")

    def force_absent_all(self):
        """Forces all present people to absent immediately (e.g., mode change or exit)."""
        for person_id, state in list(self.states.items()):
            if state == "present":
                self.states[person_id] = "absent"
                if self.on_absent:
                    try:
                        self.on_absent(person_id)
                    except Exception as e:
                        logger.error(f"Error in on_absent callback: {e}")
                        
    def is_present(self, person_id: int) -> bool:
        return self.states.get(person_id) == "present"
