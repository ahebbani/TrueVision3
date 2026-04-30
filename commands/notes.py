"""
TrueVision — Voice Notes & Reminders

Handles saving voice notes/reminders to the database and
triggering HUD toast confirmations.
"""

import logging
from typing import Callable, Optional

from database.db import Database

logger = logging.getLogger(__name__)


class NotesManager:
    def __init__(self, db: Database):
        self.db = db
        self.on_toast: Optional[Callable[[str], None]] = None

    def add_note(self, text: str):
        """Clean up the note text, save it, and trigger a toast."""
        # Strip common wake-word prefixes if they somehow slipped through
        clean_text = text.strip()
        lower_text = clean_text.lower()
        
        if lower_text.startswith("to "):
            clean_text = clean_text[3:]
        if lower_text.startswith("that "):
            clean_text = clean_text[5:]
            
        # Capitalize first letter
        if clean_text:
            clean_text = clean_text[0].upper() + clean_text[1:]
            
        note_id = self.db.add_note(clean_text)
        logger.info(f"Saved note {note_id}: {clean_text}")
        
        if self.on_toast:
            # Shorten for toast if too long
            toast_text = clean_text
            if len(toast_text) > 40:
                toast_text = toast_text[:37] + "..."
            self.on_toast(f"✓ Reminder saved: {toast_text}")

    def get_active_notes(self) -> list[str]:
        """Returns the text of active (not done) notes, for the HUD."""
        notes = self.db.get_active_notes()
        return [n[1] for n in notes]

    def mark_done(self, note_id: int):
        self.db.mark_note_done(note_id)
        
    def delete(self, note_id: int):
        self.db.delete_note(note_id)
