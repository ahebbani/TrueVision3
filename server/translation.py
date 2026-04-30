"""
TrueVision — Translation Support

Language detection and Whisper-based translation for live captions.
"""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Configuration
SOURCE_LANGUAGES = [lang.strip().lower() for lang in os.environ.get("TRANSLATION_SOURCE_LANGUAGES", "es,de").split(",") if lang.strip()]
TARGET_LANGUAGE = os.environ.get("TRANSLATION_TARGET_LANGUAGE", "en").lower()
MIN_PROBABILITY = float(os.environ.get("TRANSLATION_DETECTION_MIN_PROBABILITY", "0.65"))

# Readable names for HUD
LANG_MAP = {
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ru": "Russian",
    "en": "English"
}

class TranslationSession:
    def __init__(self):
        self.active_language: Optional[str] = None
        
    def update_language(self, detected_lang: str, probability: float) -> bool:
        """
        Updates the session's active language based on confidence.
        Returns True if translation should be used for this frame.
        """
        # If confidence is high enough, we lock into this language for the next few frames
        if probability >= MIN_PROBABILITY:
            if detected_lang in SOURCE_LANGUAGES:
                self.active_language = detected_lang
            else:
                # E.g. switched back to English
                self.active_language = None
                
        return self.active_language is not None

def format_caption(text: str, source_language: Optional[str]) -> str:
    """Prefixes the text with the language if translation is active (used by Pi, but server can format)."""
    # Note: The spec says the server sends source_language in the JSON and the Pi formats it.
    # We provide this utility here but we'll stick to sending the source_language field.
    if source_language:
        lang_name = LANG_MAP.get(source_language, source_language.upper())
        return f"({lang_name}) {text}"
    return text
