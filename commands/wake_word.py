"""
TrueVision — Wake Word Detection

Scans transcripts for the wake words "assistant" or "truevision"
and extracts the command text after the wake word.
"""

import re
from typing import Optional, Tuple

# Wake words (case-insensitive)
WAKE_WORDS = [r"\bassistant\b", r"\btruevision\b", r"\btrue vision\b"]

def extract_command(transcript: str) -> Optional[str]:
    """
    Looks for a wake word in the transcript.
    Returns the string following the wake word, or None if no wake word is found.
    """
    if not transcript:
        return None
        
    lower_transcript = transcript.lower()
    
    # Find the earliest wake word match
    best_match_idx = -1
    best_match_end = -1
    
    for pattern in WAKE_WORDS:
        match = re.search(pattern, lower_transcript)
        if match:
            if best_match_idx == -1 or match.start() < best_match_idx:
                best_match_idx = match.start()
                best_match_end = match.end()
                
    if best_match_idx != -1:
        # Extract everything after the wake word
        # (preserve original case from the original transcript)
        command_text = transcript[best_match_end:].strip()
        # Clean up leading punctuation
        command_text = re.sub(r'^[.,!?:\s]+', '', command_text)
        return command_text
        
    return None
