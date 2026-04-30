"""
TrueVision — Remote Summarizer Client

Calls the server's POST /summarize endpoint to get an LLM-generated
one-sentence summary. Falls back to the local extractive summarizer.
"""

import logging
from typing import Optional

import requests

from audio.summarizer import summarize_one_sentence as local_summarize
from connection.discovery import ServerConnection

logger = logging.getLogger(__name__)


def remote_summarize_one_sentence(
    server_conn: ServerConnection,
    transcript: str,
    previous_summary: str = "",
    person_name: str = "",
    max_chars: int = 140
) -> str:
    """
    Attempts to use the server LLM for summarization.
    Falls back to local summarization if the server is down or fails.
    """
    if not transcript:
        return ""
        
    if server_conn.is_available and server_conn.url:
        try:
            payload = {
                "transcript": transcript,
                "previous_summary": previous_summary,
                "person_name": person_name,
                "max_chars": max_chars
            }
            resp = requests.post(f"{server_conn.url}/summarize", json=payload, timeout=35.0)
            
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("summary", "")
                if summary:
                    return summary
            else:
                logger.warning(f"Server summarization failed with status {resp.status_code}. Using local fallback.")
        except Exception as e:
            logger.warning(f"Server summarization request failed: {e}. Using local fallback.")
            
    # Local fallback
    return local_summarize(transcript, max_chars)
