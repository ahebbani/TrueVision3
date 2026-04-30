"""
TrueVision — Server-side Ollama LLM Summarizer

Communicates with a local Ollama instance to generate one-sentence
memory-cue summaries from meeting transcripts.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaSummarizer:
    def __init__(self):
        self.url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        
    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                # Check if model exists
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                if any(m.startswith(self.model) for m in models):
                    return True
                logger.warning(f"Ollama is running, but model '{self.model}' is not pulled.")
            return False
        except requests.RequestException:
            return False

    def summarize(self, transcript: str, previous_summary: str = "", person_name: str = "", max_chars: int = 140) -> Optional[str]:
        if not transcript:
            return None
            
        prompt = self._build_prompt(transcript, previous_summary, person_name, max_chars)
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 80
                }
            }
            resp = requests.post(f"{self.url}/api/generate", json=payload, timeout=30.0)
            
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("response", "").strip()
                # Ensure it fits
                if len(summary) > max_chars:
                    summary = summary[:max_chars-3] + "..."
                return summary
            else:
                logger.error(f"Ollama summarization failed: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Ollama summarization error: {e}")
            return None

    def _build_prompt(self, transcript: str, previous_summary: str, person_name: str, max_chars: int) -> str:
        ctx = ""
        if person_name:
            ctx += f"We are talking to {person_name}. "
        if previous_summary:
            ctx += f"Previously we discussed: {previous_summary}\n"
            
        return f"""You are an assistant for an AR smart-glasses system. Your job is to read a conversation transcript and output EXACTLY ONE sentence summarizing the main topic or most important takeaway.
Requirements:
1. Exactly one sentence.
2. Less than {max_chars} characters.
3. Prefer concrete topics (e.g. "Discussed the Q3 budget deficit.") over filler (e.g. "They said hello.").
4. NO quotes, NO bullet points, NO intro text. Just the sentence.

Context: {ctx}
Transcript: "{transcript}"

One-sentence summary:"""

    def extract_telegram_message(self, raw_transcript: str) -> Optional[str]:
        """Extracts the clean intended Telegram message from a noisy voice command transcript."""
        prompt = f"""Extract the exact message the user wants to send via Telegram from the following noisy voice transcript. 
Remove any filler words, wake words, or routing instructions like "send a telegram saying" or "tell everyone".
Return ONLY the clean message text, with no quotes or explanation.

Transcript: "{raw_transcript}"
Clean Message:"""

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 100
                }
            }
            resp = requests.post(f"{self.url}/api/generate", json=payload, timeout=15.0)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip().strip('"\'')
            return None
        except Exception as e:
            logger.error(f"Ollama extraction error: {e}")
            return None
