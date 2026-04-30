"""
TrueVision — Telegram Handler

Server-side Telegram Bot API integration for voice commands.
Provides endpoints for direct message sending and LLM-extracted sending.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TelegramHandler:
    def __init__(self):
        # We need the bot token and chat ID to send messages
        # These are configured via environment variables or hardcoded for the system
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, text: str) -> Optional[int]:
        """
        Sends a direct text message to the configured Telegram chat.
        Returns the message ID on success, None on failure.
        """
        if not self.is_configured():
            logger.error("Telegram is not configured. Missing token or chat ID.")
            return None
            
        if not text:
            return None
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                msg_id = data.get("result", {}).get("message_id")
                logger.info(f"Telegram message sent successfully (ID: {msg_id})")
                return msg_id
            else:
                logger.error(f"Telegram API error: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Failed to reach Telegram API: {e}")
            return None
