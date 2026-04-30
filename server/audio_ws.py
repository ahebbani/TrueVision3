"""
TrueVision — WebSocket Audio Endpoint

Handles the /ws/audio WebSocket connection from the Pi.
Receives binary audio frames and JSON session control messages.
Provides live captioning with translation and final transcription.
"""

import asyncio
import json
import logging
import tempfile
import os
from typing import Dict, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.summarizer import OllamaSummarizer
from server.translation import TranslationSession
from server.whisper_service import ServerWhisperService

logger = logging.getLogger(__name__)

router = APIRouter()

# Globals initialized by app startup
whisper_service: Optional[ServerWhisperService] = None
summarizer: Optional[OllamaSummarizer] = None


class AudioSession:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        
        # Audio buffer (list of bytes)
        self.audio_chunks = []
        
        # Metadata
        self.session_key = ""
        self.person_id: Optional[int] = None
        self.meeting_id: Optional[int] = None
        
        # State
        self.is_active = False
        self.translation_session = TranslationSession()
        
        # Tasks
        self.live_task: Optional[asyncio.Task] = None

    def add_chunk(self, chunk: bytes):
        if self.is_active:
            self.audio_chunks.append(chunk)

    def get_all_audio(self) -> bytes:
        return b"".join(self.audio_chunks)

    def get_recent_audio(self, seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
        """Returns the last N seconds of audio bytes."""
        bytes_needed = int(seconds * sample_rate * 2) # 16-bit
        
        total_bytes = sum(len(c) for c in self.audio_chunks)
        if total_bytes == 0:
            return b""
            
        if total_bytes <= bytes_needed:
            return self.get_all_audio()
            
        # Collect chunks from the end
        result = []
        collected = 0
        for chunk in reversed(self.audio_chunks):
            if collected + len(chunk) <= bytes_needed:
                result.append(chunk)
                collected += len(chunk)
            else:
                needed = bytes_needed - collected
                result.append(chunk[-needed:])
                break
                
        result.reverse()
        return b"".join(result)

    def clear(self):
        self.audio_chunks.clear()
        self.translation_session = TranslationSession()


# Store active sessions by websocket
active_sessions: Dict[WebSocket, AudioSession] = {}


async def live_captioning_loop(session: AudioSession):
    """Periodically transcribes the recent audio window."""
    try:
        while session.is_active:
            await asyncio.sleep(0.7)
            
            recent_bytes = session.get_recent_audio(seconds=2.0)
            if len(recent_bytes) < 16000: # Need at least 0.5s (16000 bytes)
                continue
                
            # Write to temp file
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            try:
                import soundfile as sf
                samples = np.frombuffer(recent_bytes, dtype=np.int16)
                sf.write(temp_path, samples, 16000)
                
                # Transcribe
                # If we have an active translation language locked in, force Whisper to use it
                force_lang = session.translation_session.active_language
                
                # We need to run synchronous Whisper in a threadpool to not block asyncio
                loop = asyncio.get_event_loop()
                text, detected_lang, prob = await loop.run_in_executor(
                    None, 
                    whisper_service.transcribe_live, 
                    temp_path, 
                    force_lang
                )
                
                if text:
                    # Update language detection state
                    is_translating = session.translation_session.update_language(detected_lang, prob)
                    
                    # If translation is active, we need a second pass to translate
                    if is_translating and session.translation_session.active_language:
                        text = await loop.run_in_executor(
                            None,
                            whisper_service.translate,
                            temp_path,
                            session.translation_session.active_language
                        )
                        
                    if text:
                        # Send caption to Pi
                        msg = {
                            "type": "caption",
                            "text": text,
                            "session_key": session.session_key,
                            "source_language": session.translation_session.active_language if is_translating else None
                        }
                        await session.ws.send_json(msg)
                        
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Live captioning loop error: {e}")


async def process_session_end(session: AudioSession, previous_summary: str, person_name: str, max_chars: int):
    """Processes the full audio buffer for final transcription and summarization."""
    audio_bytes = session.get_all_audio()
    if len(audio_bytes) < 16000: # < 0.5s
        logger.warning(f"Session {session.session_key} ended with insufficient audio.")
        return
        
    logger.info(f"Processing final transcription for {session.session_key}...")
    
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    try:
        import soundfile as sf
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        sf.write(temp_path, samples, 16000)
        
        loop = asyncio.get_event_loop()
        
        # 1. Final Transcribe
        transcript = await loop.run_in_executor(None, whisper_service.transcribe, temp_path)
        
        if not transcript:
            logger.warning(f"No transcription produced for {session.session_key}.")
            return
            
        logger.info(f"Final transcript ({session.session_key}): {transcript}")
        
        # 2. Summarize
        summary = ""
        if summarizer:
            summary = await loop.run_in_executor(
                None, 
                summarizer.summarize, 
                transcript, 
                previous_summary, 
                person_name, 
                max_chars
            )
            
        # 3. Send Result
        msg = {
            "type": "result",
            "session_key": session.session_key,
            "meeting_id": session.meeting_id,
            "transcript": transcript,
            "summary": summary
        }
        
        # The websocket might be closed, so handle it gracefully
        try:
            await session.ws.send_json(msg)
            logger.info(f"Result sent for {session.session_key}")
        except Exception:
            logger.info(f"Could not send result for {session.session_key} (WebSocket closed). Result was: {summary}")
            
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


@router.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    session = AudioSession(websocket)
    active_sessions[websocket] = session
    
    logger.info("New WebSocket connection accepted.")
    
    try:
        while True:
            # We must handle both text (JSON control) and binary (audio) frames
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio frame
                session.add_chunk(message["bytes"])
                
            elif "text" in message:
                # JSON control frame
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "session_start":
                        session.session_key = data.get("session_key", "")
                        session.person_id = data.get("person_id")
                        session.meeting_id = data.get("meeting_id")
                        session.is_active = True
                        session.clear()
                        
                        # Start live captioning worker
                        if session.live_task:
                            session.live_task.cancel()
                        session.live_task = asyncio.create_task(live_captioning_loop(session))
                        
                        logger.info(f"Started session {session.session_key}")
                        
                    elif msg_type == "session_end":
                        session.is_active = False
                        if session.live_task:
                            session.live_task.cancel()
                            
                        # Process final results asynchronously so we don't block the WS
                        prev_sum = data.get("previous_summary", "")
                        p_name = data.get("person_name", "")
                        m_chars = data.get("max_chars", 140)
                        
                        asyncio.create_task(process_session_end(session, prev_sum, p_name, m_chars))
                        
                except json.JSONDecodeError:
                    pass
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        session.is_active = False
        if session.live_task:
            session.live_task.cancel()
        if websocket in active_sessions:
            del active_sessions[websocket]
