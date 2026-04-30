"""
TrueVision — Audio Forwarder (Pi → Server)

Reads audio from the serial receiver's ring buffer and forwards it
to the server over WebSocket. Handles session lifecycle messages
and receives captions/results back.
"""

import json
import logging
import threading
import time
from typing import Callable, Optional

import websocket

from audio.serial_receiver import SerialReceiver
from connection.discovery import ServerConnection

logger = logging.getLogger(__name__)


class AudioForwarder:
    def __init__(self, serial_receiver: SerialReceiver, server_conn: ServerConnection):
        self.receiver = serial_receiver
        self.server_conn = server_conn
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.running = False
        self.is_connected = False
        
        # Audio loop thread
        self.audio_thread: Optional[threading.Thread] = None
        
        # Session state
        self.session_active = False
        self.session_key = ""
        self.last_read_idx = 0
        
        # Callbacks
        self.on_caption: Optional[Callable[[str, str], None]] = None # (text, lang)
        self.on_result: Optional[Callable[[dict], None]] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        logger.info("Started AudioForwarder worker.")

    def stop(self):
        self.running = False
        if self.session_active:
            self.end_session("", "", 140)
        if self.ws:
            self.ws.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=2.0)
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        logger.info("Stopped AudioForwarder.")

    def start_session(self, session_key: str, person_id: Optional[int], meeting_id: Optional[int]):
        """Starts a forwarding session to the server."""
        if not self.server_conn.is_available:
            logger.warning("Cannot start session: Server is not available.")
            return False
            
        server_url = self.server_conn.url
        if not server_url:
            return False
            
        ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/audio"
        
        # Connect WebSocket if needed
        if not self._connect_ws(ws_url):
            return False
            
        # Send start message
        msg = {
            "type": "session_start",
            "session_key": session_key,
            "person_id": person_id,
            "meeting_id": meeting_id
        }
        try:
            self.ws.send(json.dumps(msg))
            self.session_key = session_key
            self.session_active = True
            
            # Reset read index to current write index
            with self.receiver.buffer_lock:
                self.last_read_idx = self.receiver.write_idx
                
            logger.info(f"Forwarding session '{session_key}' started.")
            return True
        except Exception as e:
            logger.error(f"Failed to send session_start: {e}")
            self.session_active = False
            return False

    def end_session(self, previous_summary: str, person_name: str, max_chars: int):
        """Ends the current forwarding session."""
        if not self.session_active or not self.ws or not self.is_connected:
            self.session_active = False
            return
            
        msg = {
            "type": "session_end",
            "previous_summary": previous_summary,
            "person_name": person_name,
            "max_chars": max_chars
        }
        try:
            self.ws.send(json.dumps(msg))
            logger.info(f"Forwarding session '{self.session_key}' ended.")
        except Exception as e:
            logger.error(f"Failed to send session_end: {e}")
            
        self.session_active = False
        self.session_key = ""

    def _connect_ws(self, ws_url: str) -> bool:
        if self.is_connected and self.ws:
            return True
            
        if self.ws:
            self.ws.close()
            
        def on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "caption" and self.on_caption:
                    self.on_caption(data.get("text", ""), data.get("source_language", ""))
                elif msg_type == "result" and self.on_result:
                    self.on_result(data)
            except json.JSONDecodeError:
                pass

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.is_connected = False
            logger.info("WebSocket closed")

        def on_open(ws):
            self.is_connected = True
            logger.info(f"WebSocket connected to {ws_url}")

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()
        
        # Wait up to 2 seconds for connection
        for _ in range(20):
            if self.is_connected:
                return True
            time.sleep(0.1)
            
        logger.error("WebSocket connection timeout.")
        return False

    def _audio_loop(self):
        """Periodically reads new audio from the receiver and sends it as binary WS frames."""
        while self.running:
            if not self.session_active or not self.is_connected or not self.ws:
                time.sleep(0.1)
                continue
                
            with self.receiver.buffer_lock:
                curr_write_idx = self.receiver.write_idx
                cap = self.receiver.capacity_samples
                
                if curr_write_idx == self.last_read_idx:
                    # No new data
                    new_data = None
                elif curr_write_idx > self.last_read_idx:
                    new_data = self.receiver.audio_buffer[self.last_read_idx:curr_write_idx].copy()
                else:
                    # Wraparound
                    part1 = self.receiver.audio_buffer[self.last_read_idx:]
                    part2 = self.receiver.audio_buffer[:curr_write_idx]
                    new_data = np.concatenate((part1, part2))
                    
                self.last_read_idx = curr_write_idx
                
            if new_data is not None and len(new_data) > 0:
                try:
                    # Send as binary frame (int16 little endian bytes)
                    self.ws.send(new_data.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    logger.error(f"Failed to send binary audio: {e}")
                    self.is_connected = False
            
            # ~32ms intervals
            time.sleep(0.032)
