"""
TrueVision — Phone Controller HTTP Server

Lightweight HTTP server that serves a single-page web app for
phone-based control of the TrueVision system.

Endpoints:
  GET  /                    — Serve the controller page
  GET  /api/status          — Current mode and server state
  POST /api/mode            — Change mode (FACE, AUDIO, BOTH)
  POST /api/launch          — Launch speed-dial content
  POST /api/close           — Dismiss launched content
  POST /api/enroll          — Add a new face
  GET  /api/notes           — List active notes
  POST /api/notes           — Add a note manually
  POST /api/notes/<id>/done — Mark note as done
  DELETE /api/notes/<id>    — Delete note
"""

import json
import logging
import os
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Callable, Optional

from database.db import Database

logger = logging.getLogger(__name__)


class ControllerServer:
    def __init__(self, port: int, db: Database):
        self.port = port
        self.db = db
        self.static_dir = Path(__file__).parent / "static"
        
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        
        # Shared State (read-only for controller, updated by main loop)
        self.current_mode = "unknown"
        self.server_available = False
        self.is_content_launched = False
        self.launched_process: Optional[subprocess.Popen] = None
        
        # Callbacks
        self.on_mode_change: Optional[Callable[[str], None]] = None
        self.on_enroll_face: Optional[Callable[[str], bool]] = None
        
    def update_state(self, mode: str, server_available: bool):
        self.current_mode = mode
        self.server_available = server_available

    def start(self):
        if self.running:
            return
            
        handler = self._create_handler()
        self.server = HTTPServer(("0.0.0.0", self.port), handler)
        
        self.running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Phone controller started at http://0.0.0.0:{self.port}")

    def stop(self):
        self.running = False
        if self.launched_process:
            self._close_content()
            
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Phone controller stopped.")

    def _launch_content(self, action: str):
        self._close_content() # Ensure old one is closed
        
        urls = {
            "news": "https://news.google.com",
            "weather": "https://weather.com",
            "instagram": "https://instagram.com",
            "youtube": "https://youtube.com",
            "database": f"http://localhost:{self.port}/db_report" # A placeholder for a real report
        }
        
        url = urls.get(action)
        if url:
            try:
                # Launch chromium in kiosk mode
                self.launched_process = subprocess.Popen(
                    ["chromium-browser", "--kiosk", "--app=" + url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.is_content_launched = True
                logger.info(f"Launched {action} at {url}")
            except Exception as e:
                logger.error(f"Failed to launch browser: {e}")

    def _close_content(self):
        if self.launched_process:
            try:
                self.launched_process.terminate()
                self.launched_process.wait(timeout=2.0)
            except Exception:
                try:
                    self.launched_process.kill()
                except:
                    pass
            self.launched_process = None
            self.is_content_launched = False
            logger.info("Closed launched content.")

    def _create_handler(self):
        controller = self
        
        class APIHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(controller.static_dir), **kwargs)

            def log_message(self, format, *args):
                # Suppress normal HTTP logs to keep console clean
                pass

            def send_json(self, data, status=200):
                self.send_response(status)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

            def do_GET(self):
                if self.path == "/api/status":
                    self.send_json({
                        "mode": controller.current_mode,
                        "server_available": controller.server_available,
                        "content_launched": controller.is_content_launched
                    })
                elif self.path == "/api/notes":
                    notes = controller.db.get_active_notes()
                    # Format: [{"id": 1, "text": "Buy milk"}, ...]
                    resp = [{"id": n[0], "text": n[1]} for n in notes]
                    self.send_json(resp)
                elif self.path == "/db_report":
                    # Simple text report for the "Database" speed dial
                    self.send_response(200)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Database Report Placeholder. Run 'make db' for the full HTML report.")
                else:
                    # Serve static files
                    super().do_GET()

            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                
                try:
                    data = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    data = {}

                if self.path == "/api/mode":
                    new_mode = data.get("mode")
                    if new_mode and controller.on_mode_change:
                        controller.on_mode_change(new_mode)
                    self.send_json({"status": "ok"})
                    
                elif self.path == "/api/launch":
                    action = data.get("action")
                    if action:
                        controller._launch_content(action)
                    self.send_json({"status": "ok"})
                    
                elif self.path == "/api/close":
                    controller._close_content()
                    self.send_json({"status": "ok"})
                    
                elif self.path == "/api/enroll":
                    name = data.get("name")
                    if name and controller.on_enroll_face:
                        success = controller.on_enroll_face(name)
                        self.send_json({"status": "ok" if success else "error"})
                    else:
                        self.send_json({"status": "error", "message": "Invalid name or no handler"})
                        
                elif self.path == "/api/notes":
                    text = data.get("text")
                    if text:
                        note_id = controller.db.add_note(text)
                        self.send_json({"status": "ok", "id": note_id})
                    else:
                        self.send_json({"status": "error"})
                        
                elif self.path.startswith("/api/notes/") and self.path.endswith("/done"):
                    note_id = int(self.path.split("/")[3])
                    controller.db.mark_note_done(note_id)
                    self.send_json({"status": "ok"})
                    
                else:
                    self.send_error(404)

            def do_DELETE(self):
                if self.path.startswith("/api/notes/"):
                    try:
                        note_id = int(self.path.split("/")[3])
                        controller.db.delete_note(note_id)
                        self.send_json({"status": "ok"})
                    except ValueError:
                        self.send_error(400)
                else:
                    self.send_error(404)

        return APIHandler
