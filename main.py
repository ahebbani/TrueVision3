"""
TrueVision — Pi Main Entry Point

Initializes all subsystems based on config and runs the main loop.
"""

import logging
import os
import signal
import sys
import time
import uuid
from typing import Dict, Optional

import cv2

from audio.live_captioner import LiveCaptioner
from audio.recorder import ESP32SerialRecorder
from audio.serial_receiver import SerialReceiver
from audio.transcriber import LocalTranscriber
from commands.intent_router import IntentRouter
from commands.notes import NotesManager
from commands.wake_word import extract_command
from config import config
from connection.audio_forwarder import AudioForwarder
from connection.discovery import ServerConnection
from connection.remote_summarizer import remote_summarize_one_sentence
from controller.server import ControllerServer
from database.db import Database
from face.camera import Camera
from face.detector import FaceDetector
from face.presence import PresenceTracker
from face.recognizer import FaceRecognizer
from face.templates import TemplateManager
from hud.display import DisplayManager

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.INFO)
logging.getLogger("multipart").setLevel(logging.WARNING)
logger = logging.getLogger("main")

# State
running = True
current_mode = "unknown"
force_mode = config.force_mode.lower() if config.force_mode else ""

# Sessions
audio_session_key = ""
person_audio_sessions: Dict[int, str] = {} # person_id -> session_key
person_meeting_ids: Dict[int, int] = {}    # person_id -> meeting_id

# Subsystems
db: Optional[Database] = None
server_conn: Optional[ServerConnection] = None
controller: Optional[ControllerServer] = None
receiver: Optional[SerialReceiver] = None
recorder: Optional[ESP32SerialRecorder] = None
transcriber: Optional[LocalTranscriber] = None
captioner: Optional[LiveCaptioner] = None
forwarder: Optional[AudioForwarder] = None
camera: Optional[Camera] = None
detector: Optional[FaceDetector] = None
recognizer: Optional[FaceRecognizer] = None
templates: Optional[TemplateManager] = None
presence: Optional[PresenceTracker] = None
hud: Optional[DisplayManager] = None
router: Optional[IntentRouter] = None
notes_mgr: Optional[NotesManager] = None

last_caption_text = ""
last_caption_lang = ""

def signal_handler(sig, frame):
    global running
    logger.info("Graceful shutdown initiated...")
    running = False

def update_mode(new_mode: str):
    global current_mode, force_mode
    if force_mode:
        new_mode = force_mode
        
    if new_mode == current_mode:
        return
        
    logger.info(f"--- MODE CHANGED: {current_mode.upper()} -> {new_mode.upper()} ---")
    old_mode = current_mode
    current_mode = new_mode
    
    if controller:
        controller.update_state(current_mode, server_conn.is_available)
        
    # Handle Stop Events (leaving a mode)
    if old_mode in ("audio", "both") and new_mode == "face":
        stop_audio_session()
        if presence:
            presence.force_absent_all() # Stop all person sessions
    elif old_mode in ("face", "both") and new_mode == "audio":
        if presence:
            presence.force_absent_all()
            
    # Handle Start Events (entering a mode)
    if new_mode in ("audio", "both") and old_mode == "face":
        start_audio_session()

def on_esp32_mode_change(mode_byte: int):
    # 0x00 = AUDIO, 0x01 = FACE
    new_mode = "audio" if mode_byte == 0x00 else "face"
    
    # Auto-upgrade to BOTH if server is available
    if server_conn and server_conn.is_available:
        new_mode = "both"
        
    update_mode(new_mode)

def on_controller_mode_change(new_mode: str):
    if new_mode.lower() in ("face", "audio", "both"):
        update_mode(new_mode.lower())

def start_audio_session():
    global audio_session_key
    audio_session_key = str(uuid.uuid4())
    logger.info("Starting global audio session.")
    
    # Start recorder
    recorder.start(prefix="global")
    
    # If server is available, use forwarder. Otherwise, use local captioner.
    if server_conn.is_available:
        forwarder.start_session(audio_session_key, None, None)
    else:
        captioner.start()

def stop_audio_session():
    global audio_session_key, last_caption_text
    logger.info("Stopping global audio session.")
    
    audio_path = recorder.stop()
    
    if forwarder.session_active:
        forwarder.end_session("", "", 140)
        
    if captioner.running:
        captioner.stop()
        
    # Trigger final local transcription if we were offline
    if audio_path and not server_conn.is_available:
        def _transcribe_offline(path):
            logger.info("Starting background offline transcription...")
            text = transcriber.transcribe(path)
            if text:
                logger.info(f"Local Offline Transcript: {text}")
                handle_wake_words(text)
                
        import threading
        threading.Thread(target=_transcribe_offline, args=(audio_path,), daemon=True).start()
            
    audio_session_key = ""
    last_caption_text = ""
    if captioner:
        captioner.clear()

def on_person_present(person_id: int):
    # Only start person sessions if we are in BOTH mode
    if current_mode != "both":
        return
        
    logger.info(f"Starting meeting session for person {person_id}")
    session_key = str(uuid.uuid4())
    person_audio_sessions[person_id] = session_key
    
    # Create meeting record in DB
    meeting_id = db.create_meeting(person_id)
    person_meeting_ids[person_id] = meeting_id
    
    # Start recorder (if not already started by global audio)
    if not recorder.is_recording:
        recorder.start(prefix=f"person_{person_id}")
        
    # Forward to server
    if server_conn.is_available:
        forwarder.start_session(session_key, person_id, meeting_id)

def on_person_absent(person_id: int):
    if person_id not in person_audio_sessions:
        return
        
    logger.info(f"Stopping meeting session for person {person_id}")
    session_key = person_audio_sessions.pop(person_id)
    meeting_id = person_meeting_ids.pop(person_id, None)
    
    audio_path = recorder.stop()
    
    if meeting_id and audio_path:
        db.update_meeting(meeting_id, audio_path=audio_path)
    
    if server_conn.is_available and forwarder.session_active:
        # Get previous summary
        prev_summary = db.get_latest_summary(person_id) or ""
        faces = db.get_faces()
        person_name = next((f[1] for f in faces if f[0] == person_id), "Unknown")
        
        forwarder.end_session(prev_summary, person_name, 140)
    elif audio_path and not server_conn.is_available:
        # Offline processing
        logger.info("Processing meeting offline...")
        
        def _process_offline(path, p_id, m_id):
            text = transcriber.transcribe(path)
            if text:
                handle_wake_words(text)
                
                prev = db.get_latest_summary(p_id) or ""
                faces = db.get_faces()
                name = next((f[1] for f in faces if f[0] == p_id), "Unknown")
                
                summary = remote_summarize_one_sentence(server_conn, text, prev, name, 140)
                
                if m_id:
                    db.update_meeting(m_id, transcript=text, summary=summary)
                    
        import threading
        threading.Thread(target=_process_offline, args=(audio_path, person_id, meeting_id), daemon=True).start()
                
    # Restart global audio if we are still in BOTH or AUDIO mode
    if current_mode in ("audio", "both") and not recorder.is_recording:
        start_audio_session()

def handle_wake_words(transcript: str):
    cmd = extract_command(transcript)
    if cmd:
        router.process_command(cmd)

def on_live_caption(text: str, lang: str):
    global last_caption_text, last_caption_lang
    last_caption_text = text
    last_caption_lang = lang

def on_server_result(data: dict):
    # Triggered when server sends final result
    meeting_id = data.get("meeting_id")
    transcript = data.get("transcript")
    summary = data.get("summary")
    
    if meeting_id:
        db.update_meeting(meeting_id, transcript=transcript, summary=summary)
        
    if transcript:
        handle_wake_words(transcript)

def on_phone_enroll(name: str) -> bool:
    """Triggered by phone app to enroll the largest face in the current frame."""
    if not camera.is_running:
        return False
        
    frame = camera.read()
    if frame is None:
        return False
        
    rects = detector.detect(frame)
    if not rects:
        return False
        
    # Get largest face
    largest = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    emb = recognizer.compute_embedding(frame, largest)
    qual = recognizer.calculate_quality(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), largest)
    
    if emb is not None:
        pid = db.add_face(name, emb)
        templates.evaluate_and_collect(pid, emb, qual)
        hud.show_toast(f"✓ Saved: {name}")
        return True
    return False

def on_voice_enroll(name: str):
    """Triggered by voice command to enroll the single visible face."""
    hud.show_toast(f"Enrolling face as: {name}...")
    on_phone_enroll(name)

def setup_subsystems():
    global db, server_conn, controller, receiver, recorder, transcriber
    global captioner, forwarder, camera, detector, recognizer, templates
    global presence, hud, router, notes_mgr

    logger.info("Initializing subsystems...")
    
    db = Database()
    
    server_conn = ServerConnection(config.server_url)
    server_conn.start()
    
    receiver = SerialReceiver.get_instance(config.serial_port, config.baud_rate)
    receiver.on_mode_change = on_esp32_mode_change
    receiver.start()
    
    recorder = ESP32SerialRecorder(config.serial_port, config.baud_rate)
    
    transcriber = LocalTranscriber(config.whisper_model, config.whisper_device, config.whisper_compute_type)
    captioner = LiveCaptioner(recorder, transcriber, config.caption_interval_sec, config.caption_window_sec)
    captioner.on_caption = lambda t: on_live_caption(t, "")
    
    forwarder = AudioForwarder(receiver, server_conn)
    forwarder.on_caption = on_live_caption
    forwarder.on_result = on_server_result
    forwarder.start()
    
    camera = Camera()
    detector = FaceDetector(config.face_detector)
    recognizer = FaceRecognizer()
    templates = TemplateManager(db)
    templates.load_all_templates()
    
    presence = PresenceTracker()
    presence.on_present = on_person_present
    presence.on_absent = on_person_absent
    
    hud = DisplayManager()
    
    controller = ControllerServer(config.controller_port, db)
    controller.on_mode_change = on_controller_mode_change
    controller.on_enroll_face = on_phone_enroll
    controller.start()
    
    router = IntentRouter(server_conn)
    notes_mgr = NotesManager(db)
    notes_mgr.on_toast = hud.show_toast
    
    router.on_note = notes_mgr.add_note
    router.on_enroll = on_voice_enroll

def main():
    global current_mode, last_caption_text
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    setup_subsystems()
    
    # Determine initial mode
    update_mode("face")
    
    hud.start()
    camera.start()
    
    logger.info("Entering main loop...")
    
    while running:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue
            
        faces_info = []
        detected_pids = []
        
        # Face Recognition
        if current_mode in ("face", "both"):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            rects = detector.detect(frame)
            
            all_known = templates.get_all_templates()
            
            for rect in rects:
                emb = recognizer.compute_embedding(frame, rect)
                qual = recognizer.calculate_quality(gray, rect)
                
                name = "Unknown"
                seen = 0
                last = ""
                summary = ""
                is_rec = False
                
                if emb is not None:
                    match = recognizer.match(emb, all_known)
                    if match:
                        pid, dist = match
                        detected_pids.append(pid)
                        
                        db.update_face_seen(pid)
                        templates.evaluate_and_collect(pid, emb, qual)
                        
                        # Lookup info
                        faces = db.get_faces()
                        f_info = next((f for f in faces if f[0] == pid), None)
                        if f_info:
                            name = f_info[1]
                            seen = f_info[3]
                            
                        # Format last seen
                        last = "just now"
                        
                        # Summary
                        summary = db.get_latest_summary(pid) or ""
                        
                        # REC
                        is_rec = recorder.is_recording and pid in person_audio_sessions
                
                faces_info.append({
                    "rect": rect,
                    "name": name,
                    "seen_count": seen,
                    "last_seen_ago": last,
                    "summary": summary,
                    "is_recording": is_rec
                })
                
        # Update presence
        presence.update(detected_pids)
        
        # Update controller status
        controller.update_state(current_mode, server_conn.is_available)
        
        # Notes
        active_notes = notes_mgr.get_active_notes()
        
        # Render HUD
        # If speed dial launched, skip rendering HUD
        if not controller.is_content_launched:
            cap_text = last_caption_text if current_mode in ("audio", "both") else ""
            keep_running = hud.render_frame(
                mode=current_mode,
                server_available=server_conn.is_available,
                faces_info=faces_info,
                caption_text=cap_text,
                source_language=last_caption_lang,
                reminders=active_notes
            )
            if not keep_running:
                break
        else:
            # Let the browser render on HDMI
            cv2.waitKey(33)

    # Cleanup
    logger.info("Cleaning up...")
    update_mode("unknown") # Stops all sessions
    hud.stop()
    camera.stop()
    controller.stop()
    forwarder.stop()
    captioner.stop()
    receiver.stop()
    server_conn.stop()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    main()
