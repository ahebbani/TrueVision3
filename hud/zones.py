"""
TrueVision — HUD Zone Renderers

Individual rendering functions for each zone of the AR heads-up display.
All functions draw on a black OpenCV frame using cv2 primitives.
"""

import logging
import platform
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from hud.toasts import ToastManager

logger = logging.getLogger(__name__)


def draw_text_with_bg(frame, text: str, pos: tuple, font=cv2.FONT_HERSHEY_SIMPLEX, 
                      scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0, 128), thickness=1):
    """Draw text with a semi-transparent background box."""
    # We can't do true alpha blending easily with just cv2 without a full copy,
    # so we'll do a simple dark filled rectangle as "semi-transparent" enough for OLED
    # For OLED, just drawing a black rect is fully transparent. If we want a dark gray bg:
    if bg_color[3] > 0:
        (t_w, t_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        # bg_color is (B, G, R, A)
        b, g, r = bg_color[:3]
        cv2.rectangle(frame, (x, y - t_h - 4), (x + t_w + 4, y + baseline + 2), (b, g, r), -1)
        
    cv2.putText(frame, text, pos, font, scale, text_color, thickness, cv2.LINE_AA)
    return pos[1] + cv2.getTextSize(text, font, scale, thickness)[0][1] # approx height advance


def render_clock_date(frame, x: int = 20, y: int = 40):
    now = datetime.now()
    
    # Time: 11:42 AM
    time_str = now.strftime("%I:%M %p").lstrip("0")
    cv2.putText(frame, time_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Date: Wed, Apr 30
    date_str = now.strftime("%a, %b %d")
    cv2.putText(frame, date_str, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)


def _get_cpu_temp() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000.0
        return temp
    except:
        return 0.0

def _get_wifi_signal() -> str:
    try:
        # Very simple iwconfig parse for Pi
        output = subprocess.check_output(["iwconfig", "wlan0"], stderr=subprocess.DEVNULL).decode('utf-8')
        if "Signal level=" in output:
            idx = output.find("Signal level=") + 13
            val = output[idx:idx+3].strip()
            return f"{val} dBm"
    except:
        pass
    return "N/A"

def render_system_status(frame, server_available: bool, mode: str, width: int):
    x = width - 250
    y = 30
    
    # CPU Temp
    temp = _get_cpu_temp()
    temp_color = (0, 255, 0) # Green
    if temp > 75: temp_color = (0, 0, 255) # Red
    elif temp > 60: temp_color = (0, 255, 255) # Yellow
    
    cv2.putText(frame, f"CPU {temp:.1f}C", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_color, 1, cv2.LINE_AA)
    
    # WiFi
    wifi = _get_wifi_signal()
    cv2.putText(frame, f"WiFi {wifi}", (x + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Server Status
    y += 25
    srv_color = (0, 255, 0) if server_available else (0, 0, 255)
    srv_text = "Connected" if server_available else "Disconnected"
    
    cv2.circle(frame, (x + 10, y - 5), 5, srv_color, -1)
    cv2.putText(frame, f"Server: {srv_text}", (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Mode
    y += 25
    cv2.putText(frame, f"Mode: {mode.upper()}", (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)


def render_face_recognition(frame, faces_info: List[dict]):
    """
    faces_info: list of dicts with:
      - rect: dlib.rectangle
      - name: str
      - seen_count: int
      - last_seen_ago: str
      - summary: str
      - is_recording: bool
    """
    for face in faces_info:
        rect = face['rect']
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Name and Seen Count
        name_text = face['name']
        if face.get('seen_count', 0) > 0:
            name_text += f" (seen {face['seen_count']}x)"
            
        cv2.putText(frame, name_text, (x2 + 10, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Last Seen
        if face.get('last_seen_ago'):
            cv2.putText(frame, f"Last: {face['last_seen_ago']}", (x2 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
        # Summary
        if face.get('summary'):
            sum_text = face['summary']
            if len(sum_text) > 48:
                sum_text = sum_text[:45] + "..."
            cv2.putText(frame, sum_text, (x2 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA) # Yellow
            
        # REC Indicator
        if face.get('is_recording'):
            cv2.circle(frame, (x1 - 15, y1 + 10), 6, (0, 0, 255), -1) # Red dot
            cv2.putText(frame, "REC", (x1 - 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


def render_live_captions(frame, text: str, source_language: str, width: int, height: int):
    if not text:
        return
        
    # Text wrapping logic
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    
    max_width = width - 40 # 20px padding on each side
    words = text.split()
    lines = []
    current_line = ""
    
    # Prepend language indicator if translation is active
    if source_language:
        # Convert code to readable name if possible, else uppercase
        lang_map = {"es": "Spanish", "de": "German", "fr": "French", "it": "Italian"}
        lang_name = lang_map.get(source_language.lower(), source_language.upper())
        current_line = f"({lang_name}) "
    
    for word in words:
        test_line = current_line + word + " "
        (t_w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if t_w > max_width and current_line:
            lines.append(current_line.strip())
            current_line = word + " "
        else:
            current_line = test_line
            
    if current_line:
        lines.append(current_line.strip())
        
    # Show at most the last 2 lines
    lines_to_show = lines[-2:]
    
    # Draw dark background strip
    bg_height = len(lines_to_show) * 35 + 20
    y_start = height - bg_height
    # Create an overlay for alpha blending
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (width, height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    y = y_start + 30
    for line in lines_to_show:
        # If it's a translation, highlight the (Language) part
        if line.startswith("(") and source_language:
            end_idx = line.find(")") + 1
            if end_idx > 0:
                prefix = line[:end_idx]
                rest = line[end_idx:]
                
                # Draw prefix in yellow
                cv2.putText(frame, prefix, (20, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
                
                # Draw rest in white
                (p_w, _), _ = cv2.getTextSize(prefix, font, scale, thickness)
                cv2.putText(frame, rest, (20 + p_w, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            else:
                cv2.putText(frame, line, (20, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, line, (20, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 35


def render_reminders(frame, reminders: List[str], height: int):
    if not reminders:
        return
        
    y = height - 120 # Above captions
    for i, text in enumerate(reminders[:3]): # Max 3
        # Draw a simple bell icon (or just a colored circle as a bullet)
        cv2.circle(frame, (25, y - 5), 4, (0, 200, 255), -1)
        cv2.putText(frame, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y -= 25


def render_toasts(frame, toast_manager: ToastManager, width: int, height: int):
    active_toasts = toast_manager.get_active_toasts()
    if not active_toasts:
        return
        
    y = height // 2
    for toast in active_toasts:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (t_w, t_h), _ = cv2.getTextSize(toast.text, font, scale, thickness)
        
        x = (width - t_w) // 2
        
        # Calculate color based on alpha
        alpha = toast.alpha
        val = int(255 * alpha)
        color = (val, val, val)
        
        if alpha > 0.05:
            cv2.putText(frame, toast.text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            y += 40
