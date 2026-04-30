"""
TrueVision — Central Configuration

Resolves configuration from CLI arguments, environment variables, and defaults.
"""

import argparse
import os
import platform
from dataclasses import dataclass


@dataclass
class TrueVisionConfig:
    # Server / Connection
    server_url: str
    server_port: int
    controller_port: int
    
    # Models & ML
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    ollama_url: str
    ollama_model: str
    face_detector: str
    
    # Captioning & UI
    caption_interval_sec: float
    caption_window_sec: float
    
    # Translation
    translation_source_languages: list[str]
    translation_target_language: str
    translation_detection_min_probability: float
    
    # Serial / Hardware
    serial_port: str
    baud_rate: int
    
    # Operation
    force_mode: str


def get_default_face_detector() -> str:
    # Default to hog on ARM (Pi), cnn on x86 (desktop/server)
    if platform.machine() in ("armv7l", "aarch64"):
        return "hog"
    return "cnn"

def get_default_whisper_model() -> str:
    if platform.machine() in ("armv7l", "aarch64"):
        return "tiny"
    return "small"

def parse_config() -> TrueVisionConfig:
    parser = argparse.ArgumentParser(description="TrueVision Pi Hub")
    
    parser.add_argument("--server-url", type=str, default=os.environ.get("TRUEVISION_SERVER_URL", ""),
                        help="Server URL for audio offload. If empty, mDNS will be used.")
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("TRUEVISION_SERVER_PORT", "8008")),
                        help="Port for the server to listen on (default: 8008)")
    parser.add_argument("--controller-port", type=int, default=int(os.environ.get("CONTROLLER_PORT", "8080")),
                        help="Port for the phone controller web app (default: 8080)")
    
    parser.add_argument("--whisper-model", type=str, default=os.environ.get("WHISPER_MODEL", get_default_whisper_model()),
                        help="Whisper model size (tiny, small, base)")
    parser.add_argument("--whisper-device", type=str, default=os.environ.get("WHISPER_DEVICE", "cpu"),
                        help="Whisper inference device (cpu, cuda, auto)")
    parser.add_argument("--whisper-compute-type", type=str, default=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
                        help="Whisper compute precision (int8, float16)")
    
    parser.add_argument("--ollama-url", type=str, default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                        help="Ollama API URL")
    parser.add_argument("--ollama-model", type=str, default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
                        help="Ollama model for summarization")
    
    parser.add_argument("--face-detector", type=str, choices=["hog", "cnn", "auto"], 
                        default=os.environ.get("FACE_DETECTOR", "auto"),
                        help="Face detector to use (hog, cnn, auto)")
    
    parser.add_argument("--caption-interval-sec", type=float, default=float(os.environ.get("CAPTION_INTERVAL_SEC", "0.7")),
                        help="Seconds between caption updates")
    parser.add_argument("--caption-window-sec", type=float, default=float(os.environ.get("CAPTION_WINDOW_SEC", "2.0")),
                        help="Rolling audio window size for live captions")
    
    parser.add_argument("--translation-source", type=str, default=os.environ.get("TRANSLATION_SOURCE_LANGUAGES", "es,de"),
                        help="Comma-separated list of languages to auto-translate")
    parser.add_argument("--translation-target", type=str, default=os.environ.get("TRANSLATION_TARGET_LANGUAGE", "en"),
                        help="Translation target language")
    parser.add_argument("--translation-min-prob", type=float, default=float(os.environ.get("TRANSLATION_DETECTION_MIN_PROBABILITY", "0.65")),
                        help="Min confidence for language detection")
    
    parser.add_argument("--serial-port", type=str, default=os.environ.get("SERIAL_PORT", "/dev/ttyAMA0"),
                        help="ESP32 UART port")
    parser.add_argument("--baud-rate", type=int, default=int(os.environ.get("BAUD_RATE", "921600")),
                        help="ESP32 UART baud rate")
    
    parser.add_argument("--force-mode", type=str, choices=["face", "audio", "both", ""], default="",
                        help="Force a specific mode, ignoring ESP32 switch")

    args = parser.parse_args()
    
    face_detector = args.face_detector
    if face_detector == "auto":
        face_detector = get_default_face_detector()

    return TrueVisionConfig(
        server_url=args.server_url,
        server_port=args.server_port,
        controller_port=args.controller_port,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        face_detector=face_detector,
        caption_interval_sec=args.caption_interval_sec,
        caption_window_sec=args.caption_window_sec,
        translation_source_languages=[x.strip() for x in args.translation_source.split(",") if x.strip()],
        translation_target_language=args.translation_target,
        translation_detection_min_probability=args.translation_min_prob,
        serial_port=args.serial_port,
        baud_rate=args.baud_rate,
        force_mode=args.force_mode
    )

config = parse_config()
