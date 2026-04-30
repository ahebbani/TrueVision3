"""
TrueVision — Server FastAPI Application

Main server entry point that provides:
  - GET  /health               — Server health check
  - POST /summarize            — LLM summarization
  - WS   /ws/audio             — WebSocket audio endpoint
  - POST /telegram             — Telegram Bot send
  - POST /telegram_llm         — LLM-extracted Telegram Bot send
  - Backfill API               — Offline batch processing
"""

import asyncio
import logging
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from zeroconf import ServiceInfo, Zeroconf

# Import subsystems
from server import audio_ws, backfill
from server.database import ServerDatabase
from server.summarizer import OllamaSummarizer
from server.telegram_handler import TelegramHandler
from server.whisper_service import ServerWhisperService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Globals
db = ServerDatabase()
whisper_svc = ServerWhisperService(
    model_size=os.environ.get("WHISPER_MODEL", "small"),
    device=os.environ.get("WHISPER_DEVICE", "auto"),
    compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
)
ollama = OllamaSummarizer()
telegram = TelegramHandler()

zc: Zeroconf = None


def get_local_ip():
    """Trick to get the local IP address on the LAN."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Server starting up...")
    
    # Init globals in routers
    audio_ws.whisper_service = whisper_svc
    audio_ws.summarizer = ollama
    backfill.db = db
    backfill.whisper_service = whisper_svc
    backfill.summarizer = ollama
    
    # Load Whisper in background
    threading = __import__("threading")
    threading.Thread(target=whisper_svc.load_model, daemon=True).start()
    
    # Setup mDNS Zeroconf advertisement
    global zc
    zc = Zeroconf()
    ip = get_local_ip()
    port = int(os.environ.get("TRUEVISION_SERVER_PORT", "8008"))
    
    desc = {
        "version": "1.0",
        "whisper_model": whisper_svc.model_size,
        "ollama_model": ollama.model
    }
    
    info = ServiceInfo(
        "_truevision._tcp.local.",
        "TrueVision Server._truevision._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties=desc,
        server="truevision.local."
    )
    
    zc.register_service(info)
    logger.info(f"mDNS advertised at {ip}:{port}")
    
    # Start backfill worker
    asyncio.create_task(backfill.trigger_worker_task())
    
    yield
    
    # Shutdown
    logger.info("Server shutting down...")
    if zc:
        zc.unregister_service(info)
        zc.close()


app = FastAPI(lifespan=lifespan)

# Register routers
app.include_router(audio_ws.router)
app.include_router(backfill.router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "1.0",
        "whisper": {
            "model": whisper_svc.model_size,
            "device": whisper_svc.device,
            "loaded": whisper_svc.model is not None
        },
        "ollama": {
            "url": ollama.url,
            "model": ollama.model,
            "available": ollama.is_available()
        },
        "telegram": {
            "configured": telegram.is_configured()
        }
    }


class SummarizeRequest(BaseModel):
    transcript: str
    previous_summary: str = ""
    person_name: str = ""
    max_chars: int = 140

@app.post("/summarize")
def summarize_endpoint(req: SummarizeRequest):
    summary = ollama.summarize(
        transcript=req.transcript,
        previous_summary=req.previous_summary,
        person_name=req.person_name,
        max_chars=req.max_chars
    )
    if not summary:
        raise HTTPException(status_code=500, detail="Summarization failed")
    return {"summary": summary}


class TelegramRequest(BaseModel):
    command: str

@app.post("/telegram")
def telegram_endpoint(req: TelegramRequest):
    # Direct send
    msg_id = telegram.send_message(req.command)
    if not msg_id:
        raise HTTPException(status_code=500, detail="Telegram send failed")
    return {"status": "ok", "message_id": msg_id}


@app.post("/telegram_llm")
def telegram_llm_endpoint(req: TelegramRequest):
    # Extract clean message using LLM, then send
    clean_msg = ollama.extract_telegram_message(req.command)
    if not clean_msg:
        raise HTTPException(status_code=500, detail="LLM extraction failed")
        
    msg_id = telegram.send_message(clean_msg)
    if not msg_id:
        raise HTTPException(status_code=500, detail="Telegram send failed")
        
    return {
        "status": "ok", 
        "extracted_message": clean_msg,
        "message_id": msg_id
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TRUEVISION_SERVER_PORT", "8008"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)
