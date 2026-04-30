"""
TrueVision — Backfill API

Endpoints for uploading audio files for offline transcription
and polling for results. Includes a background worker.
"""

import asyncio
import logging
import os
import shutil
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from server.database import ServerDatabase
from server.summarizer import OllamaSummarizer
from server.whisper_service import ServerWhisperService

logger = logging.getLogger(__name__)

router = APIRouter()

# Globals
db: Optional[ServerDatabase] = None
whisper_service: Optional[ServerWhisperService] = None
summarizer: Optional[OllamaSummarizer] = None

UPLOAD_DIR = "uploads"


class BackfillStatusResponse(BaseModel):
    id: int
    meeting_id: int
    status: str
    transcript: Optional[str]
    summary: Optional[str]
    error: Optional[str]


@router.post("/api/meetings/{meeting_id}/audio")
async def upload_audio(meeting_id: int, file: UploadFile = File(...)):
    """Uploads a WAV file for offline transcription."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
        
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_ext = os.path.splitext(file.filename)[1]
    safe_filename = f"meeting_{meeting_id}_{os.urandom(4).hex()}{file_ext}"
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        job_id = db.queue_job(meeting_id, filepath)
        
        # Try to wake up worker
        asyncio.create_task(trigger_worker_task())
        
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.get("/api/meetings/{meeting_id}/status", response_model=BackfillStatusResponse)
async def get_status(meeting_id: int):
    """Polls job status by meeting ID. (Normally we'd use job ID, but Pi uses meeting_id)."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
        
    # We just need a way to look it up. The DB gets it by job_id, but we can query by meeting_id
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, meeting_id, status, transcript, summary, error FROM jobs WHERE meeting_id = ? ORDER BY id DESC LIMIT 1", (meeting_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return BackfillStatusResponse(
        id=row[0],
        meeting_id=row[1],
        status=row[2],
        transcript=row[3],
        summary=row[4],
        error=row[5]
    )


@router.post("/api/backfill/trigger")
async def trigger_backfill():
    """Manual trigger for the background worker."""
    asyncio.create_task(trigger_worker_task())
    return {"status": "triggered"}


# Background Worker

worker_lock = asyncio.Lock()

async def trigger_worker_task():
    """Processes jobs one by one."""
    if worker_lock.locked():
        return
        
    async with worker_lock:
        if not db or not whisper_service:
            return
            
        loop = asyncio.get_event_loop()
        
        while True:
            job = db.get_next_queued_job()
            if not job:
                break
                
            logger.info(f"Processing backfill job {job['id']} for meeting {job['meeting_id']}")
            
            try:
                audio_path = job['audio_path']
                if not os.path.exists(audio_path):
                    db.update_job_error(job['id'], "Audio file not found")
                    continue
                    
                # 1. Transcribe
                transcript = await loop.run_in_executor(None, whisper_service.transcribe, audio_path)
                
                if not transcript:
                    db.update_job_error(job['id'], "Transcription produced no text")
                    continue
                    
                # 2. Summarize
                summary = ""
                if summarizer:
                    summary = await loop.run_in_executor(
                        None,
                        summarizer.summarize,
                        transcript,
                        "", "", 140
                    )
                    
                # 3. Save
                db.update_job_success(job['id'], transcript, summary)
                logger.info(f"Backfill job {job['id']} completed successfully.")
                
            except Exception as e:
                logger.error(f"Backfill job {job['id']} failed: {e}")
                db.update_job_error(job['id'], str(e))
