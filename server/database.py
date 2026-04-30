"""
TrueVision — Server Database

SQLite database for the server-side job queue (backfill processing).
"""

import sqlite3
import threading
from datetime import datetime
from typing import List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    audio_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    transcript TEXT,
    summary TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

class ServerDatabase:
    def __init__(self, db_path: str = "truevision_server.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def queue_job(self, meeting_id: int, audio_path: str) -> int:
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (meeting_id, audio_path, status, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?)",
                (meeting_id, audio_path, now, now)
            )
            job_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return job_id

    def get_job(self, job_id: int) -> Optional[dict]:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, meeting_id, audio_path, status, transcript, summary, error FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "meeting_id": row[1],
                    "audio_path": row[2],
                    "status": row[3],
                    "transcript": row[4],
                    "summary": row[5],
                    "error": row[6]
                }
            return None

    def get_next_queued_job(self) -> Optional[dict]:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, meeting_id, audio_path FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1")
            row = cursor.fetchone()
            
            if row:
                # Mark as processing
                now = datetime.now().isoformat()
                cursor.execute("UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?", (now, row[0]))
                conn.commit()
                conn.close()
                return {"id": row[0], "meeting_id": row[1], "audio_path": row[2]}
                
            conn.close()
            return None

    def update_job_success(self, job_id: int, transcript: str, summary: str):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE jobs SET status = 'done', transcript = ?, summary = ?, updated_at = ? WHERE id = ?",
                (transcript, summary, now, job_id)
            )
            conn.commit()
            conn.close()

    def update_job_error(self, job_id: int, error: str):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE jobs SET status = 'error', error = ?, updated_at = ? WHERE id = ?",
                (error, now, job_id)
            )
            conn.commit()
            conn.close()
