"""
TrueVision — Pi-side SQLite Database

Manages the faces.db database with tables:
  - faces          : known people
  - face_embeddings: per-person adaptive templates
  - meetings       : audio sessions / conversations
  - notes          : voice notes and reminders
"""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL,
    last_seen_at TEXT,
    seen_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    quality REAL NOT NULL,
    FOREIGN KEY(face_id) REFERENCES faces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    audio_path TEXT,
    transcript TEXT,
    summary TEXT,
    FOREIGN KEY(person_id) REFERENCES faces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_done INTEGER DEFAULT 0,
    dismissed_at TEXT
);
"""

class Database:
    def __init__(self, db_path: str = "faces.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.executescript(SCHEMA)
            conn.commit()
            conn.close()

    def _get_conn(self):
        # Return a thread-local connection
        return sqlite3.connect(self.db_path)

    # --- Faces ---
    
    def add_face(self, name: str, embedding: np.ndarray) -> int:
        blob = embedding.astype(np.float64).tobytes()
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (name, embedding, created_at, seen_count) VALUES (?, ?, ?, 0)",
                (name, blob, now)
            )
            face_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return face_id

    def get_faces(self) -> List[Tuple[int, str, Optional[np.ndarray], int]]:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, embedding, seen_count FROM faces")
            rows = cursor.fetchall()
            conn.close()
            
            result = []
            for row in rows:
                id_, name, blob, seen_count = row
                emb = np.frombuffer(blob, dtype=np.float64) if blob else None
                result.append((id_, name, emb, seen_count))
            return result
            
    def update_face_seen(self, face_id: int):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE faces SET last_seen_at = ?, seen_count = seen_count + 1 WHERE id = ?",
                (now, face_id)
            )
            conn.commit()
            conn.close()

    # --- Face Embeddings (Templates) ---
    
    def add_embedding(self, face_id: int, embedding: np.ndarray, quality: float) -> int:
        blob = embedding.astype(np.float64).tobytes()
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO face_embeddings (face_id, embedding, created_at, quality) VALUES (?, ?, ?, ?)",
                (face_id, blob, now, quality)
            )
            emb_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return emb_id

    def get_embeddings(self, face_id: int) -> List[Tuple[int, np.ndarray, float]]:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding, quality FROM face_embeddings WHERE face_id = ?", (face_id,))
            rows = cursor.fetchall()
            conn.close()
            
            return [(row[0], np.frombuffer(row[1], dtype=np.float64), row[2]) for row in rows]

    def delete_embeddings(self, embedding_ids: List[int]):
        if not embedding_ids:
            return
        with self.lock:
            conn = self._get_conn()
            placeholders = ",".join("?" * len(embedding_ids))
            conn.execute(f"DELETE FROM face_embeddings WHERE id IN ({placeholders})", embedding_ids)
            conn.commit()
            conn.close()

    # --- Meetings ---
    
    def create_meeting(self, person_id: Optional[int]) -> int:
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO meetings (person_id, started_at) VALUES (?, ?)",
                (person_id, now)
            )
            meeting_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return meeting_id

    def update_meeting(self, meeting_id: int, audio_path: str = None, transcript: str = None, summary: str = None):
        fields = []
        values = []
        if audio_path is not None:
            fields.append("audio_path = ?")
            values.append(audio_path)
        if transcript is not None:
            fields.append("transcript = ?")
            values.append(transcript)
        if summary is not None:
            fields.append("summary = ?")
            values.append(summary)
            
        if not fields:
            return
            
        values.append(meeting_id)
        with self.lock:
            conn = self._get_conn()
            conn.execute(f"UPDATE meetings SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
            conn.close()

    def end_meeting(self, meeting_id: int):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            conn.execute("UPDATE meetings SET ended_at = ? WHERE id = ?", (now, meeting_id))
            conn.commit()
            conn.close()

    def get_latest_summary(self, person_id: int) -> Optional[str]:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary FROM meetings WHERE person_id = ? AND summary IS NOT NULL AND summary != '' ORDER BY started_at DESC LIMIT 1",
                (person_id,)
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None

    # --- Notes ---
    
    def add_note(self, content: str) -> int:
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO notes (content, created_at, is_done) VALUES (?, ?, 0)",
                (content, now)
            )
            note_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return note_id

    def get_active_notes(self) -> List[Tuple[int, str]]:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, content FROM notes WHERE is_done = 0 ORDER BY created_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return rows
            
    def get_all_notes(self) -> List[dict]:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, created_at, is_done, dismissed_at FROM notes ORDER BY created_at DESC")
            rows = cursor.fetchall()
            conn.close()
            return [
                {"id": r[0], "content": r[1], "created_at": r[2], "is_done": bool(r[3]), "dismissed_at": r[4]}
                for r in rows
            ]

    def mark_note_done(self, note_id: int):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE notes SET is_done = 1, dismissed_at = ? WHERE id = ?",
                (now, note_id)
            )
            conn.commit()
            conn.close()

    def delete_note(self, note_id: int):
        with self.lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()
            conn.close()
