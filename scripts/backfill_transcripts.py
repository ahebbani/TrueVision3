import sqlite3
import sys
import os

def main():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, audio_path FROM meetings WHERE transcript IS NULL AND audio_path IS NOT NULL")
    rows = cursor.fetchall()
    
    if not rows:
        print("No transcripts to backfill.")
        return
        
    print(f"Found {len(rows)} meetings missing transcripts. Please use the backfill API on the server to process these.")
    print("Example: POST /api/meetings/<id>/audio with the wav file.")
    
    for row in rows:
        print(f"Meeting {row[0]}: {row[1]}")
        
    conn.close()

if __name__ == "__main__":
    main()
