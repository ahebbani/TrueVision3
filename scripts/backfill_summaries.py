import sqlite3
import requests
import os

def main():
    url = os.environ.get("TRUEVISION_SERVER_URL", "http://localhost:8008")
    
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT m.id, m.transcript, f.name FROM meetings m LEFT JOIN faces f ON m.person_id = f.id WHERE m.transcript IS NOT NULL AND (m.summary IS NULL OR m.summary = '')")
    rows = cursor.fetchall()
    
    if not rows:
        print("No summaries to backfill.")
        return
        
    print(f"Requesting remote LLM summaries for {len(rows)} meetings from {url}...")
    
    for row in rows:
        m_id, text, name = row
        try:
            resp = requests.post(f"{url}/summarize", json={"transcript": text, "person_name": name or ""}, timeout=30.0)
            if resp.status_code == 200:
                summary = resp.json().get("summary")
                cursor.execute("UPDATE meetings SET summary = ? WHERE id = ?", (summary, m_id))
                conn.commit()
                print(f"Success meeting {m_id}: {summary}")
            else:
                print(f"Failed meeting {m_id}: {resp.status_code}")
        except Exception as e:
            print(f"Error on meeting {m_id}: {e}")
            
    conn.close()

if __name__ == "__main__":
    main()
