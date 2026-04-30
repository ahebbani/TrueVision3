import sqlite3
from audio.summarizer import summarize_extractive

def main():
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, transcript FROM meetings WHERE transcript IS NOT NULL AND (summary IS NULL OR summary = '')")
    rows = cursor.fetchall()
    
    if not rows:
        print("No summaries to generate.")
        return
        
    print(f"Generating local extractive summaries for {len(rows)} meetings...")
    
    for row in rows:
        m_id = row[0]
        text = row[1]
        sum_text = summarize_extractive(text, max_sentences=1)
        
        cursor.execute("UPDATE meetings SET summary = ? WHERE id = ?", (sum_text, m_id))
        print(f"Updated meeting {m_id}")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
