import sqlite3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true", help="Generate HTML report instead of console output")
    args = parser.parse_args()
    
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    
    # Simple console output for now, html generation can be expanded
    print("--- FACES ---")
    cursor.execute("SELECT id, name, created_at, last_seen_at, seen_count FROM faces")
    for r in cursor.fetchall():
        print(f"[{r[0]}] {r[1]} | Seen: {r[4]} times | Last: {r[3]}")
        
    print("\n--- MEETINGS ---")
    cursor.execute("SELECT m.id, f.name, m.started_at, m.summary FROM meetings m LEFT JOIN faces f ON m.person_id = f.id ORDER BY m.started_at DESC LIMIT 10")
    for r in cursor.fetchall():
        print(f"[{r[0]}] with {r[1]} at {r[2]}")
        if r[3]:
            print(f"  Summary: {r[3]}")
            
    print("\n--- NOTES ---")
    cursor.execute("SELECT id, content, created_at, is_done FROM notes ORDER BY created_at DESC LIMIT 10")
    for r in cursor.fetchall():
        status = "[DONE]" if r[3] else "[ACTIVE]"
        print(f"{status} [{r[0]}] {r[1]}")
        
    conn.close()

if __name__ == "__main__":
    main()
