import sqlite3
import pickle
import sys

def main():
    try:
        conn = sqlite3.connect("file:///Z:/polymarket-bot/vault/database.sqlite?mode=ro", uri=True)
    except sqlite3.OperationalError as e:
        print(f"Could not connect to database: {e}")
        return

    cursor = conn.cursor()
    cursor.execute("SELECT id, version, asset, is_active, features FROM model_registry ORDER BY version DESC LIMIT 10")
    for row in cursor.fetchall():
        print(row)
    # Commenting out the rest for now

    conn.close()

if __name__ == "__main__":
    main()
