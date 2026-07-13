import sqlite3
import json

try:
    conn = sqlite3.connect('Z:/polymarket-bot/vault/database.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT features, model_blob IS NOT NULL FROM model_registry WHERE asset='BTC' ORDER BY created_at DESC LIMIT 1")
    res = cursor.fetchone()
    if res:
        print("Features:", json.loads(res[0]))
        print("Has blob:", bool(res[1]))
    else:
        print("No model found")
except Exception as e:
    print("Error:", e)
