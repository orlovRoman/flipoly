import sqlite3

DB = r"Z:\polymarket-bot\vault\database.sqlite"
conn = sqlite3.connect(f"file:///{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Список таблиц
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

conn.close()
