import sqlite3

db_path = "Z:/polymarket-bot/vault/database.sqlite"
conn = sqlite3.connect(f"file:///{db_path}?mode=ro", uri=True)
cursor = conn.cursor()
cursor.execute("SELECT id, asset, outcome_bought, amount_usdc, executed_price, predicted_flip_prob FROM trade_history ORDER BY id DESC LIMIT 5;")
for row in cursor.fetchall():
    print(row)
conn.close()
