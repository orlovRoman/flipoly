import sqlite3
try:
    conn = sqlite3.connect('Z:/polymarket-bot/vault/database.sqlite')
    c = conn.cursor()
    c.execute("SELECT val_auc, baseline_auc, created_at FROM model_registry WHERE asset='BTC' ORDER BY created_at DESC LIMIT 5")
    for row in c.fetchall():
        print(row)
except Exception as e:
    print(e)
