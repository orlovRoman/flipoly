import sqlite3
conn = sqlite3.connect('/home/orlovrp/polymarket-bot/vault/database.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
