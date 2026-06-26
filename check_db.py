import sqlite3
import pandas as pd
conn = sqlite3.connect('Z:/polymarket-bot/vault/database.sqlite')
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
for t in tables['name']:
    count = pd.read_sql(f"SELECT COUNT(*) as c FROM {t}", conn).iloc[0]['c']
    print(f"Table: {t} - {count} rows")
