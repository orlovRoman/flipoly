import sqlite3
import pandas as pd

try:
    print("Reading SQLite database...")
    conn = sqlite3.connect('Z:/polymarket-bot/vault/database.sqlite')
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print("Tables in SQLite:", tables['name'].tolist())
    
    for t in tables['name']:
        count = pd.read_sql(f"SELECT COUNT(*) as c FROM {t}", conn).iloc[0]['c']
        print(f"Table: {t} - {count} rows")
        
        # Print a sample row
        if count > 0:
            sample = pd.read_sql(f"SELECT * FROM {t} LIMIT 1", conn)
            print(f"Sample from {t}:")
            print(sample.to_dict('records')[0])
            print("-" * 40)
            
except Exception as e:
    print("Error:", e)
