import sqlite3

try:
    conn = sqlite3.connect('file:Z:/polymarket-bot/vault/database.sqlite?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get all tables
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    
    print(f"Found {len(tables)} tables.")
    for table in tables:
        t_name = table['name']
        print(f"\n--- TABLE: {t_name} ---")
        
        # Get one row
        cur.execute(f"SELECT * FROM {t_name} LIMIT 1")
        row = cur.fetchone()
        if row:
            print("\nSample row:")
            for k in row.keys():
                print(f"  {k}: {row[k]}")
        else:
            print("\n(Table is empty)")
            
except Exception as e:
    print(f"Error: {e}")
