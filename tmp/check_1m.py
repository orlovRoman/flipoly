import sqlite3
import pandas as pd
import os

try:
    conn = sqlite3.connect("local_db.sqlite")
    df = pd.read_sql("SELECT COUNT(*) FROM crypto_candles WHERE interval='1m'", conn)
    print("local_db 1m candles:", df.iloc[0, 0])
except Exception as e:
    print("local_db error:", e)

try:
    conn2 = sqlite3.connect("polyflip.db")
    df2 = pd.read_sql("SELECT COUNT(*) FROM crypto_candles WHERE interval='1m'", conn2)
    print("polyflip 1m candles:", df2.iloc[0, 0])
except Exception as e:
    print("polyflip error:", e)
