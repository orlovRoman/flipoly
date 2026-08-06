import psycopg2
import os
import json

# DB connection using environment or fallback for port forwarding if needed
try:
    conn = psycopg2.connect("postgresql://polyflip:0000@localhost:5432/polyflip")
    cur = conn.cursor()
    cur.execute("""
        SELECT created_at, asset, direction_status, decision_details->>'lgbm_inverted' as inverted, 
               decision_details->>'lgbm_p_up_raw' as p_up_raw, decision_details->>'lgbm_p_down_raw' as p_down_raw, 
               direction_probability as p_up_effective,
               action
        FROM decision_funnel_log 
        ORDER BY created_at DESC 
        LIMIT 10;
    """)
    rows = cur.fetchall()
    for row in rows:
        print(row)
except Exception as e:
    print("Error:", e)
