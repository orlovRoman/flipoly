import psycopg2
import os
import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://polyflip:secret@db:5432/polyflip")
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO runtime_settings (key, value, updated_at, updated_by)
    VALUES ('INVERT_LGBM_SIGNAL', 'false', %s, 'system')
    ON CONFLICT(key) DO NOTHING;
""", (datetime.datetime.now(datetime.timezone.utc),))
conn.commit()
conn.close()
print("Migration successful")
