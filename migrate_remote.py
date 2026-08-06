import sqlite3
import datetime

conn = sqlite3.connect('flipoly/vault/database.sqlite')
cursor = conn.cursor()
cursor.execute('''
INSERT INTO runtime_settings (key, value, updated_at, updated_by)
VALUES ('INVERT_LGBM_SIGNAL', 'false', ?, 'system')
ON CONFLICT(key) DO UPDATE SET value='false';
''', (datetime.datetime.now(datetime.timezone.utc).isoformat(),))
conn.commit()
conn.close()
print('Remote Migration successful')
