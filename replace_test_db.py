import sys

with open('tests/test_settings_api.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('await update_setting(', 'await update_setting(')
# Actually I'll do this better:
import re
code = re.sub(r'await update_setting\("([^"]+)", SettingValue\(value="([^"]+)"\)\)', r'await update_setting("\1", SettingValue(value="\2"), db=db_session)', code)
code = code.replace('update_settings_bulk(payload)', 'update_settings_bulk(payload, db=db_session)')

with open('tests/test_settings_api.py', 'w', encoding='utf-8') as f:
    f.write(code)
