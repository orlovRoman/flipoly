import os
for path in ['tests/db/test_execution_schema.py', 'tests/trading/test_stoploss_worker.py', 'tests/trading/test_take_profit.py']:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('assert req.requested_mode == "PAPER"', 'assert req.requested_mode == trade.mode')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
