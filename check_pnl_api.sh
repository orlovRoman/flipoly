#!/bin/bash
API_KEY=$(docker exec polyflip_api env | grep API_KEY | cut -d= -f2 | head -1)
curl -s "http://localhost:8001/api/dashboard/model_pnl?requested_mode=PAPER" \
  -H "X-API-Key: ${API_KEY}" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
for key, val in data.get('data', {}).items():
    if val.get('total_trades', 0) > 0:
        print(f\"{key}: trades={val['total_trades']}, pnl={val['pnl']}, wr={val['win_rate']}%\")
"
