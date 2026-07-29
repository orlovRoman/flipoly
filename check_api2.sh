#!/bin/bash
# Get the API key from the container environment
API_KEY=$(docker exec polyflip_api env | grep API_KEY | cut -d= -f2 | head -1)
echo "API_KEY found: ${API_KEY:0:10}..."

# Query trade logs with auth
curl -s "http://localhost:8001/api/dashboard/trade_logs?page=1&page_size=5" \
  -H "X-API-Key: ${API_KEY}" | \
  python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for t in data.get('items', []):
        print(f\"id={t['id']} asset={t['asset']} status={t['status']} price={t.get('executed_price')} amount={t.get('amount_usdc')}\")
except Exception as e:
    print(f'Error: {e}')
    sys.stdin.seek(0)
    print(sys.stdin.read()[:500])
"
