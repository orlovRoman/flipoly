#!/bin/bash
curl -s 'http://localhost:8000/api/dashboard/trade_logs?page=1&page_size=10' | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t in data['items']:
    print(f\"id={t['id']} asset={t['asset']} status={t['status']} executed_price={t['executed_price']} amount_usdc={t['amount_usdc']}\")
"
