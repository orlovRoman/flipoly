import subprocess

cmd = [
    'ssh', 
    'agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732', 
    "docker exec polyflip_db psql -U polyflip -d polyflip -c \"COPY (SELECT asset, status, pnl, active_features as strategy_type, amount_usdc, executed_price, outcome_bought FROM trade_history WHERE status IN ('SUCCESS', 'FAILED') AND pnl IS NOT NULL) TO STDOUT WITH CSV HEADER;\""
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    with open('data.csv', 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    print("CSV written successfully!")
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
