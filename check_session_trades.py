import subprocess

sql = """
SELECT id, asset, status, position_status, amount_usdc, entry_filled_shares, remaining_shares, realized_pnl_usdc
FROM trade_history 
WHERE live_session_id = '51f6a53a-6358-4626-9066-986a68885b00'
ORDER BY id DESC;
"""

cmd = [
    "docker", "compose", "exec", "-T", "db", 
    "psql", "-U", "polyflip", "-d", "polyflip", "-c", sql
]

result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/orlovrp/flipoly")
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
