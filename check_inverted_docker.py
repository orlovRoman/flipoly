import subprocess

sql = """
SELECT id, asset, config_snapshot
FROM trade_history 
ORDER BY id DESC 
LIMIT 1;
"""

cmd = [
    "docker", "compose", "exec", "-T", "db", 
    "psql", "-U", "polyflip", "-d", "polyflip", "-c", sql
]

result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/orlovrp/flipoly")
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
