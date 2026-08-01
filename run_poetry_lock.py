import subprocess
import sys
cmd = 'cd /home/orlovrp/flipoly && sudo docker run --rm -v /home/orlovrp/flipoly:/app -w /app python:3.12-slim bash -c "pip install poetry && poetry lock --no-update"'
print("Running...")
res = subprocess.run(['ssh', 'agent-gemini-cli-poly.asia-northeast3-a.gen-lang-client-0035894732', cmd], capture_output=True, text=True, encoding='utf-8')
print("STDOUT:")
print(res.stdout)
print("STDERR:")
print(res.stderr)
if res.returncode != 0:
    sys.exit(res.returncode)
