import re

# 1. Fix engine.py
with open(r'C:\Users\orlov\.gemini\antigravity\scratch\flipoly\polyflip\trading\engine.py', 'r', encoding='utf-8') as f:
    engine_content = f.read()

engine_content = engine_content.replace(
    "from polyflip.execution.outbox import EnqueueRejected",
    "from polyflip.trading.trade_recorder import EnqueueRejected"
)
with open(r'C:\Users\orlov\.gemini\antigravity\scratch\flipoly\polyflip\trading\engine.py', 'w', encoding='utf-8') as f:
    f.write(engine_content)

# 2. Fix outbox.py
with open(r'C:\Users\orlov\.gemini\antigravity\scratch\flipoly\polyflip\execution\outbox.py', 'r', encoding='utf-8') as f:
    outbox_content = f.read()

outbox_content = re.sub(
    r'(ExecutionEvent\(\s*level="ERROR" if error else "INFO",\s*event_type=f"REQUEST_{state}",\s*message=error or f"Request transitioned to {state}",\s*source="execution_worker",\s*request_id=req\.id,\s*trade_history_id=req\.trade_history_id,)',
    r'\1\n            created_at=now,',
    outbox_content,
    flags=re.DOTALL
)

with open(r'C:\Users\orlov\.gemini\antigravity\scratch\flipoly\polyflip\execution\outbox.py', 'w', encoding='utf-8') as f:
    f.write(outbox_content)
print("Bugs fixed!")
