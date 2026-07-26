import os

path = 'polyflip/trading/engine.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_try = False
try_start_idx = -1
for i, line in enumerate(lines):
    if line.strip() == "try:" and "if asset_mode == TRADING_MODE_ML:" in lines[i+1]:
        in_try = True
        try_start_idx = i
        new_lines.append(line)
        continue
    
    if in_try:
        if line.startswith("            except Exception as e:"):
            in_try = False
            new_lines.append(line)
            continue
            
        if line.startswith("    "):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
