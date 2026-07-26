import re

with open('polyflip/trading/engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_try = False
for i, line in enumerate(lines):
    if "try:" in line and i + 1 < len(lines) and "existing_skipped = guard_res.existing_skipped" in lines[i+1]:
        new_lines.append(line)
        in_try = True
        continue
    
    if in_try:
        if line.strip() == "finally:":
            in_try = False
            new_lines.append(line)
            continue
        
        if line.strip():
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('polyflip/trading/engine.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
