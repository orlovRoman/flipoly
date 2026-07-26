import os

path = 'polyflip/trading/engine.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # From line 102 to 147 (0-indexed 101 to 146)
    if 101 <= i <= 146:
        # Add 4 spaces
        new_lines.append("    " + line)
    else:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
