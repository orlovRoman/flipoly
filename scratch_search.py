import os, re
with open("out.txt", "w", encoding="utf-8") as f:
    for root, _, files in os.walk('polyflip'):
        for file in files:
            if file.endswith('.py'):
                try:
                    for line_idx, line in enumerate(open(os.path.join(root, file), encoding='utf-8')):
                        if 'epsilon' in line.lower() or 'quantile' in line.lower():
                            f.write(f"{root}/{file}:{line_idx+1}: {line.strip()}\n")
                except Exception as e:
                    pass
