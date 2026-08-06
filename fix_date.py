import os

path = "C:/Users/orlov/.gemini/antigravity/scratch/flipoly/polyflip/templates/execution.html"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace substring(11, 19) to show date as well (MM-DD HH:MM:SS)
content = content.replace(".substring(11, 19)", ".substring(5, 19).replace('T', ' ')")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
