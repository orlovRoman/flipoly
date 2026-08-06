import re
import os

path = "C:/Users/orlov/.gemini/antigravity/scratch/flipoly/polyflip/templates/execution.html"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix h3 and button flex layout
content = content.replace('<h3>Торгуемые позиции</h3>', '<h3 style="margin: 0;">Торгуемые позиции</h3>')
content = content.replace('<h3>Завершённые рынки</h3>', '<h3 style="margin: 0;">Завершённые рынки</h3>')
content = content.replace('<h3>Архив</h3>', '<h3 style="margin: 0;">Архив</h3>')
content = content.replace('<h3>LIVE-Заявки (Execution Requests)</h3>', '<h3 style="margin: 0;">LIVE-Заявки (Execution Requests)</h3>')

content = content.replace('<div style="display: flex; justify-content: space-between; align-items: center;">', '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">')

# Fix TableManager pagination styling to add margin-bottom and better colors
css_path = "C:/Users/orlov/.gemini/antigravity/scratch/flipoly/polyflip/static/css/style.css"
with open(css_path, "r", encoding="utf-8") as f:
    css = f.read()

css = css.replace(
'''    margin-top: 10px;
    font-size: 0.85rem;''',
'''    margin-top: 15px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    padding-top: 10px;
    border-top: 1px solid var(--border-color);'''
)

css = css.replace(
'''    background: var(--accent-color, #3182ce);
    color: white;
    border: none;
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;''',
'''    background: #4a5568;
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.2s;'''
)

css = css.replace(
'''.table-pagination button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
}''',
'''.table-pagination button:hover:not(:disabled) {
    background: #2b6cb0;
}
.table-pagination button:disabled {
    opacity: 0.3;
    cursor: not-allowed;
    background: #2d3748;
}'''
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

with open(css_path, "w", encoding="utf-8") as f:
    f.write(css)
