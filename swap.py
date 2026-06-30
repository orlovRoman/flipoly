import io

path = r'C:\Users\orlov\.gemini\antigravity\scratch\polyflip\polyflip\templates\trading.html'
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

settings_start = -1
settings_end = -1
logs_start = -1
logs_end = -1

for i, line in enumerate(lines):
    if '<!-- Trading Settings -->' in line:
        settings_start = i
    elif '<!-- LOGS SECTION -->' in line:
        logs_start = i

if settings_start != -1 and logs_start != -1:
    # We know LOGS SECTION is right after Trading Settings.
    # We need to find the ends.
    # Trading settings ends right before logs_start.
    settings_end = logs_start - 1
    # Logs section ends before </section>
    for i in range(logs_start, len(lines)):
        if '</section>' in lines[i]:
            logs_end = i - 1
            break
            
print(f"Settings: {settings_start}-{settings_end}")
print(f"Logs: {logs_start}-{logs_end}")

# Now extract
settings_block = lines[settings_start:settings_end+1]
logs_block = lines[logs_start:logs_end+1]

# Apply styling to settings block
settings_block[1] = settings_block[1].replace('margin-top: 1.5rem;', 'margin-top: 1.5rem; max-width: 800px; margin-left: auto; margin-right: auto;')
style_block = """                    <style>
                        #trading-settings-form .form-group label {
                            font-size: 0.85rem !important;
                            color: var(--text-muted) !important;
                            font-weight: 500 !important;
                            margin-bottom: 0.4rem !important;
                        }
                        #trading-settings-form .form-group input {
                            font-size: 0.9rem !important;
                            padding: 0.6rem 0.85rem !important;
                        }
                        #trading-settings-form .form-group small {
                            font-size: 0.8rem !important;
                        }
                    </style>
"""
settings_block.insert(2, style_block)

# Put them together: logs first, then settings
new_lines = lines[:settings_start] + logs_block + ["\n"] + settings_block + lines[logs_end+1:]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('SUCCESS')
