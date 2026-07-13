with open("out.txt", "w", encoding="utf-8") as out:
    for i, line in enumerate(open('polyflip/templates/index.html', encoding='utf-8')):
        if 'time_left' in line:
            out.write(f"{i+1}: {line.strip()}\n")
