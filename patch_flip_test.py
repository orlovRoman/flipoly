"""Патч test_trade_on_flip.py: исправить последний тест под актуальную логику."""
import re

path = "tests/test_trade_on_flip.py"
content = open(path, "rb").read().decode("utf-8")

# 1. Меняем TRADE_MAX_PRICE в последнем тесте (у него есть TRADE_FLIP_THRESHOLD_BTC)
# Ищем именно тот блок где есть TRADE_FLIP_THRESHOLD_BTC
old = 'key="TRADE_MAX_PRICE", value="0.60"'
# Нужно поменять только в последнем тесте — там есть TRADE_FLIP_THRESHOLD_BTC
# Проверяем что OLD встречается ровно один раз в последнем тесте
last_test_start = content.rfind("@pytest.mark.asyncio")
last_test = content[last_test_start:]
count = last_test.count(old)
print(f"Вхождений TRADE_MAX_PRICE=0.60 в последнем тесте: {count}")

if count == 1:
    # Заменяем только в последнем тесте
    before = content[:last_test_start]
    after = last_test.replace(old, 'key="TRADE_MAX_PRICE", value="0.95"', 1)
    content = before + after
    print("TRADE_MAX_PRICE заменён на 0.95")
else:
    print("ОШИБКА: неожиданное количество вхождений")

# 2. Меняем модель и комментарии в последнем тесте
last_test_start = content.rfind("@pytest.mark.asyncio")
before = content[:last_test_start]
last_test = content[last_test_start:]

# Комментарий + модель
old_model_block = (
    '    # Model predicts flip prob = 0.25 (NO wins)\n'
    '    # YES is favorite (0.70), NO is outsider (0.30)\n'
    '    # 0.25 < FLIP_THRESHOLD=0.70, so trade should be skipped\n'
    '    model = MockModel([0.75, 0.25])'
)
new_model_block = (
    '    # DEAD_ZONE_WIDTH=0.05, FLIP_THRESHOLD=0.70, AUTO_DEAD_ZONE=false\n'
    '    # => compute_dead_zone(0.70, 0.05, auto_mode=False) => lower=0.65, upper=0.70\n'
    '    # p_flip=0.67 попадает в мертвую зону [0.65, 0.70] — сделка должна быть пропущена\n'
    '    model = MockModel([0.33, 0.67])'
)

# Учтем возможный \r\n
old_model_block_crlf = old_model_block.replace("\n", "\r\n")
new_model_block_crlf = new_model_block.replace("\n", "\r\n")

if old_model_block_crlf in last_test:
    last_test = last_test.replace(old_model_block_crlf, new_model_block_crlf, 1)
    print("Комментарий+модель заменены (CRLF)")
elif old_model_block in last_test:
    last_test = last_test.replace(old_model_block, new_model_block, 1)
    print("Комментарий+модель заменены (LF)")
else:
    print("ПРЕДУПРЕЖДЕНИЕ: блок комментария+модели не найден")
    # Попробуем найти просто строку модели
    if "MockModel([0.75, 0.25])" in last_test:
        last_test = last_test.replace("MockModel([0.75, 0.25])", "MockModel([0.33, 0.67])", 1)
        print("Заменена только строка модели")

# 3. Меняем assert на более точный
old_assert_crlf = (
    "         assert \"\u043c\u0451\u0440\u0442\u0432\u0430\u044f \u0437\u043e\u043d\u0430\" in trades[0].error_msg.lower()"
    " or \"< threshold\" in trades[0].error_msg.lower()"
    " or \"< threshold\" in trades[0].error_msg.lower()"
)
# Ищем через регекс (символы кирилицы могут не совпасть)
pattern = r'assert .+мёртвая зона.+ in trades\[0\]\.error_msg\.lower\(\).+'
match = re.search(pattern, last_test)
if match:
    old_line = match.group(0)
    new_lines = (
        "         err = trades[0].error_msg.lower()\n"
        "         assert \"dead\" in err or \"zone\" in err or \"\u043c\u0451\u0440\u0442\u0432\" in err or \"\u0437\u043e\u043d\u0430\" in err, (\n"
        "             f\"\u041e\u0436\u0438\u0434\u0430\u043b\u043e\u0441\u044c \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435 \u043e \u043c\u0451\u0440\u0442\u0432\u043e\u0439 \u0437\u043e\u043d\u0435, \u043f\u043e\u043b\u0443\u0447\u0435\u043d\u043e: {trades[0].error_msg!r}\"\n"
        "         )"
    )
    last_test = last_test.replace(old_line, new_lines.replace("\n", "\r\n"))
    print("Assert заменён")
else:
    print("ПРЕДУПРЕЖДЕНИЕ: assert не найден через regex, ищем другие варианты")
    # Поробуем просто найти substring
    if "мёртвая зона" in last_test:
        idx = last_test.rfind("assert")
        print(f"Последний assert в тесте: {repr(last_test[idx:idx+200])}")

content = before + last_test

open(path, "wb").write(content.encode("utf-8"))
print("Файл сохранён")
