"""Исправить отступ в тесте."""
with open('tests/test_trade_on_flip.py', 'rb') as f:
    data = f.read().decode('utf-8')

# Строка с неправильным отступом
BAD  = '                   err = trades[0].error_msg.lower()\r\n'
GOOD = '         err = trades[0].error_msg.lower()\r\n'

if BAD in data:
    data = data.replace(BAD, GOOD)
    print('Отступ исправлен')
else:
    # Попробуем LF
    BAD2  = '                   err = trades[0].error_msg.lower()\n'
    GOOD2 = '         err = trades[0].error_msg.lower()\n'
    if BAD2 in data:
        data = data.replace(BAD2, GOOD2)
        print('Отступ исправлен (LF)')
    else:
        print('Плохая строка не найдена!')
        for i, line in enumerate(data.splitlines()):
            if 'err = trades' in line:
                print(f'Строка {i+1}: {repr(line)}')

with open('tests/test_trade_on_flip.py', 'wb') as f:
    f.write(data.encode('utf-8'))
print('Готово')
