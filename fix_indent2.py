"""Исправить точный отступ."""
with open('tests/test_trade_on_flip.py', 'rb') as f:
    data = f.read().decode('utf-8')

# Строка 203 имеет 18 пробелов (не 9 пробелов как должна)
BAD  = '                  err = trades[0].error_msg.lower()\r\n'
GOOD = '         err = trades[0].error_msg.lower()\r\n'

if BAD in data:
    data = data.replace(BAD, GOOD)
    print('OK - исправлено')
else:
    print('Не найдено')

with open('tests/test_trade_on_flip.py', 'wb') as f:
    f.write(data.encode('utf-8'))
