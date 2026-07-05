"""Показать байты вокруг проблемной строки."""
with open('tests/test_trade_on_flip.py', 'rb') as f:
    data = f.read().decode('utf-8')

for i, line in enumerate(data.splitlines(keepends=True)):
    if 'err = trades' in line or ('assert' in line and 'dead' in line):
        print(f'Строка {i+1} ({len(line)} байт): {repr(line)}')
