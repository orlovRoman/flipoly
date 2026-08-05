import re
import os

def test_trade_logs_table_header_order():
    template_path = os.path.join(os.path.dirname(__file__), "..", "polyflip", "templates", "trading.html")
    with open(template_path, encoding='utf-8') as f:
        html = f.read()
    
    # Extract the thead block specifically for #trade-logs-table
    table_split = html.split('id="trade-logs-table"')
    assert len(table_split) > 1, "Could not find trade-logs-table"
    table_content = table_split[1].split('<tbody>')[0]
    
    thead_match = re.search(r'<thead[^>]*>(.*?)</thead>', table_content, re.DOTALL)
    assert thead_match, "Could not find thead for trade-logs-table"
    
    thead_html = thead_match.group(1)
    headers = re.findall(r'<th[^>]*>(.*?)</th>', thead_html, re.DOTALL)
    headers = [h.strip() for h in headers]
    
    # Порядок должен совпадать с JS-рендером строк
    expected_order = [
        "Мин. отрезка", "Время (UTC)", "Вопрос", "Направление", "Модель",
        "Статус", "Тип ставки", "Ставка", "Цена", "PnL", "Flip %", "Edge",
        "Причина / Ошибка", "Диагн."
    ]
    
    assert headers == expected_order, f"Column order mismatch:\nGot: {headers}\nExpected: {expected_order}"
