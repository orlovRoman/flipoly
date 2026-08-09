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
        "До конца", "Время (UTC)", "Вопрос", "Направление (LGBM)", "LogReg",
        "Статус", "Тип ставки", "Ставка", "Цена", "PnL", "Flip %", "Edge",
        "Причина / Ошибка", "Диагн."
    ]
    
    assert headers == expected_order, f"Column order mismatch:\nGot: {headers}\nExpected: {expected_order}"



def test_direction_badge_does_not_infer_signal_from_bought_outcome():
    """The direction column must display the frozen signal, never the executed side."""
    js_path = os.path.join(
        os.path.dirname(__file__), "..", "polyflip", "static", "js", "trading.js"
    )
    with open(js_path, encoding="utf-8") as file:
        source = file.read()

    badge_block = source.split('let directionBadge = "-";', 1)[1].split(
        "const betText", 1
    )[0]
    assert 'log.direction_value || "NONE"' in badge_block
    assert "outcome_bought" not in badge_block
