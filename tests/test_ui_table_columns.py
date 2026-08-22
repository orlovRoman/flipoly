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
        "Статус", "Тип ставки", "Ставка", "Цена", "PnL", "Flip %", "Edge", "MRF",
        "Причина / Ошибка", "Диагн."
    ]
    
    assert headers == expected_order, f"Column order mismatch:\nGot: {headers}\nExpected: {expected_order}"



def test_direction_badge_uses_api_display_contract():
    js_path = os.path.join(
        os.path.dirname(__file__), "..", "polyflip", "static", "js", "trading.js"
    )
    with open(js_path, encoding="utf-8") as file:
        source = file.read()

    assert 'let dirVal = log.direction_display || "NONE";' in source
    assert 'dirVal = log.outcome_bought' not in source


def test_direction_display_distinguishes_none_from_unavailable():
    from polyflip.ui_helpers import direction_display_value

    assert direction_display_value("UP", "MODEL_NOT_LOADED") == "UP"
    assert direction_display_value("DOWN", "SHADOW_NOT_APPLIED") == "DOWN"
    assert direction_display_value(None, "READY") == "NONE"
    assert direction_display_value(None, "OK") == "NONE"
    assert direction_display_value(None, "MODEL_NOT_LOADED") == "UNAVAILABLE"
    assert direction_display_value(None, "PREDICT_ERROR") == "UNAVAILABLE"
    assert direction_display_value(None, "READY", "DIRECTION_UNAVAILABLE") == "UNAVAILABLE"
