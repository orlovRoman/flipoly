import ast
from polyflip.api.settings import get_all_settings

def test_settings_sync():
    """
    Проверяет, что все ключи, которые движок запрашивает через settings_db.get(),
    присутствуют в словаре, возвращаемом get_all_settings() в API.
    """
    engine_path = "polyflip/trading/engine.py"
    with open(engine_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=engine_path)
        
    engine_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Ищем вызовы settings_db.get("KEY", ...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "settings_db":
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        key = node.args[0].value
                        # Игнорируем динамические ключи, такие как f"TRADING_MODE_{asset}"
                        if not key.startswith("TRADING_MODE_") and \
                           not key.startswith("MIN_EDGE_") and \
                           not key.startswith("TRADE_MAX_PRICE_"):
                            engine_keys.add(key)
                            
    # Получаем ключи из API (нам нужен только словарь)
    # Так как get_all_settings - асинхронная и зависит от БД, 
    # в тесте мы можем просто проверить ключи, которые он формирует в settings_dict.
    # Но так как он читает из БД, мы не можем просто вызвать его без моков.
    # Лучше спарсить api/settings.py
    settings_path = "polyflip/api/settings.py"
    with open(settings_path, "r", encoding="utf-8") as f:
        api_tree = ast.parse(f.read(), filename=settings_path)
        
    api_keys = set()
    for node in ast.walk(api_tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    api_keys.add(k.value)
                    
    # Проверяем, что все ключи из движка есть в API
    missing_in_api = engine_keys - api_keys
    
    # Игнорируем ключи, которые генерируются динамически или не нужны в API
    ignore_list = {"FAVORITE_MIN_EDGE"} 
    missing_in_api = missing_in_api - ignore_list
    
    assert not missing_in_api, f"Ключи, используемые в engine.py, но отсутствующие в get_all_settings (API): {missing_in_api}"
