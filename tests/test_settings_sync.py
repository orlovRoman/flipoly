import ast
from polyflip.settings_registry import registry_keys

def test_settings_sync():
    """
    Проверяет, что все ключи, которые движок запрашивает через settings_db.get(),
    присутствуют в едином реестре настроек settings_registry.
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
                           not key.startswith("OUTS_MIN_EDGE_") and \
                           not key.startswith("TRADE_MAX_PRICE_"):
                            engine_keys.add(key)
                            
    # Ключи из единого реестра
    registry_all_keys = registry_keys()
                    
    # Проверяем, что все ключи из движка есть в реестре
    missing_in_registry = engine_keys - registry_all_keys
    
    # Игнорируем ключи, которые генерируются динамически или не нужны в API/реестре
    ignore_list = {"FAVORITE_MIN_EDGE"} 
    missing_in_registry = missing_in_registry - ignore_list
    
    assert not missing_in_registry, f"Ключи, используемые в engine.py, но отсутствующие в settings_registry.py: {missing_in_registry}"
