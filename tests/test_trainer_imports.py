def test_trainer_module_imports_cleanly():
    """trainer.py должен импортироваться без ошибок и без E402."""
    import importlib
    import polyflip.crypto.trainer as m
    # Проверяем что assert прошёл (иначе модуль не загрузился бы)
    assert hasattr(m, "CRYPTO_FEATURES")
    assert hasattr(m, "CryptoModelTrainer")


def test_no_mid_file_imports():
    """Top-level импорты должны быть в начале файла."""
    import ast, pathlib
    src = pathlib.Path("polyflip/crypto/trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Ищем только импорты на самом верхнем уровне модуля (в tree.body)
    late_imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if node.lineno > 50:
                late_imports.append((node.lineno, ast.dump(node)))
    assert not late_imports, f"Top-level импорты после строки 50: {late_imports}"
