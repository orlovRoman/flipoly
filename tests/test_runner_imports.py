# tests/test_runner_imports.py
def test_runner_imports_no_duplicate():
    """Проверяет что FEATURE_COLUMNS доступен на top-level без внутреннего импорта."""
    import inspect
    from polyflip.backtesting import runner
    source = inspect.getsource(runner.BacktestRunner._predict_flip)
    assert "from polyflip" not in source, "Внутренний импорт в _predict_flip не должен существовать"

def test_runner_empty_model_blob():
    from polyflip.backtesting.runner import BacktestRunner
    r = BacktestRunner(config={}, model_blob=b"", features="")
    assert r.model is None

def test_runner_none_model_blob():
    from polyflip.backtesting.runner import BacktestRunner
    r = BacktestRunner(config={}, model_blob=None, features="")
    assert r.model is None
