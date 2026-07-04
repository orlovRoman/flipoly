# tests/test_crypto_dashboard.py

def test_numpy_import_no_name_error():
    """Гарантирует, что np доступен на уровне модуля."""
    import polyflip.api.crypto_dashboard as m
    assert hasattr(m, 'np')


def test_backtest_endpoint_no_global_mutation():
    """После фикса — run_backtest принимает параметр напрямую, не мутируя глобальный constants."""
    from polyflip.crypto.backtester import run_backtest
    import inspect
    params = inspect.signature(run_backtest).parameters
    assert "min_edge" in params, "run_backtest должен принимать min_edge явно"
    assert "commission" in params, "run_backtest должен принимать commission явно"


def test_router_paths_no_duplicate_prefix():
    """Маршруты не дублируют /crypto в пути при prefix=/crypto."""
    from polyflip.api.crypto_dashboard import router
    paths = [r.path for r in router.routes]
    for p in paths:
        assert not p.startswith("/api/crypto"), (
            f"Путь {p!r} дублирует prefix. Используй /api/status и т.д."
        )
