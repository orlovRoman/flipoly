# tests/test_backtest_api_field_names.py
def test_model_registry_has_trained_at():
    """Убедиться что поле называется trained_at, а не created_at."""
    from polyflip.db.models import ModelRegistry
    assert hasattr(ModelRegistry, "trained_at"), \
        "ModelRegistry должен иметь поле trained_at"
    assert not hasattr(ModelRegistry, "created_at"), \
        "ModelRegistry НЕ должен иметь поле created_at"

def test_group_snapshots_accepts_min_snapshots():
    """Убедиться что функция принимает параметр min_snapshots."""
    import inspect
    from polyflip.backtesting.market_replay import group_snapshots_into_replays
    sig = inspect.signature(group_snapshots_into_replays)
    assert "min_snapshots" in sig.parameters
