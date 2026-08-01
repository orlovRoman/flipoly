def test_engine_uses_min_candles_constant():
    """engine.py не должен хардкодить limit=100 для get_recent_candles."""
    import ast, pathlib
    src = pathlib.Path("polyflip/trading/engine.py").read_text(encoding="utf-8")
    # Ищем вызов get_recent_candles с числовым литералом limit
    assert "limit=100" not in src, (
        "engine.py хардкодит limit=100 — используй MIN_CANDLES_REQUIRED из predictor.py"
    )
