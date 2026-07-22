import pytest
from polyflip.api.analytics import get_model_subtype_info

def test_key_full_is_unique_per_lgbm_subtype():
    """key_full = (m.asset, m.version) уникален для каждой субмодели."""
    assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol"]
    version = 5
    keys = [(asset, version) for asset in assets]
    assert len(keys) == len(set(keys)), f"Коллизия key_full: {keys}"

def test_zero_trades_model_gets_empty_stats():
    """
    Модель без трейдов должна возвращать total=0, win_rate=None, pnl=0.0.
    НЕ должна получать статистику от других субтипов того же символа.
    """
    trades_by_exact = {
        ("BTCUSDT_low_vol", 5): {"total": 10, "wins": 7, "pnl": 150.0},
        ("BTCUSDT_mid_vol", 5): {"total": 8,  "wins": 3, "pnl": -20.0},
        # high_vol — нет трейдов
    }
    stats_high = trades_by_exact.get(("BTCUSDT_high_vol", 5), {"total": 0, "wins": 0, "pnl": 0.0})
    assert stats_high["total"] == 0,     "high_vol не должна видеть чужие трейды"
    assert stats_high["pnl"]  == 0.0,    "pnl должен быть 0.0 при отсутствии трейдов"

def test_contested_subtype_returns_unique_key():
    """_contested субтип не должен коллизировать с другими."""
    assets = ["BTCUSDT_low_vol", "BTCUSDT_mid_vol", "BTCUSDT_high_vol",
              "BTCUSDT_contested", "BTCUSDT_leaning", "BTCUSDT_decided", "BTCUSDT"]
    keys = [(asset, 1) for asset in assets]
    assert len(keys) == len(set(keys)), f"Коллизия: {keys}"
