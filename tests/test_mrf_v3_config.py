import pytest

from polyflip.crypto.market_regime import MIN_HISTORY_CANDLES
from polyflip.trading.trading_config import parse_trading_settings


def test_parse_mrf_v3_settings_and_minimum_history():
    cfg = parse_trading_settings(
        {
            "MARKET_REGIME_FILTER_VERSION": "3",
            "MARKET_REGIME_VETO_THRESHOLD": "0.20",
            "MARKET_REGIME_EDGE_OVERRIDE_MARGIN": "0.07",
            "MARKET_REGIME_ASSET_WEIGHT": "0.8",
            "MARKET_REGIME_GLOBAL_WEIGHT": "0.2",
            "MARKET_REGIME_MIN_HISTORY": "20",
        }
    )
    assert cfg.mrf_version == 3
    assert cfg.mrf_veto_threshold == pytest.approx(0.20)
    assert cfg.mrf_edge_override_margin == pytest.approx(0.07)
    assert cfg.mrf_asset_weight == pytest.approx(0.8)
    assert cfg.mrf_global_weight == pytest.approx(0.2)
    assert cfg.mrf_min_history == MIN_HISTORY_CANDLES


def test_parse_mrf_rejects_unknown_version():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_trading_settings({"MARKET_REGIME_FILTER_VERSION": "4"})
