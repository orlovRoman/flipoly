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


def test_parse_mrf_defaults_non_finite_version_without_crashing_scheduler():
    cfg = parse_trading_settings({"MARKET_REGIME_FILTER_VERSION": "inf"})

    assert cfg.mrf_version == 1


def test_parse_mrf_defaults_unknown_version_without_crashing_scheduler():
    cfg = parse_trading_settings({"MARKET_REGIME_FILTER_VERSION": "4"})

    assert cfg.mrf_version == 1


@pytest.mark.parametrize(
    "raw",
    [
        {"MARKET_REGIME_ASSET_WEIGHT": "-1"},
        {"MARKET_REGIME_GLOBAL_WEIGHT": "nan"},
        {
            "MARKET_REGIME_ASSET_WEIGHT": "0",
            "MARKET_REGIME_GLOBAL_WEIGHT": "0",
        },
        {"MARKET_REGIME_VETO_THRESHOLD": "2"},
        {"MARKET_REGIME_EDGE_OVERRIDE_MARGIN": "-0.1"},
    ],
)
def test_parse_mrf_invalid_gate_config_uses_safe_defaults(raw):
    cfg = parse_trading_settings(raw)

    assert cfg.mrf_asset_weight == pytest.approx(0.70)
    assert cfg.mrf_global_weight == pytest.approx(0.30)
    assert cfg.mrf_veto_threshold == pytest.approx(0.15)
    assert cfg.mrf_edge_override_margin == pytest.approx(0.05)
