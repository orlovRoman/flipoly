"""Per-asset PAPER routing tests."""

from polyflip.trading.engine import _settings_for_asset
from polyflip.trading.trading_config import parse_trading_settings


def test_btc_can_use_ct_while_global_mode_is_combined():
    raw = {
        "TRADING_MODE": "combined",
        "TRADING_MODE_BTC": "ct_outsider",
        "TRADE_ASSETS": "BTC,ETH,SOL,XRP,DOGE",
    }

    btc = parse_trading_settings(_settings_for_asset(raw, "BTC"))
    eth = parse_trading_settings(_settings_for_asset(raw, "ETH"))

    assert btc.trading_mode == "ct_outsider"
    assert eth.trading_mode == "combined"
    assert btc.trade_assets == eth.trade_assets == ["BTC", "ETH", "SOL", "XRP", "DOGE"]


def test_blank_override_keeps_global_mode_and_does_not_mutate_input():
    raw = {"TRADING_MODE": "combined", "TRADING_MODE_ETH": ""}
    resolved = _settings_for_asset(raw, "ETH")

    assert resolved["TRADING_MODE"] == "combined"
    assert raw["TRADING_MODE"] == "combined"
    assert resolved is not raw
