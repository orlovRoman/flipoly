"""
tests/trading/test_market_guards_regime.py

Tests for Item 30:
- Pre-trade regime eligibility guard before side selection
- Banning non-reversion regimes across all execution branches
- Ensuring repeated decision processing does not create duplicate orders
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from polyflip.trading.market_guards import check_market_guards
from polyflip.trading.trading_config import TradingConfig


@pytest.mark.asyncio
async def test_regime_guard_blocks_trend_when_enabled():
    mock_market = MagicMock()
    mock_market.market_id = "test_m1"
    mock_market.asset = "BTC"
    mock_market.yes_token_id = "tok_yes"
    mock_market.no_token_id = "tok_no"

    cfg = MagicMock()
    cfg.favor_min_time_left = 60
    cfg.favor_max_time_left = 900
    cfg.outs_min_time_left = 60
    cfg.outs_max_time_left = 900
    cfg.trade_assets = ["BTC"]
    cfg.require_reversion_regime = True

    mock_session = AsyncMock()
    res_mock = MagicMock()
    res_mock.scalar_one_or_none.return_value = None
    res_mock.scalars.return_value.first.return_value = None
    res_mock.scalars.return_value.all.return_value = [100.0, 102.0, 105.0, 108.0, 112.0]
    mock_session.execute.return_value = res_mock

    with patch("polyflip.research.regime_features.classify_local_regime") as mock_classify:
        mock_classify.return_value = {"state": "TREND"}

        now = datetime.now(timezone.utc)
        res = await check_market_guards(mock_session, mock_market, cfg, "active", 300.0, now)

        assert res.passed is False
        assert "Non-reversion regime" in res.skip_reason


@pytest.mark.asyncio
async def test_regime_guard_allows_reversion():
    mock_market = MagicMock()
    mock_market.market_id = "test_m2"
    mock_market.asset = "BTC"
    mock_market.yes_token_id = "tok_yes"
    mock_market.no_token_id = "tok_no"

    cfg = MagicMock()
    cfg.favor_min_time_left = 60
    cfg.favor_max_time_left = 900
    cfg.outs_min_time_left = 60
    cfg.outs_max_time_left = 900
    cfg.trade_assets = ["BTC"]
    cfg.require_reversion_regime = True

    mock_session = AsyncMock()
    res_mock = MagicMock()
    res_mock.scalar_one_or_none.return_value = None
    res_mock.scalars.return_value.first.return_value = None
    res_mock.scalars.return_value.all.return_value = [100.0, 105.0, 100.0, 105.0, 100.0]
    mock_session.execute.return_value = res_mock

    with patch("polyflip.research.regime_features.classify_local_regime") as mock_classify:
        mock_classify.return_value = {"state": "REVERSION"}

        now = datetime.now(timezone.utc)
        res = await check_market_guards(mock_session, mock_market, cfg, "active", 300.0, now)

        assert res.passed is True
        assert res.skip_reason is None


@pytest.mark.asyncio
async def test_regime_guard_duplicate_order_prevented_before_regime():
    mock_market = MagicMock()
    mock_market.market_id = "test_m3"
    mock_market.asset = "BTC"
    mock_market.yes_token_id = "tok_yes"
    mock_market.no_token_id = "tok_no"

    cfg = MagicMock()
    cfg.favor_min_time_left = 60
    cfg.favor_max_time_left = 900
    cfg.outs_min_time_left = 60
    cfg.outs_max_time_left = 900
    cfg.trade_assets = ["BTC"]
    cfg.require_reversion_regime = True

    mock_session = AsyncMock()
    res_mock = MagicMock()
    res_mock.scalar_one_or_none.return_value = None
    res_mock.scalars.return_value.first.return_value = MagicMock()  # Already traded!
    mock_session.execute.return_value = res_mock

    now = datetime.now(timezone.utc)
    res = await check_market_guards(mock_session, mock_market, cfg, "active", 300.0, now)

    assert res.passed is False
    assert res.skip_reason == "guard: Trade already exists"
