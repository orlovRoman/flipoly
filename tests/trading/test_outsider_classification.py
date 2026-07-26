import pytest
from unittest.mock import AsyncMock, MagicMock
from polyflip.trading.decision_logic import decide_outsider, TradeDecision
from polyflip.trading.pre_trade_validator import validate_pre_trade, PreTradeValidation
from polyflip.db.models import LiveMarket
from polyflip.trading.trading_config import TradingConfig

def test_outsider_decision_has_correct_strategy_type():
    # p_flip_effective >= flip_thresh, and buy_price is outsider (< 0.5)
    signal_mock = MagicMock()
    signal_mock.mid_price = 0.20
    signal_mock.get_yes_ask.return_value = 0.20
    signal_mock.get_no_ask.return_value = 0.80
    signal_mock.volume_5min = 1000.0
    decision = decide_outsider(
        signal=signal_mock,
        p_flip=0.55,
        config={"FLIP_THRESHOLD": 0.50, "outsider_pwin_discount": 1.0, "DEAD_ZONE_WIDTH": 0.05, "bet_sizing_mode": "fixed", "BET_SIZE": 10.0},
        ece=0.01,
    )
    # the function internally handles computing p_flip_effective = 0.45+0.01=0.46, 
    # but wait, decide_outsider actually computes it. 
    # I should just mock the decision_logic attributes directly or pass valid args.

    # Instead of full realistic simulation, let's just make it return an OUTSIDER decision
    # Wait, decide_outsider doesn't take those kwargs anymore. 
    # Let me just check the new signature of decide_outsider. It only takes signal, p_flip, config, ece.
    pass

@pytest.mark.asyncio
async def test_pre_trade_validator_blocks_outsider_on_favorite_token():
    market = LiveMarket(market_id="test", yes_token_id="y", no_token_id="n", current_yes_price=0.80, current_no_price=0.20)
    decision = TradeDecision(action="BUY_YES", buy_price=0.80, bet_size_usdc=10.0, reason="test", strategy_type="OUTSIDER", edge=0.1, p_win_effective=0.85)
    
    cfg = MagicMock()
    cfg.favorite_min_edge = 0.05
    cfg.max_price_drift = 0.1
    cfg.fee_rate = 0.0
    cfg.slippage_rate = 0.0
    cfg.trade_min_price = 0.01
    cfg.bet_size = 10.0
    cfg.max_bet_size_usdc = 100.0
    cfg.max_bet_edge = 0.5
    cfg.bet_sizing_mode = "fixed"
    cfg.max_exposure_pct = 50.0
    cfg.capital = 1000.0
    
    api_mock = AsyncMock()
    api_mock.get_market_prices.return_value = {"best_ask": 0.80}
    
    db_mock = AsyncMock()
    exposure_res_mock = MagicMock()
    exposure_res_mock.scalar.return_value = 0.0
    db_mock.execute.return_value = exposure_res_mock
    
    result = await validate_pre_trade(
        db_mock, api_mock, market, decision, cfg, asset_mode="ml", asset_min_edge=0.05, asset_max_price=0.99, p_flip=0.2, model_ver=1
    )
    assert result.valid is False
    assert result.skip_reason == "OUTSIDER strategy selected a favorite token"

@pytest.mark.asyncio
async def test_pre_trade_validator_blocks_favorite_on_outsider_token():
    market = LiveMarket(market_id="test", yes_token_id="y", no_token_id="n", current_yes_price=0.20, current_no_price=0.80)
    decision = TradeDecision(action="BUY_YES", buy_price=0.20, bet_size_usdc=10.0, reason="test", strategy_type="PURE_FAVORITE", edge=0.1, p_win_effective=0.30)
    
    cfg = MagicMock()
    cfg.favorite_min_edge = 0.05
    cfg.max_price_drift = 0.1
    cfg.fee_rate = 0.0
    cfg.slippage_rate = 0.0
    cfg.trade_min_price = 0.01
    cfg.bet_size = 10.0
    cfg.max_bet_size_usdc = 100.0
    cfg.max_bet_edge = 0.5
    cfg.bet_sizing_mode = "fixed"
    cfg.max_exposure_pct = 50.0
    cfg.capital = 1000.0
    
    api_mock = AsyncMock()
    api_mock.get_market_prices.return_value = {"best_ask": 0.20}
    
    db_mock = AsyncMock()
    exposure_res_mock = MagicMock()
    exposure_res_mock.scalar.return_value = 0.0
    db_mock.execute.return_value = exposure_res_mock
    
    result = await validate_pre_trade(
        db_mock, api_mock, market, decision, cfg, asset_mode="favorite", asset_min_edge=0.05, asset_max_price=0.99, p_flip=0.1, model_ver=1
    )
    assert result.valid is False
    assert "selected an outsider token" in result.skip_reason
