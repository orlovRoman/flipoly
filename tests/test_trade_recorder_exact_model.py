import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from polyflip.db.models import TradeHistory, LiveMarket
from polyflip.trading.trade_recorder import execute_and_record
from polyflip.trading.decision_logic import TradeDecision
from polyflip.trading.pre_trade_validator import PreTradeValidation


@pytest.mark.asyncio
async def test_trade_recorder_persists_exact_model_fields(db_session):
    market = LiveMarket(
        market_id="test_mkt_1",
        asset="DOGE",
        question="DOGE test?",
        yes_token_id="yes1",
        no_token_id="no1",
        end_time_est=datetime.now(timezone.utc),
        current_yes_price=0.55,
        current_no_price=0.45,
        current_spread=0.01,
        volume_5min=100.0,
        price_velocity=0.01,
        last_updated=datetime.now(timezone.utc),
    )

    decision = TradeDecision(
        action="BUY_YES",
        p_up=0.6,
        strike=0.5,
        reason="Test",
        strategy_type="LEANING",
        buy_price=0.55,
        bet_size_usdc=1.0,
        decision_details={"p_flip_effective": 0.6},
    )

    validation = PreTradeValidation(
        valid=True,
        buy_price=0.55,
        actual_bet_size=1.0,
        edge=0.05,
        skip_reason=None,
        market_role="FAVORITE",
    )

    cfg = MagicMock()
    cfg.stop_loss_enabled = False
    cfg.take_profit_enabled = False
    start_time = datetime.now(timezone.utc)

    await execute_and_record(
        db_session=db_session,
        market=market,
        decision_obj=decision,
        validation=validation,
        asset_mode="COMBINED",
        active_features="f1,f2",
        p_flip=0.6,
        model_ver=8,
        cfg=cfg,
        existing_skipped=None,
        start_time=start_time,
        model_key="DOGE_leaning",
        confirm_model_key="BTCUSDT_low_vol",
        confirm_model_version=12,
    )

    await db_session.flush()

    from sqlalchemy import select

    res = (
        await db_session.execute(
            select(TradeHistory).where(TradeHistory.market_id == "test_mkt_1")
        )
    ).scalar_one()
    assert res.model_key == "DOGE_leaning"
    assert res.confirm_model_key == "BTCUSDT_low_vol"
    assert res.confirm_model_version == 12
    assert res.model_attribution_source == "EXACT"
