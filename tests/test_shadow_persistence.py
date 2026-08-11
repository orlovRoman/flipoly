from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from polyflip.db.models import TradeHistory
from polyflip.trading.trade_recorder import save_or_update_skipped_trade


def _decision_details() -> dict:
    return {
        "direction_status": "SHADOW_NOT_APPLIED",
        "direction_model_key": "BTCUSDT_low_vol",
        "direction_model_version": 14,
        "direction_value": "UP",
        "direction_probability": 0.84,
        "entry_model_key": "BTC_decided",
        "entry_model_version": 12,
        "entry_model_source": "PHASE",
        "p_candidate_win": 0.78,
        "p_logreg_win": 0.78,
        "gross_edge": 0.08,
        "cost_buffer": 0.02,
        "net_edge": 0.06,
        "decision_run_id": "dec_shadow_1",
    }


@pytest.mark.asyncio
async def test_new_skip_persists_shadow_direction_attribution():
    db = MagicMock()
    market = SimpleNamespace(market_id="m-shadow", asset="BTC")

    await save_or_update_skipped_trade(
        db_session=db,
        market=market,
        reason="insufficient edge",
        p_flip_val=0.22,
        model_version=12,
        start_time=datetime.now(timezone.utc),
        decision_details=_decision_details(),
        direction_value="UP",
        model_key="BTC_decided",
    )

    history = db.add.call_args.args[0]
    assert history.direction_value == "UP"
    assert history.direction_model_key == "BTCUSDT_low_vol"
    assert history.direction_model_version == 14
    assert history.entry_model_key == "BTC_decided"
    assert history.entry_model_version == 12
    assert history.net_edge == 0.06


@pytest.mark.asyncio
async def test_existing_skip_updates_direction_when_previous_value_was_null():
    db = MagicMock()
    market = SimpleNamespace(market_id="m-shadow", asset="BTC")
    existing = TradeHistory(
        market_id="m-shadow",
        asset="BTC",
        outcome_bought="NONE",
        amount_usdc=0.0,
        executed_price=0.0,
        predicted_flip_prob=0.22,
        active_features="COMBINED_ML_LGBM",
        status="SKIPPED",
        error_msg="insufficient edge",
        created_at=datetime.now(timezone.utc),
        direction_value=None,
    )

    await save_or_update_skipped_trade(
        db_session=db,
        market=market,
        reason="insufficient edge",
        p_flip_val=0.22,
        model_version=12,
        start_time=datetime.now(timezone.utc),
        existing_skipped=existing,
        active_features="COMBINED_ML_LGBM",
        decision_details=_decision_details(),
        direction_value="DOWN",
        model_key="BTC_decided",
    )

    assert existing.direction_value == "DOWN"
    assert existing.direction_model_key == "BTCUSDT_low_vol"
    assert existing.direction_model_version == 14
