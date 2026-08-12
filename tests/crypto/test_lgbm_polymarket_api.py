from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_stored_lgbm_polymarket_backtest_exposes_selected_branch(db_session):
    from polyflip.api.crypto_dashboard import _stored_lgbm_polymarket_backtest
    from polyflip.db.models import ModelRegistry

    variant = {
        "n_markets": 10,
        "n_quotes": 9,
        "n_oof": 10,
        "n_eligible": 3,
        "n_trades": 2,
        "win_rate": 0.5,
        "total_invested": 2.0,
        "net_profit": 0.25,
        "avg_edge": 0.08,
        "avg_net_edge": 0.06,
        "avg_entry_price": 0.30,
        "slices": [{"dimension": "PRICE", "bucket": "0.20-0.35", "trades": 2}],
        "equity_curve": [
            {
                "entry_time": "2026-08-01T00:00:00+00:00",
                "trade_pnl": 0.25,
                "pnl": 0.25,
            }
        ],
    }
    model = ModelRegistry(
        asset="BTCUSDT_low_vol",
        version=4,
        model_type="lgbm",
        model_blob=b"model",
        accuracy=0.62,
        baseline=0.5,
        is_active=False,
        trained_at=datetime.now(timezone.utc),
        training_params={
            "target_source": "POLYMARKET_FINAL_OUTCOME",
            "feature_set": "B",
            "backtest_pnl_mode": "POLYMARKET_OOF",
            "backtest_variants": {"COMBINED": variant},
        },
    )
    db_session.add(model)
    await db_session.commit()

    result = await _stored_lgbm_polymarket_backtest(
        db_session,
        symbol="BTCUSDT",
        feature_set="B",
        strategy_branch="COMBINED",
    )
    assert result["pnl_mode"] == "POLYMARKET_OOF"
    assert result["strategy_branch"] == "COMBINED"
    assert result["n_trades"] == 2
    assert result["net_profit"] == pytest.approx(0.25)
    assert result["regimes"]["BTCUSDT_low_vol"]["version"] == 4