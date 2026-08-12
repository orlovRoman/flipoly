from datetime import datetime, timezone

import pandas as pd
import pytest

from polyflip.crypto.oof_artifact import OOF_ARTIFACT_SCHEMA_VERSION, serialize_oof_artifact
from polyflip.db.models import ModelRegistry, ModelRegistryOOFArtifact


@pytest.mark.asyncio
async def test_saved_candidate_backtest_uses_persisted_artifact(db_session):
    from polyflip.api.crypto_dashboard import _saved_lgbm_model_polymarket_backtest

    frame = pd.DataFrame({
        "market_id": ["m1", "m2"],
        "asset": ["BTC", "BTC"],
        "market_start": pd.to_datetime(["2026-08-01T00:00:00Z", "2026-08-01T00:15:00Z"], utc=True),
        "recorded_at": pd.to_datetime(["2026-08-01T00:00:00Z", "2026-08-01T00:15:00Z"], utc=True),
        "target": [0, 1],
        "final_outcome": ["NO", "YES"],
    })
    quotes = pd.DataFrame({
        "market_id": ["m1", "m2"],
        "mid_price": [0.8, 0.2],
        "best_bid": [0.79, 0.19],
        "best_ask": [0.81, 0.21],
        "spread": [0.02, 0.02],
    })
    model = ModelRegistry(
        asset="BTCUSDT_low_vol", version=9, model_type="lgbm", model_blob=b"model",
        accuracy=0.61, baseline=0.5, is_active=False,
        trained_at=datetime.now(timezone.utc),
        training_params={"target_source": "POLYMARKET_FINAL_OUTCOME", "feature_set": "B"},
    )
    db_session.add(model)
    await db_session.flush()
    db_session.add(ModelRegistryOOFArtifact(
        model_registry_id=model.id,
        schema_version=OOF_ARTIFACT_SCHEMA_VERSION,
        row_count=2,
        artifact_blob=serialize_oof_artifact(frame, [0.2, 0.8], quotes, feature_set="B"),
        created_at=datetime.now(timezone.utc),
    ))
    await db_session.commit()

    result = await _saved_lgbm_model_polymarket_backtest(
        db_session, model_id=model.id, strategy_branch="OUTSIDER_ONLY"
    )
    assert result["model_id"] == model.id
    assert result["model_version"] == 9
    assert result["n_trades"] == 2
    assert result["net_profit"] > 0


@pytest.mark.asyncio
async def test_saved_candidate_without_artifact_is_explicit_error(db_session):
    from fastapi import HTTPException
    from polyflip.api.crypto_dashboard import _saved_lgbm_model_polymarket_backtest

    model = ModelRegistry(
        asset="ETHUSDT_mid_vol", version=2, model_type="lgbm", model_blob=b"model",
        accuracy=0.55, baseline=0.5, is_active=False,
        trained_at=datetime.now(timezone.utc),
    )
    db_session.add(model)
    await db_session.commit()
    with pytest.raises(HTTPException) as exc:
        await _saved_lgbm_model_polymarket_backtest(
            db_session, model_id=model.id, strategy_branch="COMBINED"
        )
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] == "OOF_ARTIFACT_MISSING"
