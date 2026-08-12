from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_lgbm_experiment_groups_include_a_b_candidates(db_session):
    from polyflip.api.crypto_dashboard import lgbm_experiment_groups
    from polyflip.db.models import ModelRegistry

    common = {
        "target_source": "POLYMARKET_FINAL_OUTCOME",
        "comparison_key": "BTCUSDT_low_vol|same-window",
        "validation_scheme": "TIME_SERIES_SPLIT",
        "feature_set_version": "A-control-v1",
        "backtest_variants": {"COMBINED": {"net_profit": 1.0, "n_trades": 3, "roi_pct": 10.0}},
    }
    for version, feature_set, feature_version in (
        (1, "A", "A-control-v1"),
        (2, "B", "B-direction-sequence-v1"),
    ):
        params = {**common, "feature_set": feature_set, "feature_set_version": feature_version}
        db_session.add(ModelRegistry(
            asset="BTCUSDT_low_vol", version=version, model_type="lgbm", model_blob=b"model",
            accuracy=0.55 + version / 100, baseline=0.5, ece=0.02,
            trained_at=datetime.now(timezone.utc), training_params=params,
        ))
    await db_session.commit()

    response = await lgbm_experiment_groups(db_session)
    assert response["count"] == 1
    group = response["groups"][0]
    assert group["comparable"] is True
    assert {candidate["feature_set"] for candidate in group["variants"]} == {"A", "B"}

    from polyflip.api.crypto_dashboard import lgbm_experiment_report
    report = await lgbm_experiment_report(
        comparison_key="BTCUSDT_low_vol|same-window",
        strategy_branch="COMBINED",
        db=db_session,
    )
    assert report["recommended_variant"] == "B"
    assert report["recommendation_status"] == "PROVISIONAL_LOW_SAMPLE"
    assert report["activation_policy"] == "MANUAL_SHADOW_REQUIRED"
