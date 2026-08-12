import pytest

from polyflip.api.crypto_dashboard import (
    ExperimentConfigRequest,
    copy_experiment_config,
    CopyExperimentConfigRequest,
    create_experiment_config,
    _list_experiment_configs,
)


@pytest.mark.asyncio
async def test_create_list_and_copy_experiment_config(db_session):
    created = await create_experiment_config(
        ExperimentConfigRequest(
            name="BTC B exploratory",
            asset="btcusdt",
            feature_set="B",
            model={"num_leaves": 19},
            backtest={"min_edge": 0.03},
        ),
        db=db_session,
    )
    row = created["config"]
    assert row["feature_set"] == "B"
    assert row["model"]["num_leaves"] == 19
    assert row["config_hash"]

    listed = await _list_experiment_configs(db=db_session, limit=1)
    assert [item["id"] for item in listed["configs"]] == [row["id"]]

    copied = await copy_experiment_config(row["id"], payload=CopyExperimentConfigRequest(name="BTC B copy", created_by="test"), db=db_session)
    assert copied["config"]["parent_id"] == row["id"]
    assert copied["config"]["config_hash"] == row["config_hash"]
    assert copied["config"]["created_by"] == "test"


@pytest.mark.asyncio
async def test_create_experiment_config_rejects_invalid_asset(db_session):
    with pytest.raises(Exception, match="supported crypto symbol"):
        await create_experiment_config(
            ExperimentConfigRequest(name="bad", asset="NOT_SUPPORTED"),
            db=db_session,
        )
