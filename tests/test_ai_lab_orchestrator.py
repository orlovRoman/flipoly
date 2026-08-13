from types import SimpleNamespace


from polyflip.ai_lab.orchestrator import (
    build_experiment_report,
    default_plan_steps,
)


def _result(
    config_id: int,
    *,
    kind: str = "POLYMARKET_OOT",
    pnl: float | None = None,
    trades: int | None = None,
    drawdown: float | None = None,
    artifact_id: int | None = None,
    auc: float | None = None,
):
    return SimpleNamespace(
        config_id=config_id,
        evaluation_kind=kind,
        net_pnl=pnl,
        trade_count=trades,
        max_drawdown=drawdown,
        artifact_id=artifact_id,
        metrics={"auc": auc} if auc is not None else {},
    )


def test_default_plan_steps_use_global_unique_indices():
    steps = default_plan_steps([11, 12])
    assert [step["step_index"] for step in steps] == list(range(6))
    assert [step["config_id"] for step in steps] == [11, 11, 11, 12, 12, 12]
    assert [step["action"] for step in steps[:3]] == [
        "TRAIN_MODEL",
        "RUN_OOT_BACKTEST",
        "RUN_POLYMARKET_OOT",
    ]


def test_report_ranks_by_median_polymarket_pnl_not_auc():
    results = [
        _result(1, pnl=1.0, trades=10, drawdown=-2.0, artifact_id=101, auc=0.60),
        _result(1, pnl=3.0, trades=10, drawdown=-1.0, artifact_id=101, auc=0.61),
        _result(1, pnl=2.0, trades=10, drawdown=-1.5, artifact_id=101, auc=0.59),
        # High AUC alone cannot make this candidate eligible.
        _result(2, kind="OOT", pnl=None, trades=None, auc=0.99),
    ]
    report = build_experiment_report(results, min_trades=3)
    assert report["recommended_config_id"] == 1
    assert report["recommendation_status"] == "READY_FOR_SHADOW"
    winner = next(row for row in report["rows"] if row["config_id"] == 1)
    assert winner["median_oot_pnl"] == 2.0
    assert winner["median_oot_trades"] == 10
    assert winner["eligible_for_shadow"] is True
    other = next(row for row in report["rows"] if row["config_id"] == 2)
    assert other["eligible_for_shadow"] is False


def test_report_does_not_recommend_without_polymarket_pnl_sample():
    report = build_experiment_report(
        [_result(7, kind="OOT", pnl=100.0, trades=100, auc=0.99)],
        min_trades=3,
    )
    assert report["recommendation_status"] == "NO_PNL_SAMPLE"
    assert report["recommended_config_id"] is None
    assert report["rows"][0]["median_oot_pnl"] is None
