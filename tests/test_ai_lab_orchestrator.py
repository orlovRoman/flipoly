from types import SimpleNamespace

from polyflip.ai_lab.orchestrator import (
    MIN_TOTAL_TRADES,
    MIN_WINDOWS,
    RESULT_CLOSING_STATUSES,
    build_experiment_report,
    default_plan_steps,
    evaluate_finalization_gate,
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
    window_start: str | None = None,
    window_end: str | None = None,
    status: str = "SUCCEEDED",
):
    return SimpleNamespace(
        config_id=config_id,
        evaluation_kind=kind,
        status=status,
        net_pnl=pnl,
        trade_count=trades,
        max_drawdown=drawdown,
        artifact_id=artifact_id,
        oot_window_start=window_start,
        oot_window_end=window_end,
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
        _result(
            1,
            pnl=1.0,
            trades=20,
            drawdown=-2.0,
            artifact_id=101,
            auc=0.60,
            window_start="2026-01-01",
            window_end="2026-01-15",
        ),
        _result(
            1,
            pnl=3.0,
            trades=20,
            drawdown=-1.0,
            artifact_id=101,
            auc=0.61,
            window_start="2026-01-16",
            window_end="2026-01-31",
        ),
        _result(
            1,
            pnl=2.0,
            trades=20,
            drawdown=-1.5,
            artifact_id=101,
            auc=0.59,
            window_start="2026-02-01",
            window_end="2026-02-15",
        ),
        # High AUC alone without Polymarket PnL cannot make this candidate eligible.
        _result(2, kind="OOT", pnl=None, trades=None, auc=0.99),
    ]
    report = build_experiment_report(results, min_trades=50, min_windows=3)
    assert report["recommended_config_id"] == 1
    assert report["recommendation_status"] == "READY_FOR_SHADOW"
    assert report["rejection_reasons"] == []
    winner = next(row for row in report["rows"] if row["config_id"] == 1)
    assert winner["median_oot_pnl"] == 2.0
    assert winner["total_trades"] == 60
    assert winner["window_count"] == 3
    assert winner["eligible_for_shadow"] is True
    other = next(row for row in report["rows"] if row["config_id"] == 2)
    assert other["eligible_for_shadow"] is False


def test_report_does_not_recommend_without_polymarket_pnl_sample():
    report = build_experiment_report(
        [_result(7, kind="OOT", pnl=100.0, trades=100, auc=0.99)],
        min_trades=50,
        min_windows=3,
    )
    assert report["recommendation_status"] == "NO_PNL_SAMPLE"
    assert report["recommended_config_id"] is None
    assert report["rows"][0]["median_oot_pnl"] is None


def test_report_ignores_failed_polymarket_evaluations():
    failed = _result(
        9,
        pnl=50.0,
        trades=100,
        auc=0.99,
        window_start="2026-01-01",
        window_end="2026-01-15",
        status="FAILED",
    )
    report = build_experiment_report([failed], min_trades=50, min_windows=3)
    assert report["recommendation_status"] == "NO_PNL_SAMPLE"
    assert report["recommended_config_id"] is None


def test_insufficient_data_closes_queue_step_as_skipped():
    assert "INSUFFICIENT_DATA" in RESULT_CLOSING_STATUSES


def test_finalization_gate_rejects_insufficient_trades():
    # 3 windows, positive PnL, but total trades = 49 < 50
    results = [
        _result(1, pnl=1.0, trades=16, window_start="2026-01-01", window_end="2026-01-15"),
        _result(1, pnl=2.0, trades=16, window_start="2026-01-16", window_end="2026-01-31"),
        _result(1, pnl=3.0, trades=17, window_start="2026-02-01", window_end="2026-02-15"),
    ]
    report = build_experiment_report(results, min_trades=50, min_windows=3)
    assert report["recommendation_status"] == "INSUFFICIENT_TRADES"
    assert report["recommended_config_id"] is None
    assert "INSUFFICIENT_TRADES" in report["rejection_reasons"]
    row = report["rows"][0]
    assert row["eligible_for_shadow"] is False
    assert row["total_trades"] == 49
    assert "INSUFFICIENT_TRADES" in row["rejection_reasons"]


def test_finalization_gate_rejects_insufficient_windows():
    # 2 windows only, total trades = 60 >= 50, positive PnL
    results = [
        _result(1, pnl=1.0, trades=30, window_start="2026-01-01", window_end="2026-01-15"),
        _result(1, pnl=2.0, trades=30, window_start="2026-01-16", window_end="2026-01-31"),
    ]
    report = build_experiment_report(results, min_trades=50, min_windows=3)
    assert report["recommendation_status"] == "INSUFFICIENT_WINDOWS"
    assert report["recommended_config_id"] is None
    assert "INSUFFICIENT_WINDOWS" in report["rejection_reasons"]
    row = report["rows"][0]
    assert row["eligible_for_shadow"] is False
    assert row["window_count"] == 2


def test_finalization_gate_rejects_non_positive_pnl():
    # 3 windows, total trades = 60 >= 50, but median PnL <= 0.0
    results = [
        _result(1, pnl=-1.0, trades=20, window_start="2026-01-01", window_end="2026-01-15"),
        _result(1, pnl=-0.5, trades=20, window_start="2026-01-16", window_end="2026-01-31"),
        _result(1, pnl=0.0, trades=20, window_start="2026-02-01", window_end="2026-02-15"),
    ]
    report = build_experiment_report(results, min_trades=50, min_windows=3)
    assert report["recommendation_status"] == "NON_POSITIVE_PNL"
    assert report["recommended_config_id"] is None
    assert "NON_POSITIVE_PNL" in report["rejection_reasons"]
    row = report["rows"][0]
    assert row["eligible_for_shadow"] is False
    assert row["median_oot_pnl"] == -0.5


def test_finalization_gate_rejects_invalid_nan_pnl():
    gate_res = evaluate_finalization_gate(
        {
            "polymarket_oot_evaluation_count": 3,
            "window_count": 3,
            "total_trades": 60,
            "median_oot_pnl": float("nan"),
            "median_oot_drawdown": -1.0,
        }
    )
    assert gate_res["eligible"] is False
    assert "INVALID_RESULT" in gate_res["rejection_reasons"]


def test_finalization_gate_accepts_valid_candidate_with_contract_fields():
    results = [
        _result(
            1,
            pnl=1.0,
            trades=25,
            drawdown=-0.5,
            artifact_id=101,
            window_start="2026-01-01",
            window_end="2026-01-15",
        ),
        _result(
            1,
            pnl=1.24,
            trades=25,
            drawdown=-0.4,
            artifact_id=101,
            window_start="2026-01-16",
            window_end="2026-01-31",
        ),
        _result(
            1,
            pnl=1.5,
            trades=24,
            drawdown=-0.6,
            artifact_id=101,
            window_start="2026-02-01",
            window_end="2026-02-15",
        ),
    ]
    report = build_experiment_report(results, min_trades=50, min_windows=3)
    assert report["recommendation_status"] == "READY_FOR_SHADOW"
    assert report["rejection_reasons"] == []
    assert report["window_count"] == 3
    assert report["total_trades"] == 74
    assert report["median_pnl"] == 1.24
    assert report["recommended_config_id"] == 1
