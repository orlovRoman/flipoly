import inspect
from types import SimpleNamespace
import unittest

from polyflip.ai_lab.orchestrator import (
    MIN_TOTAL_TRADES,
    MIN_WINDOWS,
    RESULT_CLOSING_STATUSES,
    build_experiment_report,
    default_plan_steps,
    evaluate_finalization_gate,
    plan_run,
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


class TestAILabOrchestrator(unittest.TestCase):

    def test_default_plan_steps_use_global_unique_indices(self):
        steps = default_plan_steps([11, 12])
        self.assertEqual([step["step_index"] for step in steps], list(range(6)))
        self.assertEqual([step["config_id"] for step in steps], [11, 11, 11, 12, 12, 12])
        self.assertEqual(
            [step["action"] for step in steps[:3]],
            [
                "TRAIN_MODEL",
                "RUN_OOT_BACKTEST",
                "RUN_POLYMARKET_OOT",
            ],
        )

    def test_report_ranks_by_median_polymarket_pnl_not_auc(self):
        results = [
            _result(
                11,
                pnl=1.2,
                trades=25,
                drawdown=2.0,
                artifact_id=101,
                auc=0.51,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                11,
                pnl=1.4,
                trades=25,
                drawdown=1.5,
                artifact_id=101,
                auc=0.52,
                window_start="2026-08-06",
                window_end="2026-08-10",
            ),
            _result(
                11,
                pnl=1.0,
                trades=24,
                drawdown=2.2,
                artifact_id=101,
                auc=0.53,
                window_start="2026-08-11",
                window_end="2026-08-15",
            ),
            _result(
                12,
                pnl=0.2,
                trades=30,
                drawdown=4.0,
                artifact_id=102,
                auc=0.99,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                12,
                pnl=0.1,
                trades=30,
                drawdown=4.2,
                artifact_id=102,
                auc=0.98,
                window_start="2026-08-06",
                window_end="2026-08-10",
            ),
            _result(
                12,
                pnl=0.3,
                trades=30,
                drawdown=3.8,
                artifact_id=102,
                auc=0.97,
                window_start="2026-08-11",
                window_end="2026-08-15",
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertEqual(report["recommended_config_id"], 11)
        self.assertEqual(report["recommendation_status"], "READY_FOR_SHADOW")
        self.assertEqual(report["rejection_reasons"], [])

    def test_report_does_not_recommend_without_polymarket_pnl_sample(self):
        results = [
            _result(
                11,
                kind="TRAIN",
                pnl=10.0,
                trades=100,
                drawdown=1.0,
                artifact_id=101,
                auc=0.85,
            ),
            _result(
                11,
                kind="OOT",
                pnl=5.0,
                trades=50,
                drawdown=2.0,
                artifact_id=101,
                auc=0.80,
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertIsNone(report["recommended_config_id"])
        self.assertEqual(report["recommendation_status"], "NO_PNL_SAMPLE")

    def test_report_ignores_failed_polymarket_evaluations(self):
        results = [
            _result(
                11,
                status="FAILED",
                pnl=-10.0,
                trades=50,
                drawdown=10.0,
                artifact_id=101,
                window_start="2026-08-01",
                window_end="2026-08-05",
            )
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertIsNone(report["recommended_config_id"])
        self.assertEqual(report["recommendation_status"], "NO_PNL_SAMPLE")

    def test_insufficient_data_closes_queue_step_as_skipped(self):
        self.assertIn("INSUFFICIENT_DATA", RESULT_CLOSING_STATUSES)

    def test_finalization_gate_rejects_insufficient_trades(self):
        results = [
            _result(
                11,
                pnl=1.0,
                trades=10,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                11,
                pnl=1.0,
                trades=10,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-06",
                window_end="2026-08-10",
            ),
            _result(
                11,
                pnl=1.0,
                trades=10,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-11",
                window_end="2026-08-15",
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertIsNone(report["recommended_config_id"])
        self.assertEqual(report["recommendation_status"], "INSUFFICIENT_TRADES")
        self.assertIn("INSUFFICIENT_TRADES", report["rejection_reasons"])

    def test_finalization_gate_rejects_insufficient_windows(self):
        results = [
            _result(
                11,
                pnl=1.0,
                trades=30,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                11,
                pnl=1.0,
                trades=30,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-06",
                window_end="2026-08-10",
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertIsNone(report["recommended_config_id"])
        self.assertEqual(report["recommendation_status"], "INSUFFICIENT_WINDOWS")
        self.assertIn("INSUFFICIENT_WINDOWS", report["rejection_reasons"])

    def test_finalization_gate_rejects_non_positive_pnl(self):
        results = [
            _result(
                11,
                pnl=-0.5,
                trades=20,
                drawdown=3.0,
                artifact_id=101,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                11,
                pnl=0.0,
                trades=20,
                drawdown=2.0,
                artifact_id=101,
                window_start="2026-08-06",
                window_end="2026-08-10",
            ),
            _result(
                11,
                pnl=-0.1,
                trades=20,
                drawdown=2.5,
                artifact_id=101,
                window_start="2026-08-11",
                window_end="2026-08-15",
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertIsNone(report["recommended_config_id"])
        self.assertEqual(report["recommendation_status"], "NON_POSITIVE_PNL")
        self.assertIn("NON_POSITIVE_PNL", report["rejection_reasons"])

    def test_finalization_gate_rejects_invalid_nan_pnl(self):
        res = evaluate_finalization_gate(
            {
                "polymarket_oot_evaluation_count": 3,
                "window_count": 3,
                "total_trades": 60,
                "median_oot_pnl": float("nan"),
                "median_oot_drawdown": 1.0,
            }
        )
        self.assertFalse(res["eligible"])
        self.assertIn("INVALID_RESULT", res["rejection_reasons"])

    def test_finalization_gate_accepts_valid_candidate_with_contract_fields(self):
        results = [
            _result(
                11,
                pnl=1.0,
                trades=25,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                11,
                pnl=1.2,
                trades=25,
                drawdown=1.2,
                artifact_id=101,
                window_start="2026-08-06",
                window_end="2026-08-10",
            ),
            _result(
                11,
                pnl=1.5,
                trades=24,
                drawdown=1.1,
                artifact_id=101,
                window_start="2026-08-11",
                window_end="2026-08-15",
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertEqual(report["recommended_config_id"], 11)
        self.assertEqual(report["recommendation_status"], "READY_FOR_SHADOW")
        self.assertEqual(report["rejection_reasons"], [])
        self.assertEqual(report["window_count"], 3)
        self.assertEqual(report["total_trades"], 74)
        self.assertEqual(report["median_pnl"], 1.2)

    def test_plan_run_preserves_positional_api_contract(self):
        params = list(inspect.signature(plan_run).parameters.values())
        self.assertEqual([p.name for p in params[:3]], ["session", "run_id", "config_ids"])

    def test_report_rejects_duplicate_oot_window_without_inflating_trades(self):
        results = [
            _result(
                11,
                pnl=1.0,
                trades=25,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                11,
                pnl=1.0,
                trades=25,
                drawdown=1.0,
                artifact_id=101,
                window_start="2026-08-01",
                window_end="2026-08-05",
            ),
            _result(
                11,
                pnl=1.5,
                trades=25,
                drawdown=1.1,
                artifact_id=101,
                window_start="2026-08-11",
                window_end="2026-08-15",
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertIsNone(report["recommended_config_id"])
        self.assertIn("INVALID_RESULT", report["rejection_reasons"])

    def test_report_tolerates_non_mapping_diagnostic_metrics(self):
        res = _result(
            11,
            pnl=1.0,
            trades=60,
            drawdown=1.0,
            artifact_id=101,
            window_start="2026-08-01",
            window_end="2026-08-05",
        )
        res.metrics = "corrupted_string"
        results = [
            res,
            _result(
                11,
                pnl=1.2,
                trades=20,
                drawdown=1.2,
                artifact_id=101,
                window_start="2026-08-06",
                window_end="2026-08-10",
            ),
            _result(
                11,
                pnl=1.5,
                trades=20,
                drawdown=1.1,
                artifact_id=101,
                window_start="2026-08-11",
                window_end="2026-08-15",
            ),
        ]
        report = build_experiment_report(results, min_trades=50, min_windows=3)
        self.assertEqual(report["recommended_config_id"], 11)


if __name__ == "__main__":
    unittest.main()
