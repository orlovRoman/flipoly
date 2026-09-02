import json
from datetime import datetime, timedelta, timezone

from scripts.weighted_policy_shadow_evidence import (
    summarize_live_rows,
    summarize_shadow_rows,
)


def _payload(p: float, side: str, ask: float) -> str:
    return json.dumps(
        {
            name: {
                "p_final_yes": p,
                "selected_side": side,
                "selected_ask": ask,
            }
            for name in (
                "MARKET_ONLY",
                "MARKET_LOGREG",
                "MARKET_LGBM",
                "FULL_WEIGHTED",
                "FULL_WEIGHTED_MRF",
                "OUTSIDER_AGREE_ONLY",
            )
        }
    )


def test_shadow_summary_reports_counts_quality_and_all_arm_coverage():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "market_id": "m-1",
            "created_at": start,
            "outcome_yes": "YES",
            "weighted_p_market_yes": 0.60,
            "p_logreg_yes": 0.58,
            "weighted_selected_side": "BUY_YES",
            "weighted_cost_per_share": 0.01,
            "candidate_ask": 0.40,
            "legacy_action": "BUY_YES",
            "weighted_benchmark_json": _payload(0.65, "BUY_YES", 0.40),
        },
        {
            "market_id": "m-2",
            "created_at": start + timedelta(days=14),
            "outcome_yes": "NO",
            "weighted_p_market_yes": 0.40,
            "p_logreg_yes": 0.42,
            "weighted_selected_side": "BUY_NO",
            "weighted_cost_per_share": 0.01,
            "candidate_ask": 0.40,
            "legacy_action": "BUY_NO",
            "weighted_benchmark_json": _payload(0.35, "BUY_NO", 0.40),
        },
    ]
    result = summarize_shadow_rows(rows)
    assert result["shadow_days"] == 14.0
    assert result["shadow_resolved_markets"] == 2
    assert result["shadow_candidate_trades"] == 2
    assert result["weighted_brier"] is not None
    assert result["market_brier"] is not None
    assert result["legacy_brier"] is not None
    assert result["pnl_ci_lower"] is not None
    assert result["calibration_error"] is not None
    assert result["telemetry"]["rows_with_benchmark"] == 2
    assert result["telemetry"]["arm_coverage"]["FULL_WEIGHTED_MRF"] == 2


def test_live_summary_measures_expected_realized_price_drag():
    result = summarize_live_rows(
        [
            {
                "weighted_expected_execution_price": 0.40,
                "executed_price": 0.41,
            },
            {
                "weighted_expected_execution_price": 0.50,
                "executed_price": 0.49,
            },
        ]
    )
    assert result["live_fills"] == 2
    assert result["expected_realized_samples"] == 2
    assert result["execution_drag"] == 0.01

def test_shadow_summary_reports_policy_identity_and_mixed_ids():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"market_id": "m-1", "created_at": start, "weighted_policy_id": "artifact-a"},
        {
            "market_id": "m-2",
            "created_at": start + timedelta(days=1),
            "weighted_policy_id": "artifact-a",
        },
    ]
    result = summarize_shadow_rows(rows)
    assert result["policy_id"] == "artifact-a"
    assert result["policy_ids"] == ["artifact-a"]

    mixed = summarize_shadow_rows(
        [
            {"market_id": "m-1", "created_at": start, "weighted_policy_id": "artifact-a"},
            {"market_id": "m-2", "created_at": start, "weighted_policy_id": "artifact-b"},
        ]
    )
    assert mixed["policy_id"] is None
    assert mixed["policy_ids"] == ["artifact-a", "artifact-b"]

    live = summarize_live_rows(
        [{"weighted_policy_id": "artifact-a", "executed_price": 0.4}]
    )
    assert live["policy_id"] == "artifact-a"
    assert live["policy_ids"] == ["artifact-a"]
