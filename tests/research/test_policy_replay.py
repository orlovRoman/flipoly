import pytest

from polyflip.research.policy_replay import replay_opportunities


def _row(**overrides):
    row = {
        "opportunity_id": "o1",
        "market_id": "m1",
        "asset": "BTC",
        "decision_at": "2026-09-01T00:00:00+00:00",
        "time_left_sec": 240,
        "yes_ask": 0.20,
        "no_ask": 0.80,
        "yes_mid": 0.21,
        "no_mid": 0.79,
        "p_model_yes": 0.80,
        "final_outcome": "YES",
    }
    row.update(overrides)
    return row


def test_replay_keeps_all_opportunities_and_does_not_zero_missing_data():
    rows, summary = replay_opportunities(
        [_row(), _row(opportunity_id="o2", yes_ask="", p_model_yes="")]
    )
    assert summary["opportunities"] == 2
    assert summary["same_opportunity_set"] is True
    assert any(item["status"] == "INSUFFICIENT_DATA" for item in rows)
    assert all(item["net_pnl"] is None for item in rows if item["status"] == "INSUFFICIENT_DATA")


def test_replay_reports_fixed_role_variants_without_threshold_search():
    rows, summary = replay_opportunities([_row()])
    assert set(summary["variants"]) == {"CURRENT", "STRICT", "FAVORITE_ONLY", "OUTSIDER_ONLY"}
    assert summary["variants"]["FAVORITE_ONLY"]["status_counts"].get("BUY", 0) == 0
    assert summary["variants"]["OUTSIDER_ONLY"]["status_counts"].get("BUY", 0) == 1


def test_replay_rejects_duplicate_opportunity_ids():
    with pytest.raises(ValueError, match="unique"):
        replay_opportunities([_row(), _row()])


def test_replay_rejects_non_causal_label_timestamp():
    with pytest.raises(ValueError, match="precedes"):
        replay_opportunities(
            [_row(label_available_at="2026-08-31T23:59:00+00:00")]
        )


def test_replay_requires_timezone_on_decision_timestamp():
    with pytest.raises(ValueError, match="invalid causal timestamp"):
        replay_opportunities([_row(decision_at="2026-09-01T00:00:00")])
