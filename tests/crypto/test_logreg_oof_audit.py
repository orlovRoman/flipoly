from __future__ import annotations

import pandas as pd

from polyflip.crypto.logreg_oof_audit import (
    build_audit_report,
    classify_oof_artifact,
    recover_close_time_sources,
)


def _frame(*, close_time: bool = False) -> pd.DataFrame:
    data = {
        "market_id": ["m1", "m1"],
        "recorded_at": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:01Z"]),
        "mid_price": [0.4, 0.41],
        "final_outcome": ["YES", "YES"],
    }
    if close_time:
        data["market_close_at"] = pd.to_datetime(["2026-01-01T01:00Z"] * 2)
    return pd.DataFrame(data)


def test_close_time_recovery_uses_unique_market_id_join():
    result = recover_close_time_sources(
        _frame(),
        {"m1": [{"end_date": None, "resolved_at": None, "end_time_est": "2026-01-01T01:00Z"}]},
    )
    assert result["status"] == "RECOVERED"
    assert result["join_key"] == "market_id"
    assert result["source_counts"] == {"end_time_est": 1}
    assert result["missing_count"] == 0


def test_missing_close_time_is_invalid_and_requires_retrain():
    result = classify_oof_artifact(
        model_registry_id=820,
        model_version=1,
        oof_artifact_id=80,
        artifact_blob=b"artifact",
        schema_version=2,
        row_count=2,
        frame=_frame(),
        quotes=pd.DataFrame({"market_id": ["m1"], "mid_price": [0.4]}),
        raw_scores=[0.2, 0.3],
        calibrated_scores=[0.25, 0.35],
        live_markets={"m1": []},
        evaluation_commit="test",
        metrics_schema_version="canonical_pnl_v1",
    )
    assert result["artifact_status"] == "INVALID_OOT_ARTIFACT"
    assert result["invalid_reason"] == "MISSING_CLOSE_TIME"
    assert result["retrain_required"] is True
    assert result["replay_allowed"] is False


def test_direct_close_time_is_valid_for_replay():
    result = classify_oof_artifact(
        model_registry_id=820,
        model_version=1,
        oof_artifact_id=80,
        artifact_blob=b"artifact",
        schema_version=2,
        row_count=2,
        frame=_frame(close_time=True),
        quotes=pd.DataFrame({"market_id": ["m1"], "mid_price": [0.4]}),
        raw_scores=[0.2, 0.3],
        calibrated_scores=[0.25, 0.35],
        live_markets={},
        evaluation_commit="test",
        metrics_schema_version="canonical_pnl_v1",
    )
    assert result["artifact_status"] == "VALID_FOR_REPLAY"
    assert result["replay_allowed"] is True
    assert result["close_time_recovery"]["direct_count"] == 1


def test_report_counts_both_classifications():
    records = [
        {"artifact_status": "VALID_FOR_REPLAY", "retrain_required": False, "replay_allowed": True, "close_time_recovery": {"market_count": 1, "missing_count": 0, "ambiguous_count": 0}},
        {"artifact_status": "INVALID_OOT_ARTIFACT", "retrain_required": True, "replay_allowed": False, "close_time_recovery": {"market_count": 1, "missing_count": 1, "ambiguous_count": 0}},
    ]
    report = build_audit_report(records, generated_at="now", evaluation_commit="test", metrics_schema_version="canonical_pnl_v1")
    assert report["counts"] == {
        "total": 2,
        "valid_for_replay": 1,
        "invalid_oot_artifact": 1,
        "close_time_recovered_markets": 1,
        "close_time_missing_markets": 1,
        "close_time_ambiguous_markets": 0,
        "retrain_required": 1,
        "replay_allowed": 1,
    }
