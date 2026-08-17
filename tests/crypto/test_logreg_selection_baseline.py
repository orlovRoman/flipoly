from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def test_candidate_snapshot_is_read_only_and_reproducible():
    snapshot = json.loads(
        (REPORTS / "logreg_candidates_820_879_snapshot_20260817.json").read_text()
    )
    baseline = json.loads(
        (REPORTS / "logreg_selection_baseline_20260817.json").read_text()
    )
    candidates = snapshot["candidates"]

    assert snapshot["candidate_count"] == 60
    assert len(candidates) == 60
    assert all(item["model_registry"]["is_active"] is False for item in candidates)
    assert all(item["model_registry"].get("activation_source") is None for item in candidates)
    assert all(item["model_registry"].get("activated_at") is None for item in candidates)
    assert all(item["oof_artifacts"] for item in candidates)
    assert baseline["candidate_safety_checks"] == {
        "candidate_count": 60,
        "all_inactive": True,
        "all_activation_source_null": True,
        "all_activated_at_null": True,
        "all_have_oof": True,
    }


def test_snapshot_sql_is_a_read_only_candidate_query():
    sql = (REPORTS / "logreg_candidates_820_879_snapshot_20260817.sql").read_text()
    lowered = sql.lower()
    assert "select" in lowered
    assert "between 820 and 879" in lowered
    assert "delete" not in lowered
    assert "update" not in lowered
    assert "insert" not in lowered
