from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from polyflip.crypto.logreg_replay import (
    classification_metrics,
    first_snapshot_per_market,
    split_market_windows,
)
from polyflip.scripts.replay_logreg_candidates import write_reports


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["m1", "m1", "m2", "m3"],
            "recorded_at": pd.to_datetime([
                "2026-01-01T00:01Z", "2026-01-01T00:00Z",
                "2026-01-02T00:00Z", "2026-01-03T00:00Z",
            ]),
            "mid_price": [0.40, 0.40, 0.60, 0.40],
            "final_outcome": ["YES", "YES", "NO", "YES"],
        }
    )


def test_replay_windows_are_chronological_and_disjoint():
    result = split_market_windows(
        _frame(),
        {"m1": "2026-01-01T01:00Z", "m2": "2026-01-02T01:00Z", "m3": "2026-01-03T01:00Z"},
    )
    assert result == {"T1": {"m1"}, "T2": {"m2"}, "T3": {"m3"}}
    assert not result["T1"] & result["T2"]


def test_first_snapshot_and_classification_metrics_are_reproducible():
    frame = _frame()
    selected, p_yes = first_snapshot_per_market(frame, [0.8, 0.7, 0.2, 0.9])
    assert list(selected["market_id"]) == ["m1", "m2", "m3"]
    assert p_yes.tolist() == pytest.approx([0.7, 0.8, 0.9])
    first = classification_metrics(selected, p_yes)
    second = classification_metrics(selected, p_yes)
    assert first == second
    assert first["n_scored"] == 3
    assert first["brier"] is not None
    assert first["ece"] is not None
    assert first["log_loss"] is not None


def test_concrete_candidate_report_round_trip_preserves_metrics(tmp_path):
    metric = {
        "coverage_pct": 75.0,
        "n_trades": 3,
        "win_rate": 2 / 3,
        "net_profit": 0.42,
        "roi_pct": 14.0,
        "max_drawdown_usdc": 1.0,
        "brier": 0.12,
        "ece": 0.08,
        "log_loss": 0.31,
    }
    records = []
    for model_id in range(820, 880):
        records.append(
            {
                "model_registry_id": model_id,
                "model_version": 48,
                "oof_artifact_id": 80 + model_id - 820,
                "artifact_status": "VALID_FOR_REPLAY",
                "evaluation_commit": "02c9ed7",
                "metrics_schema_version": "canonical_pnl_v1",
                "evaluation_protocol_version": "polymarket_logreg_eval_v1",
                "evaluations": {
                    "RAW": {"windows": {"T1": {"COMBINED": metric}}},
                },
            }
        )
    payload = {
        "report_version": "logreg_candidate_replay_v2",
        "candidate_count_observed": 60,
        "records": records,
    }

    write_reports(payload, tmp_path)

    saved = json.loads((tmp_path / "logreg_candidate_replay_20260817_v2.json").read_text())
    with (tmp_path / "logreg_candidate_replay_20260817_v2.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert saved["records"][0]["model_registry_id"] == 820
    assert saved["records"][0]["oof_artifact_id"] == 80
    assert rows[0]["strategy_branch"] == "COMBINED"
    assert float(rows[0]["net_profit"]) == pytest.approx(0.42)
    assert len(rows) == 60
