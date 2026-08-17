from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from polyflip.crypto.polymarket_evaluation_protocol import (
    CANONICAL_EVALUATION_PROTOCOL,
    evaluate_logreg_with_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "polyflip" / "config" / "logreg_evaluation_protocol_v1.json"
REPORT_PATH = ROOT / "reports" / "logreg_evaluation_protocol_20260817.json"


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    recorded_at = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:01Z"])
    frame = pd.DataFrame(
        {
            "market_id": ["m1", "m2"],
            "recorded_at": recorded_at,
            "mid_price": [0.30, 0.70],
            "final_outcome": ["YES", "NO"],
        }
    )
    quotes = pd.DataFrame(
        {
            "market_id": ["m1", "m2"],
            "yes_price": [0.30, 0.70],
            "no_price": [0.70, 0.30],
            "mid_price": [0.30, 0.70],
            "final_outcome": ["YES", "NO"],
            "recorded_at": recorded_at,
        }
    )
    return frame, quotes


def _digest(result: dict) -> str:
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_config_matches_immutable_protocol_and_report_is_offline_only():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert config["protocol_version"] == CANONICAL_EVALUATION_PROTOCOL.protocol_version
    assert {
        key: config[key]
        for key in CANONICAL_EVALUATION_PROTOCOL.as_dict()
    } == CANONICAL_EVALUATION_PROTOCOL.as_dict()
    assert report["protocol"] == CANONICAL_EVALUATION_PROTOCOL.as_dict()
    assert report["execution_scope"] == "offline_replay_only"
    assert report["production_activation"] is False
    assert report["model_execution"] is False


def test_two_replay_runs_have_identical_canonical_result():
    frame, quotes = _fixture()
    first = evaluate_logreg_with_protocol(frame, [0.8, 0.8], quotes)
    second = evaluate_logreg_with_protocol(frame, [0.8, 0.8], quotes)
    first_digest = _digest(first)
    second_digest = _digest(second)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    probe = report["deterministic_probe"]

    assert first == second
    assert first_digest == second_digest
    assert probe["run_1_sha256"] == first_digest
    assert probe["run_2_sha256"] == second_digest
    assert probe["equal"] is True
