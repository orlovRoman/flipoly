import pandas as pd

from polyflip.research.canonical_models.registry import build_registry, coverage_gate


def _live():
    return pd.DataFrame([
        {"market_id": "m1", "asset": "BTC", "yes_token_id": "u1", "no_token_id": "d1",
         "end_time_est": "2026-09-09T00:15:00Z", "market_start_at": None, "market_end_at": None,
         "strike_value": float("nan"), "strike_source": None, "settlement_price_source": "chainlink",
         "resolution_source": None, "resolved_at": None},
        {"market_id": "m2", "asset": "BTC", "yes_token_id": "u2", "no_token_id": "u2",
         "end_time_est": "2026-09-09T00:30:00Z", "market_start_at": None, "market_end_at": None,
         "strike_value": float("nan"), "strike_source": None, "settlement_price_source": None,
         "resolution_source": None, "resolved_at": None},
        {"market_id": "m3", "asset": "ETH", "yes_token_id": "u3", "no_token_id": "d3",
         "end_time_est": "2026-09-09T00:30:00Z", "market_start_at": None, "market_end_at": None,
         "strike_value": float("nan"), "strike_source": None, "settlement_price_source": None,
         "resolution_source": None, "resolved_at": None},
    ])


def _snaps():
    return pd.DataFrame([
        {"market_id": "m1", "recorded_at": "2026-09-09T00:10:00Z", "final_outcome": "YES"},
        {"market_id": "m1", "recorded_at": "2026-09-09T00:16:00Z", "final_outcome": "YES"},
        {"market_id": "m3", "recorded_at": "2026-09-09T00:10:00Z", "final_outcome": "PENDING"},
    ])


def test_registry_unique_mapping_and_separation():
    reg, issues = build_registry(_live(), _snaps())
    assert list(reg["market_id"]) == ["m1"]  # m2 ambiguous map, m3 unresolved
    assert reg["market_id"].is_unique
    reasons = sorted(issues["reason"].str.split(":").str[0])
    assert reasons == ["AMBIGUOUS_TOKEN_MAPPING", "UNRESOLVED"]


def test_gate_reconciles_and_flags_strike():
    reg, issues = build_registry(_live(), _snaps())
    v = coverage_gate(reg, issues)
    assert v["n_registry"] + v["n_issues"] == 3
    assert v["n_canonical_strike"] == 0
    assert v["model_track"].startswith("DATA_BLOCKED")
    assert v["ct_track"] == "FEASIBLE"
