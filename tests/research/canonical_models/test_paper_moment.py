import json

import pandas as pd

from polyflip.research.canonical_models.paper_moment import run

DEC = pd.Timestamp("2026-09-09T18:40:00Z")
SPEC = {"spec_id": "BTC_CT_T5_V1",
        "spec_hash": "a14c4a4b81c3743454842007dc8f2d89e055c93e010ea63c6c73a711cca31c16",
        "ct_regime": "REVERSION"}


def _paper():
    return pd.DataFrame([{
        "market_id": "m1", "decision_at": DEC, "action": "BUY", "side": "UP",
        "reason": "CT_SIGNAL_REVERSION", "limit_price": 0.09,
        "trade_history_id": 1, "decision_details": json.dumps(SPEC)}])


def _live():
    return pd.DataFrame([{"market_id": "m1", "asset": "BTC",
                          "end_time_est": "2026-09-09T18:45:00Z",
                          "yes_token_id": "u", "no_token_id": "d"}])


def _snaps(up_age_s=5.0, ask=0.09):
    return pd.DataFrame([{
        "market_id": "m1", "recorded_at": DEC - pd.Timedelta(seconds=up_age_s),
        "mid_price": 0.08, "best_ask": ask, "final_outcome": "NO",
        "best_bid": 0.07}])


def _depth(down_age_s=5.0, ask=0.9):
    return pd.DataFrame([{
        "market_id": "m1", "outcome_side": "NO",
        "event_at": DEC - pd.Timedelta(seconds=down_age_s),
        "received_at": DEC - pd.Timedelta(seconds=down_age_s),
        "best_bid_price": round(ask - 0.01, 4), "best_ask_price": ask,
        "asks": json.dumps([{"price": ask, "size": 100.0}]),
        "is_truncated": False}])


def test_missing_book():
    df = run(_paper(), _snaps(), _depth().iloc[0:0], _live())
    assert df.iloc[0]["why"] == "MISSING_BOOK"


def test_stale_beats_range():
    # ask 0.9 is out of range AND book is stale -> STALE first (precedence)
    df = run(_paper(), _snaps(up_age_s=30.0), _depth(down_age_s=30.0), _live())
    assert df.iloc[0]["why"] == "STALE_BOOK"


def test_fresh_out_of_range_proven():
    snaps = _snaps(up_age_s=5.0, ask=0.91)
    snaps["mid_price"] = 0.90
    df = run(_paper(), snaps, _depth(down_age_s=5.0, ask=0.50), _live())
    assert df.iloc[0]["why"] == "PRICE_OUT_OF_RANGE"


def test_fresh_usable_and_ct_source_labelled():
    from polyflip.research.canonical_models.paper_moment import summarize
    df = run(_paper(), _snaps(up_age_s=5.0, ask=0.09),
             _depth(down_age_s=5.0, ask=0.90), _live())
    assert bool(df.iloc[0]["usable"]) is True
    assert df.iloc[0]["side"] == "UP"
    s = summarize(df)
    assert "REGISTERED_PAPER" in s["ct_source"]
    assert s["tiers"]["L1_sufficient_same_book"] == 1
