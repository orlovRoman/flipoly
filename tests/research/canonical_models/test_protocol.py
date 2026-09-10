from polyflip.research.canonical_models.protocol import (
    attach_hash, load_protocol, protocol_hash,
)


def test_protocol_pins():
    p = load_protocol()
    assert p["assets"] == ["BTC", "ETH", "SOL", "XRP", "DOGE"]
    assert p["decision_point"]["max_lateness_sec"] == 15
    assert p["trading_universe"]["ask_min_inclusive"] == 0.01
    assert p["trading_universe"]["ask_max_inclusive"] == 0.40
    assert p["trading_universe"]["budget_usd"] == 1.0
    assert p["trading_universe"]["budget_fee_included"] is True
    ids = [v["id"] for v in p["forecast_variants"] if v["id"] != "CT"]
    assert ids == ["market", "M0", "M1", "M2", "M3", "M4"]


def test_protocol_hash_stable_and_attached():
    h1, h2 = protocol_hash(), protocol_hash()
    assert h1 == h2 and len(h1) == 64
    out = attach_hash({"n": 1})
    assert out["protocol_hash"] == h1
