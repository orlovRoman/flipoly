# tests/test_market_replay.py
def test_group_respects_min_snapshots_param():
    from polyflip.backtesting.market_replay import group_snapshots_into_replays
    from unittest.mock import MagicMock

    def make_snap(market_id, outcome="YES"):
        s = MagicMock()
        s.market_id = market_id
        s.asset = "BTC"
        s.final_outcome = outcome
        s.mid_price = 0.7
        s.spread = 0.02
        s.volume_5min = 0.0
        s.price_velocity = 0.0
        s.hour_of_day = 12
        s.time_left_min = 10.0
        s.recorded_at = None
        return s

    # 2 снимка — при min=3 должно быть пропущено, при min=2 — принято
    snaps = [make_snap("m1"), make_snap("m1")]
    assert group_snapshots_into_replays(snaps, min_snapshots=3) == {}
    assert "m1" in group_snapshots_into_replays(snaps, min_snapshots=2)
