import pytest
from polyflip.collector.resolver import extract_final_outcome
from polyflip.db.models import MarketSnapshot
from datetime import datetime, timezone

@pytest.mark.parametrize("market_data, expected", [
    ({"answer": "Yes"}, "YES"),
    ({"answer": "Up"}, "YES"),
    ({"winnerOutcome": "Down"}, "NO"),
    ({"resolvedBy": "0x123..."}, None),
    # Closed markets use terminal prices if no answer
    ({"closed": True, "outcomePrices": ["1", "0"], "outcomes": ["Yes", "No"]}, "YES"),
    ({"closed": True, "outcomePrices": ["0", "1"], "outcomes": ["Yes", "No"]}, "NO"),
    ({"closed": True, "outcomePrices": ["0.52", "0.48"], "outcomes": ["Yes", "No"]}, None),
    ({"closed": True, "outcomePrices": "[\"1\", \"0\"]", "outcomes": "[\"Yes\", \"No\"]"}, "YES"),
    # Open markets should NOT use terminal prices
    ({"closed": False, "outcomePrices": ["1", "0"], "outcomes": ["Yes", "No"]}, None),
    # Explicit answer wins
    ({"answer": "INVALID"}, "INVALID"),
    ({"answer": "unknown value"}, None),
    ({"answer": "No"}, "NO"),
    # Malformed strings/types
    ({"answer": {"some": "dict"}}, None),
    ({"answer": True}, None),
])
def test_extract_final_outcome(market_data, expected):
    assert extract_final_outcome(market_data) == expected

@pytest.mark.asyncio
async def test_invalid_commits_successfully(db_session):
    # Ensure INVALID outcome with flip_vs_final=None can be committed
    snap = MarketSnapshot(
        market_id="test_invalid",
        asset="BTC",
        time_left_min=10,
        best_bid=0.49,
        best_ask=0.51,
        mid_price=0.5,
        spread=0.01,
        volume_5min=100.0,
        price_velocity=0.0,
        hour_of_day=12,
        final_outcome="INVALID",
        flip_vs_final=None,
        recorded_at=datetime.now(timezone.utc)
    )
    db_session.add(snap)
    await db_session.commit()
    
    # Verify
    saved = await db_session.get(MarketSnapshot, snap.id)
    assert saved.final_outcome == "INVALID"
    assert saved.flip_vs_final is None


