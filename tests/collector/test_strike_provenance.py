from datetime import datetime, timezone
import pytest

from polyflip.collector.client import StrikeProvenance, _canonical_strike_provenance
from polyflip.db.models import LiveMarket, MarketSnapshot


def test_strike_provenance_dataclass_contract():
    """Verify StrikeProvenance properties and float conversion semantics."""
    now = datetime.now(timezone.utc)
    prov = StrikeProvenance(
        strike_value=64500.25,
        strike_source="market.underlying_price",
        strike_effective_at=now,
        strike_received_at=now,
    )
    assert prov.strike_value == 64500.25
    assert prov.strike_source == "market.underlying_price"
    assert prov.strike_effective_at == now
    assert prov.strike_received_at == now
    assert float(prov) == 64500.25

    # None strike value converts safely to 0.0
    prov_none = StrikeProvenance(
        strike_value=None,
        strike_source="UNKNOWN",
        strike_effective_at=None,
        strike_received_at=now,
    )
    assert prov_none.strike_value is None
    assert float(prov_none) == 0.0


def test_canonical_strike_provenance_hierarchy_and_timestamps():
    """Verify priority extraction and ISO timestamp parsing in _canonical_strike_provenance."""
    # 1. Direct market.underlying_price priority
    market1 = {
        "underlying_price": "60123.50",
        "startDate": "2026-09-08T12:00:00Z",
    }
    event1 = {
        "strikePrice": "59000.00",
        "startDate": "2026-09-08T11:00:00Z",
    }
    prov1 = _canonical_strike_provenance(market1, event1)
    assert prov1.strike_value == 60123.50
    assert prov1.strike_source == "market.underlying_price"
    assert prov1.strike_effective_at == datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

    # 2. Event level fallback when market level is absent
    market2 = {"startDate": "2026-09-08T12:00:00Z"}
    event2 = {"priceToBeat": 61200.0}
    prov2 = _canonical_strike_provenance(market2, event2)
    assert prov2.strike_value == 61200.0
    assert prov2.strike_source == "event.priceToBeat"

    # 3. Missing strike falls back to UNKNOWN
    prov3 = _canonical_strike_provenance({}, {})
    assert prov3.strike_value is None
    assert prov3.strike_source == "UNKNOWN"


def test_db_models_strike_provenance_columns():
    """Verify MarketSnapshot and LiveMarket persist strike provenance metadata."""
    now = datetime.now(timezone.utc)
    snap = MarketSnapshot(
        asset="BTC",
        market_id="m_test",
        time_left_min=10.0,
        mid_price=0.5,
        spread=0.01,
        volume_5min=100.0,
        price_velocity=0.0,
        hour_of_day=12,
        final_outcome="PENDING",
        recorded_at=now,
        strike_value=60500.0,
        strike_source="market.underlying_price",
        strike_effective_at=now,
        strike_received_at=now,
    )
    assert snap.strike_value == 60500.0
    assert snap.strike_source == "market.underlying_price"
    assert snap.strike_effective_at == now

    live = LiveMarket(
        market_id="m_test",
        asset="BTC",
        yes_token_id="y_1",
        no_token_id="n_1",
        end_time_est=now,
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.01,
        volume_5min=100.0,
        last_updated=now,
        strike_value=60500.0,
        strike_source="market.underlying_price",
        strike_effective_at=now,
        strike_received_at=now,
    )
    assert live.strike_value == 60500.0
    assert live.strike_source == "market.underlying_price"
