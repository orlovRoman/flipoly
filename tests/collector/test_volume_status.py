from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from polyflip.collector.client import PolymarketClient, VolumeResult
from polyflip.db.models import LiveMarket, MarketSnapshot


def test_volume_result_contract_and_type_safety():
    """Verify VolumeResult float coercion, equality guards, and status reporting."""
    now = datetime.now(timezone.utc)

    # 1. Valid volume
    v_valid = VolumeResult(volume=5432.1, status="VALID", timestamp=now)
    assert float(v_valid) == 5432.1
    assert v_valid == 5432.1
    assert v_valid.status == "VALID"

    # 2. Unavailable volume (AUTH_REQUIRED / HTTP_ERROR / UNAVAILABLE)
    v_auth = VolumeResult(volume=None, status="AUTH_REQUIRED", timestamp=now)
    assert v_auth.status == "AUTH_REQUIRED"
    assert v_auth.volume is None
    with pytest.raises(TypeError, match="Cannot cast unavailable VolumeResult to float"):
        float(v_auth)
    # Equality must safely return False rather than erroring or falsely matching 0
    assert (v_auth == 0.0) is False
    assert (v_auth == 100.0) is False

    # 3. Valid zero volume (no trades during window)
    v_zero = VolumeResult(volume=0.0, status="VALID_ZERO", timestamp=now)
    assert float(v_zero) == 0.0
    assert v_zero == 0.0
    assert v_zero.status == "VALID_ZERO"


@pytest.mark.asyncio
async def test_get_recent_trades_volume_status_variants():
    """Verify client.get_recent_trades_volume returns typed status for all response types."""
    client = PolymarketClient.__new__(PolymarketClient)
    client.client = MagicMock()
    client.CLOB_API = "https://clob.polymarket.com"

    # 1. Successful non-empty trades
    mock_resp_success = MagicMock()
    mock_resp_success.status_code = 200
    mock_resp_success.json.return_value = {
        "data": [
            {"size": "100.0", "timestamp": str(int(datetime.now(timezone.utc).timestamp()))},
            {"size": "50.5", "timestamp": str(int(datetime.now(timezone.utc).timestamp()))},
        ]
    }
    client.client.get = AsyncMock(return_value=mock_resp_success)
    res_success = await client.get_recent_trades_volume("token_1", minutes=5)
    assert res_success.status == "VALID"
    assert res_success.volume == 150.5

    # 2. Empty trades list -> VALID_ZERO
    mock_resp_empty = MagicMock()
    mock_resp_empty.status_code = 200
    mock_resp_empty.json.return_value = {"data": []}
    client.client.get = AsyncMock(return_value=mock_resp_empty)
    res_empty = await client.get_recent_trades_volume("token_1", minutes=5)
    assert res_empty.status == "VALID_ZERO"
    assert res_empty.volume == 0.0
    assert float(res_empty) == 0.0

    # 3. Auth required (401 / 403)
    mock_resp_auth = MagicMock()
    mock_resp_auth.status_code = 401
    client.client.get = AsyncMock(return_value=mock_resp_auth)
    res_auth = await client.get_recent_trades_volume("token_1", minutes=5)
    assert res_auth.status == "AUTH_REQUIRED"
    assert res_auth.volume is None

    # 4. HTTP 500 error
    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    client.client.get = AsyncMock(return_value=mock_resp_500)
    res_500 = await client.get_recent_trades_volume("token_1", minutes=5)
    assert res_500.status == "HTTP_ERROR"
    assert res_500.volume is None

    # 5. Network transport exception
    client.client.get = AsyncMock(side_effect=httpx.NetworkError("connection refused"))
    res_err = await client.get_recent_trades_volume("token_1", minutes=5)
    assert res_err.status == "UNAVAILABLE"
    assert res_err.volume is None


def test_db_models_volume_status_fields():
    """Verify MarketSnapshot and LiveMarket tables include volume_status column."""
    snap = MarketSnapshot(
        asset="BTC",
        market_id="m_test",
        time_left_min=10.0,
        mid_price=0.5,
        spread=0.01,
        volume_5min=123.4,
        volume_status="VALID",
        price_velocity=0.0,
        hour_of_day=12,
        final_outcome="PENDING",
        recorded_at=datetime.now(timezone.utc),
    )
    assert snap.volume_status == "VALID"
    assert snap.volume_5min == 123.4

    live = LiveMarket(
        market_id="m_test",
        asset="BTC",
        yes_token_id="y_1",
        no_token_id="n_1",
        end_time_est=datetime.now(timezone.utc),
        current_yes_price=0.5,
        current_no_price=0.5,
        current_spread=0.01,
        volume_5min=0.0,
        volume_status="VALID_ZERO",
        last_updated=datetime.now(timezone.utc),
    )
    assert live.volume_status == "VALID_ZERO"
    assert live.volume_5min == 0.0
