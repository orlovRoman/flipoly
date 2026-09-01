import pytest

from polyflip.collector.client import PolymarketClient


@pytest.mark.asyncio
async def test_market_fee_schedule_reads_decimal_rate(monkeypatch):
    client = PolymarketClient.__new__(PolymarketClient)

    async def market_info(condition_id):
        return {
            "mos": 5,
            "fd": {"r": "0.07", "e": 2, "to": True},
        }

    monkeypatch.setattr(client, "get_clob_market_info", market_info)
    result = await client.get_market_fee_schedule("condition")

    assert result == {
        "fee_rate": 0.07,
        "maker_fee_rate": 0.0,
        "fees_enabled": True,
        "fee_exponent": 2.0,
        "taker_only": True,
        "min_order_shares": 5.0,
        "source": "CLOB_FD",
    }


@pytest.mark.asyncio
async def test_market_fee_schedule_converts_basis_points_payload():
    client = PolymarketClient.__new__(PolymarketClient)

    async def market_info(condition_id):
        return {"fee_schedule": {"feeRate": 700}}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(client, "get_clob_market_info", market_info)
    try:
        result = await client.get_market_fee_schedule("condition")
    finally:
        monkeypatch.undo()

    assert result["fee_rate"] == pytest.approx(0.07)


@pytest.mark.asyncio
async def test_market_fee_schedule_does_not_treat_missing_schedule_as_zero_fee():
    client = PolymarketClient.__new__(PolymarketClient)

    async def market_info(condition_id):
        return {"feesEnabled": True}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(client, "get_clob_market_info", market_info)
    try:
        result = await client.get_market_fee_schedule("condition")
    finally:
        monkeypatch.undo()

    assert result is None


@pytest.mark.asyncio
async def test_market_fee_schedule_parses_string_disabled_flag():
    client = PolymarketClient.__new__(PolymarketClient)

    async def market_info(condition_id):
        return {"feeSchedule": {"r": "0.07"}, "feesEnabled": "false"}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(client, "get_clob_market_info", market_info)
    try:
        result = await client.get_market_fee_schedule("condition")
    finally:
        monkeypatch.undo()

    assert result["fees_enabled"] is False
    assert result["fee_rate"] == 0.0
    assert result["fee_exponent"] == 1.0


@pytest.mark.asyncio
async def test_clob_market_info_uses_v2_endpoint():
    class Response:
        status_code = 200

        def json(self):
            return {"mos": 5, "fd": {"r": 0.07, "e": 1, "to": True}}

    class Client:
        def __init__(self):
            self.urls = []

        async def get(self, url):
            self.urls.append(url)
            return Response()

    client = PolymarketClient.__new__(PolymarketClient)
    http_client = Client()
    client.client = http_client
    client._market_info_cache = {}

    result = await client.get_clob_market_info("condition")

    assert result["mos"] == 5
    assert http_client.urls == [
        "https://clob.polymarket.com/clob-markets/condition"
    ]


@pytest.mark.asyncio
async def test_market_fee_schedule_treats_null_v2_fd_as_fee_free():
    client = PolymarketClient.__new__(PolymarketClient)

    async def market_info(condition_id):
        return {"mos": 5, "fd": None}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(client, "get_clob_market_info", market_info)
    try:
        result = await client.get_market_fee_schedule("condition")
    finally:
        monkeypatch.undo()

    assert result["fees_enabled"] is False
    assert result["fee_rate"] == 0.0
    assert result["min_order_shares"] == 5.0
