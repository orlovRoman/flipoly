import pytest

from polyflip.api.main import app
from polyflip.api.execution_api import get_live_trading_status


def test_trading_pnl_markers_route_matches_frontend_contract():
    paths = {route.path for route in app.routes}

    assert "/api/trading/pnl-markers" in paths
    assert "/pnl-markers" in paths


@pytest.mark.asyncio
async def test_execution_status_disables_live_switch_in_paper(
    db_session,
    monkeypatch,
):
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")

    result = await get_live_trading_status(db_session)

    assert result["execution_mode"] == "PAPER"
    assert result["kill_switch_available"] is False
