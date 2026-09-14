import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

from polyflip.research.lp_rewards.universe import fetch_all_rewards_markets


@pytest.mark.asyncio
async def test_universe_pagination_until_terminal_marker():
    # Mock responses for 2 pages ending with LTE=
    page_1 = {
        "data": [{"condition_id": f"cond_{i}"} for i in range(500)],
        "next_cursor": "CURSOR_PAGE_2",
    }
    page_2 = {
        "data": [{"condition_id": f"cond_{i}"} for i in range(500, 750)],
        "next_cursor": "LTE=",
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # In httpx, Response.json() and raise_for_status() are synchronous methods
    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = page_1
    mock_resp1.raise_for_status.return_value = None

    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = page_2
    mock_resp2.raise_for_status.return_value = None

    mock_client.get.side_effect = [mock_resp1, mock_resp2]

    results = await fetch_all_rewards_markets(
        client=mock_client,
        terminal_marker="LTE=",
        page_size=500,
    )

    assert len(results) == 750
    assert mock_client.get.call_count == 2
    # Verify no duplicates
    condition_ids = [r["condition_id"] for r in results]
    assert len(set(condition_ids)) == 750
