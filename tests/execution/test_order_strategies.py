from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from polyflip.execution.contracts import GatewayOrder, SubmissionResult
from polyflip.execution.order_strategies import execute_gtc_ttl
from polyflip.execution.outbox import _terminal_code


class _PostOnlyRejectGateway:
    def __init__(self):
        self.submit = AsyncMock(
            return_value=SubmissionResult(
                accepted=False,
                provider_status="POST_ONLY_REJECT",
                error_message="post_only order would take liquidity",
            )
        )


@pytest.mark.asyncio
async def test_gtc_ttl_classifies_post_only_rejection_without_fallback():
    gateway = _PostOnlyRejectGateway()
    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-1",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="token-1",
        side="BUY",
        limit_price="0.50",
        requested_shares="2",
    )

    result = await execute_gtc_ttl(gateway, order, ttl_seconds=0.01)

    assert result.accepted is False
    assert result.provider_status == "POST_ONLY_REJECTED"
    assert result.rejection_code == "POST_ONLY_REJECTED"
    assert "POST_ONLY_REJECTED" in (result.error_message or "")
    gateway.submit.assert_awaited_once()
    assert gateway.submit.await_args.args[0].post_only is True


def test_terminal_codes_keep_manual_review_separate_from_network_errors():
    assert _terminal_code("MANUAL_REVIEW_FAILED", None) == "MANUAL_REJECTED"
    assert _terminal_code("REJECTED", "POST_ONLY_REJECTED: would take") == "POST_ONLY_REJECTED"