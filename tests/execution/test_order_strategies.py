from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from polyflip.execution.contracts import GatewayOrder, SubmissionResult
from polyflip.execution.order_strategies import execute_gtc_ttl, execute_maker_limit, calculate_maker_price
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


@pytest.mark.asyncio
async def test_maker_reprices_buy_to_best_bid_and_recalculates_shares():
    gateway = _PostOnlyRejectGateway()
    gateway.submit = AsyncMock(side_effect=[
        SubmissionResult(
            accepted=False,
            provider_status="POST_ONLY_REJECTED",
            error_message="invalid post-only order: order crosses book",
        ),
        SubmissionResult(
            accepted=True,
            provider_order_id="maker-2",
            provider_status="OPEN",
        ),
    ])

    class _Prices:
        async def get_market_prices(self, token_id):
            assert token_id == "token-1"
            return {"best_bid": "0.77", "best_ask": "0.80"}

    order = GatewayOrder(
        attempt_id=uuid4(), market_id="market-1", asset="BTC",
        outcome_to_buy="YES", token_id="token-1", side="BUY",
        limit_price="0.82", requested_shares="1.0", max_spend_usdc="1.10",
    )
    result = await execute_maker_limit(
        gateway, order, api_client=_Prices(), max_reprice_attempts=1,
    )

    assert result.accepted is True
    assert result.maker_attempts == 2
    assert result.maker_status == "MAKER_REPRICED"
    assert result.submitted_limit_price == Decimal("0.77")
    assert result.submitted_requested_shares == Decimal("1.10") / Decimal("0.77")
    assert gateway.submit.await_count == 2
    second_order = gateway.submit.await_args_list[1].args[0]
    assert second_order.post_only is True
    assert second_order.limit_price == Decimal("0.77")
    assert second_order.requested_shares == Decimal("1.10") / Decimal("0.77")


@pytest.mark.asyncio
async def test_maker_accepts_first_submission_without_quote():
    gateway = _PostOnlyRejectGateway()
    gateway.submit = AsyncMock(return_value=SubmissionResult(
        accepted=True, provider_order_id="maker-1", provider_status="OPEN",
    ))
    order = GatewayOrder(
        attempt_id=uuid4(), market_id="market-1", asset="BTC",
        outcome_to_buy="YES", token_id="token-1", side="BUY",
        limit_price="0.50", requested_shares="1.0",
    )
    result = await execute_maker_limit(gateway, order)
    assert result.accepted is True
    assert result.maker_status == "RESTING"
    assert result.maker_attempts == 1
    assert result.maker_best_bid is None
    assert result.maker_best_ask is None

def test_maker_price_uses_best_bid_for_buy_and_never_crosses_ask():
    order = GatewayOrder(
        attempt_id=uuid4(), market_id="market-1", asset="BTC",
        outcome_to_buy="YES", token_id="token-1", side="BUY",
        limit_price="0.82", requested_shares="1.0",
    )
    price, best_bid, best_ask, reason = calculate_maker_price(
        order, {"best_bid": "0.77", "best_ask": "0.80"},
    )
    assert price == Decimal("0.77")
    assert best_bid == Decimal("0.77")
    assert best_ask == Decimal("0.80")
    assert reason is None

def test_maker_price_normalizes_float_limits_and_tick():
    order = GatewayOrder(
        attempt_id=uuid4(), market_id="market-1", asset="BTC",
        outcome_to_buy="YES", token_id="token-1", side="BUY",
        limit_price="0.50", requested_shares="1.0",
    )
    price, _, _, reason = calculate_maker_price(
        order, {"best_bid": "0.49", "best_ask": "0.51"},
        max_acceptable_price=0.5, tick_size=0.01,
    )
    assert price == Decimal("0.49")
    assert reason is None

@pytest.mark.asyncio
async def test_maker_cross_retry_is_capped_at_one_attempt():
    gateway = _PostOnlyRejectGateway()
    gateway.submit = AsyncMock(return_value=SubmissionResult(
        accepted=False,
        provider_status="POST_ONLY_REJECTED",
        error_message="invalid post-only order: order crosses book",
    ))
    order = GatewayOrder(
        attempt_id=uuid4(), market_id="market-1", asset="BTC",
        outcome_to_buy="YES", token_id="token-1", side="BUY",
        limit_price="0.82", requested_shares="1.0", max_spend_usdc="1.10",
    )
    result = await execute_maker_limit(
        gateway, order,
        api_client=type("Prices", (), {"get_market_prices": AsyncMock(return_value={"best_bid": "0.77", "best_ask": "0.80"})})(),
        max_reprice_attempts=99,
    )
    assert result.accepted is False
    assert result.provider_status == "MAKER_NOT_POSTABLE"
    assert result.rejection_code == "MAKER_NOT_POSTABLE"
    assert result.maker_status == "MAKER_NOT_POSTABLE"
    assert gateway.submit.await_count == 2

def test_terminal_codes_keep_manual_review_separate_from_network_errors():
    assert _terminal_code("MANUAL_REVIEW_FAILED", None) == "MANUAL_REJECTED"
    assert _terminal_code("REJECTED", "POST_ONLY_REJECTED: would take") == "POST_ONLY_REJECTED"
