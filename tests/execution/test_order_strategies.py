from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from polyflip.execution.contracts import GatewayOrder, SubmissionResult
from polyflip.execution.order_strategies import (
    FAKRetryEdgePolicy,
    calculate_maker_price,
    evaluate_fak_retry_buy_price,
    execute_fak_retry,
    execute_gtc_ttl,
    execute_maker_limit,
)
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


@pytest.mark.asyncio
async def test_gtc_ttl_keeps_matched_order_for_reconciliation_when_fill_lags():
    """A MATCHED order must not be converted into a false TTL rejection."""

    class _MatchedGateway:
        submit = AsyncMock(
            return_value=SubmissionResult(
                accepted=True,
                provider_order_id="provider-order-1",
                provider_status="OPEN",
            )
        )
        cancel_order = AsyncMock(return_value=True)
        fetch_order_fills = AsyncMock(return_value=())
        get_order = AsyncMock(
            return_value=SubmissionResult(
                accepted=True,
                provider_order_id="provider-order-1",
                provider_status="MATCHED",
                settlement_state="PENDING",
            )
        )

    gateway = _MatchedGateway()
    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-1",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="token-1",
        side="BUY",
        limit_price="0.19",
        requested_shares="5.78",
    )

    result = await execute_gtc_ttl(gateway, order, ttl_seconds=0.01)

    assert result.accepted is True
    assert result.provider_status == "MATCHED"
    assert result.settlement_state == "PENDING"
    assert result.maker_status == "MATCHED_PENDING_SETTLEMENT"
    gateway.get_order.assert_awaited_once_with("provider-order-1")


@pytest.mark.asyncio
async def test_fak_retry_refreshes_paper_ask_and_recalculates_shares():
    from polyflip.execution.gateways.fake import FakeExecutionGateway

    quotes = iter(
        [
            {
                "best_bid": "0.26",
                "best_ask": "0.30",
                "asks": [{"price": "0.30", "size": "20"}],
                "bids": [{"price": "0.26", "size": "20"}],
            },
            {
                "best_bid": "0.27",
                "best_ask": "0.28",
                "asks": [{"price": "0.28", "size": "20"}],
                "bids": [{"price": "0.27", "size": "20"}],
            },
            {
                "best_bid": "0.27",
                "best_ask": "0.28",
                "asks": [{"price": "0.28", "size": "20"}],
                "bids": [{"price": "0.27", "size": "20"}],
            },
        ]
    )

    async def quote_provider(token_id):
        return next(quotes)

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY",
        quote_provider=quote_provider,
        slippage_pct="0.5",
        fee_rate="0",
    )
    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-1",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="token-1",
        side="BUY",
        limit_price="0.27",
        requested_shares=Decimal("1") / Decimal("0.27"),
        max_spend_usdc="1",
        max_acceptable_price="0.284",
    )

    result = await execute_fak_retry(
        gateway,
        order,
        max_attempts=2,
        delay_seconds=0,
        max_acceptable_price=Decimal("0.284"),
    )

    assert result.accepted is True
    assert result.provider_status == "FILLED"
    assert result.submitted_limit_price == Decimal("0.28")
    assert result.submitted_requested_shares == Decimal("1") / Decimal("0.28")


@pytest.mark.asyncio
async def test_fak_retry_does_not_reprice_above_max_acceptable_price():
    from polyflip.execution.gateways.fake import FakeExecutionGateway

    async def quote_provider(token_id):
        return {
            "best_bid": "0.29",
            "best_ask": "0.30",
            "asks": [{"price": "0.30", "size": "20"}],
            "bids": [{"price": "0.29", "size": "20"}],
        }

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY",
        quote_provider=quote_provider,
        slippage_pct="0.5",
        fee_rate="0",
    )
    order = GatewayOrder(
        attempt_id=uuid4(),
        market_id="market-1",
        asset="BTC",
        outcome_to_buy="YES",
        token_id="token-1",
        side="BUY",
        limit_price="0.27",
        requested_shares="3.7",
        max_spend_usdc="1",
        max_acceptable_price="0.284",
    )

    result = await execute_fak_retry(
        gateway,
        order,
        max_attempts=2,
        delay_seconds=0,
        max_acceptable_price=Decimal("0.284"),
    )

    assert result.accepted is False
    assert result.provider_status == "PRICE_MOVED"
    assert result.rejection_code == "MAX_ACCEPTABLE_PRICE_EXCEEDED"


def _outsider_dynamic_edge_policy(
    probability: str = "0.40",
) -> FAKRetryEdgePolicy:
    return FAKRetryEdgePolicy(
        p_candidate_win=Decimal(probability),
        cost_buffer=Decimal("0.02"),
        min_net_edge=Decimal("0.04"),
        market_role="OUTSIDER",
        trade_min_price=Decimal("0.01"),
        trade_max_price=Decimal("0.90"),
        favorite_min_price=Decimal("0.55"),
        favorite_max_price=Decimal("0.90"),
        outsider_max_price=Decimal("0.40"),
    )


def test_dynamic_edge_accepts_fresh_price_above_stale_drift_cap():
    accepted, net_edge, reason = evaluate_fak_retry_buy_price(
        _outsider_dynamic_edge_policy(),
        Decimal("0.33"),
    )

    assert accepted is True
    assert net_edge == Decimal("0.05")
    assert reason is None


def test_dynamic_edge_rejects_fresh_price_when_edge_is_spent():
    accepted, net_edge, reason = evaluate_fak_retry_buy_price(
        _outsider_dynamic_edge_policy("0.38"),
        Decimal("0.33"),
    )

    assert accepted is False
    assert net_edge == Decimal("0.03")
    assert "Dynamic net edge" in (reason or "")


@pytest.mark.asyncio
async def test_fak_retry_reprices_above_stale_cap_when_dynamic_edge_survives():
    from polyflip.execution.gateways.fake import FakeExecutionGateway

    quotes = iter(
        [
            {
                "best_bid": "0.26",
                "best_ask": "0.30",
                "asks": [{"price": "0.30", "size": "20"}],
                "bids": [{"price": "0.26", "size": "20"}],
            },
            {
                "best_bid": "0.32",
                "best_ask": "0.33",
                "asks": [{"price": "0.33", "size": "20"}],
                "bids": [{"price": "0.32", "size": "20"}],
            },
            {
                "best_bid": "0.32",
                "best_ask": "0.33",
                "asks": [{"price": "0.33", "size": "20"}],
                "bids": [{"price": "0.32", "size": "20"}],
            },
        ]
    )

    async def quote_provider(token_id):
        return next(quotes)

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY",
        quote_provider=quote_provider,
        slippage_pct="0.5",
        fee_rate="0",
    )
    result = await execute_fak_retry(
        gateway,
        GatewayOrder(
            attempt_id=uuid4(),
            market_id="market-1",
            asset="BTC",
            outcome_to_buy="YES",
            token_id="token-1",
            side="BUY",
            limit_price="0.27",
            requested_shares=Decimal("1") / Decimal("0.27"),
            max_spend_usdc="1",
            max_acceptable_price="0.303",
        ),
        max_attempts=2,
        delay_seconds=0,
        max_acceptable_price=Decimal("0.303"),
        edge_policy=_outsider_dynamic_edge_policy(),
    )

    assert result.accepted is True
    assert result.submitted_limit_price == Decimal("0.33")
    assert result.submitted_requested_shares == Decimal("1") / Decimal("0.33")
    assert result.fak_retry_dynamic_edge_checked is True
    assert result.fak_retry_dynamic_net_edge == Decimal("0.05")
    assert result.fak_retry_dynamic_max_price == Decimal("0.40")


@pytest.mark.asyncio
async def test_fak_retry_rejects_refreshed_price_when_dynamic_edge_is_spent():
    class _NoLiquidityGateway:
        def __init__(self):
            self.submit = AsyncMock(
                return_value=SubmissionResult(
                    accepted=False,
                    provider_status="NO_LIQUIDITY_FAK",
                )
            )
            self.quote_provider = AsyncMock(
                return_value={"best_bid": "0.32", "best_ask": "0.33"}
            )

    gateway = _NoLiquidityGateway()
    result = await execute_fak_retry(
        gateway,
        GatewayOrder(
            attempt_id=uuid4(),
            market_id="market-1",
            asset="BTC",
            outcome_to_buy="YES",
            token_id="token-1",
            side="BUY",
            limit_price="0.27",
            requested_shares=Decimal("1") / Decimal("0.27"),
            max_spend_usdc="1",
            max_acceptable_price="0.303",
        ),
        max_attempts=2,
        delay_seconds=0,
        max_acceptable_price=Decimal("0.303"),
        edge_policy=_outsider_dynamic_edge_policy("0.38"),
    )

    assert result.accepted is False
    assert result.provider_status == "PRICE_MOVED"
    assert result.rejection_code == "DYNAMIC_EDGE_REJECTED"
    assert result.fak_retry_dynamic_edge_checked is True
    assert result.fak_retry_dynamic_net_edge == Decimal("0.03")
    assert "Dynamic net edge" in (result.error_message or "")
    assert gateway.submit.await_count == 1


@pytest.mark.asyncio
async def test_fake_fak_allows_equal_ask_with_modeled_slippage():
    from polyflip.execution.gateways.fake import FakeExecutionGateway

    async def quote_provider(token_id):
        return {
            "best_bid": "0.26",
            "best_ask": "0.27",
            "asks": [{"price": "0.27", "size": "20"}],
            "bids": [{"price": "0.26", "size": "20"}],
        }

    gateway = FakeExecutionGateway(
        profile="LIVE_PARITY",
        quote_provider=quote_provider,
        slippage_pct="0.5",
        fee_rate="0",
    )
    result = await gateway.submit(
        GatewayOrder(
            attempt_id=uuid4(),
            market_id="market-1",
            asset="BTC",
            outcome_to_buy="YES",
            token_id="token-1",
            side="BUY",
            limit_price="0.27",
            requested_shares="3.7",
            max_spend_usdc="1",
            max_acceptable_price="0.284",
        ),
        order_type="FAK",
    )

    assert result.accepted is True
    assert result.provider_status == "FILLED"
