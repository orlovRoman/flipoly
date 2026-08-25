import asyncio
from decimal import Decimal
from typing import Any, Optional
import structlog
from polyflip.execution.contracts import GatewayOrder, SubmissionResult

logger = structlog.get_logger(__name__)

DEFAULT_GTC_TTL_SECONDS = 10.0
DEFAULT_FAK_RETRY_MAX_ATTEMPTS = 3
DEFAULT_FAK_RETRY_DELAY_SEC = 0.75

POST_ONLY_REJECT_MARKERS = (
    "post_only",
    "post only",
    "post-only",
    "crosses book",
    "cross the book",
    "would_take",
    "maker only",
)



def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _is_post_only_rejection(result: SubmissionResult) -> bool:
    status_text = " ".join(
        str(value or "") for value in (result.provider_status, result.error_message)
    ).lower()
    return any(marker in status_text for marker in POST_ONLY_REJECT_MARKERS)


def _is_cross_error(error: str) -> bool:
    normalized = str(error or "").lower()
    return any(
        marker in normalized
        for marker in (
            "crosses book", "cross the book", "would_take", "would take",
            "take liquidity", "immediately take", "post-only order",
        )
    )


def _round_maker_price(price: Decimal, side: str, tick_size: Decimal) -> Decimal:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    units = price / tick_size
    rounding = "ROUND_CEILING" if side.upper() == "SELL" else "ROUND_FLOOR"
    return units.to_integral_value(rounding=rounding) * tick_size


def calculate_maker_price(
    order: GatewayOrder,
    prices: dict[str, Any] | None,
    *,
    max_acceptable_price: Decimal | None = None,
    tick_size: Decimal = Decimal("0.01"),
) -> tuple[Decimal | None, Decimal | None, Decimal | None, str | None]:
    """Return a passive top-of-book price and quote telemetry."""
    if not prices:
        return None, None, None, "MAKER_NOT_POSTABLE"
    normalized_tick = _decimal(tick_size)
    if normalized_tick is None or normalized_tick <= 0:
        return None, None, None, "MAKER_NOT_POSTABLE"
    max_price = _decimal(max_acceptable_price)
    best_bid = _decimal(prices.get("best_bid"))
    best_ask = _decimal(prices.get("best_ask"))
    if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0:
        return None, best_bid, best_ask, "MAKER_NOT_POSTABLE"
    original = _decimal(order.limit_price)
    if original is None or original <= 0:
        return None, best_bid, best_ask, "MAKER_NOT_POSTABLE"
    side = order.side.upper()
    if side == "BUY":
        candidate = min(original, best_bid)
    elif side == "SELL":
        candidate = max(original, best_ask)
    else:
        return None, best_bid, best_ask, "MAKER_NOT_POSTABLE"
    maker_price = _round_maker_price(candidate, side, normalized_tick)
    if maker_price <= 0:
        return None, best_bid, best_ask, "MAKER_NOT_POSTABLE"
    if side == "BUY" and maker_price >= best_ask:
        return None, best_bid, best_ask, "MAKER_NOT_POSTABLE"
    if side == "SELL" and maker_price <= best_bid:
        return None, best_bid, best_ask, "MAKER_NOT_POSTABLE"
    if max_price is not None and side == "BUY" and maker_price > max_price:
        return None, best_bid, best_ask, "PRICE_MOVED"
    return maker_price, best_bid, best_ask, None


async def _fresh_prices(api_client: Any, token_id: str) -> dict[str, Any] | None:
    if api_client is None or not hasattr(api_client, "get_market_prices"):
        return None
    try:
        return await asyncio.wait_for(api_client.get_market_prices(token_id), timeout=2.0)
    except Exception as exc:
        logger.warning("maker_reprice_quote_failed", token_id=token_id, error=str(exc))
        return None


async def execute_maker_limit(
    gateway: Any,
    order: GatewayOrder,
    *,
    order_type: str = "GTC",
    api_client: Any = None,
    max_acceptable_price: Decimal | None = None,
    max_reprice_attempts: int = 1,
    tick_size: Decimal = Decimal("0.01"),
) -> SubmissionResult:
    """Submit a post-only GTC/GTD order and reprice once after a book cross."""
    attempts_allowed = 1 + max(0, min(int(max_reprice_attempts), 1))
    current_order = order.model_copy(update={"post_only": True})
    last_best_bid = None
    last_best_ask = None
    for attempt_no in range(1, attempts_allowed + 1):
        try:
            try:
                result = await gateway.submit(current_order, order_type=order_type)
            except TypeError:
                result = await gateway.submit(current_order)
        except Exception as exc:
            if _is_cross_error(str(exc)):
                result = SubmissionResult(
                    accepted=False, provider_status="POST_ONLY_REJECTED",
                    rejection_code="POST_ONLY_REJECTED", error_message=str(exc),
                )
            else:
                return SubmissionResult(
                    accepted=False, provider_status="NETWORK_ERROR",
                    error_message=str(exc), maker_attempts=attempt_no,
                )
        if result.accepted or not _is_post_only_rejection(result):
            return result.model_copy(update={
                "submitted_limit_price": current_order.limit_price,
                "submitted_requested_shares": current_order.requested_shares,
                "maker_attempts": attempt_no,
                "maker_status": ("MAKER_REPRICED" if result.accepted and attempt_no > 1 else ("RESTING" if result.accepted else None)),
                "maker_best_bid": last_best_bid,
                "maker_best_ask": last_best_ask,
            })
        cross_text = " ".join(str(value or "") for value in (result.provider_status, result.error_message))
        if not _is_cross_error(cross_text):
            error_message = str(result.error_message or "")
            if "POST_ONLY_REJECTED" not in error_message:
                error_message = f"POST_ONLY_REJECTED: {error_message}".rstrip()
            return result.model_copy(update={
                "provider_status": "POST_ONLY_REJECTED",
                "rejection_code": "POST_ONLY_REJECTED",
                "error_message": error_message,
                "maker_attempts": attempt_no,
            })
        if api_client is None:
            error_message = str(result.error_message or "")
            if "POST_ONLY_REJECTED" not in error_message:
                error_message = f"POST_ONLY_REJECTED: {error_message}".rstrip()
            return result.model_copy(update={"provider_status": "POST_ONLY_REJECTED", "rejection_code": "POST_ONLY_REJECTED", "error_message": error_message, "maker_attempts": attempt_no})
        if attempt_no >= attempts_allowed:
            error_message = str(result.error_message or "")
            if "MAKER_NOT_POSTABLE" not in error_message:
                error_message = f"MAKER_NOT_POSTABLE: {error_message}".rstrip()
            return result.model_copy(update={
                "provider_status": "MAKER_NOT_POSTABLE",
                "rejection_code": "MAKER_NOT_POSTABLE",
                "error_message": error_message,
                "maker_attempts": attempt_no, "maker_status": "MAKER_NOT_POSTABLE",
            })
        prices = await _fresh_prices(api_client, current_order.token_id)
        maker_price, best_bid, best_ask, failure = calculate_maker_price(
            current_order, prices, max_acceptable_price=max_acceptable_price, tick_size=tick_size,
        )
        last_best_bid = best_bid
        last_best_ask = best_ask
        if maker_price is None:
            status = failure or "MAKER_NOT_POSTABLE"
            return result.model_copy(update={
                "provider_status": status, "rejection_code": status, "error_message": status,
                "maker_attempts": attempt_no, "maker_status": status,
                "maker_best_bid": best_bid, "maker_best_ask": best_ask,
            })
        requested_shares = current_order.requested_shares
        if current_order.side.upper() == "BUY" and current_order.max_spend_usdc:
            requested_shares = current_order.max_spend_usdc / maker_price
        current_order = current_order.model_copy(update={
            "limit_price": maker_price, "requested_shares": requested_shares,
        })
        logger.info("maker_order_repriced", token_id=current_order.token_id,
                    attempt=attempt_no + 1, order_type=order_type,
                    best_bid=str(best_bid), best_ask=str(best_ask), maker_price=str(maker_price))
    return SubmissionResult(accepted=False, provider_status="MAKER_NOT_POSTABLE")

async def execute_gtc_ttl(
    gateway: Any,
    order: GatewayOrder,
    ttl_seconds: float = DEFAULT_GTC_TTL_SECONDS,
    post_only: bool = True,
    max_attempts: int = DEFAULT_FAK_RETRY_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_FAK_RETRY_DELAY_SEC,
    *,
    api_client: Any = None,
    max_acceptable_price: Decimal | None = None,
    max_reprice_attempts: int = 1,
    tick_size: Decimal = Decimal("0.01"),
) -> SubmissionResult:
    """Place a maker GTC order, wait for the TTL, then cancel if unfilled."""
    maker_order = order.model_copy(update={"post_only": post_only})
    if not post_only:
        return await gateway.submit(maker_order, order_type="GTC")
    sub_res = await execute_maker_limit(
        gateway, maker_order, order_type="GTC", api_client=api_client,
        max_acceptable_price=max_acceptable_price,
        max_reprice_attempts=max_reprice_attempts, tick_size=tick_size,
    )
    if not sub_res.accepted or not sub_res.provider_order_id:
        return sub_res
    provider_order_id = sub_res.provider_order_id
    token_id = maker_order.token_id
    filled = False
    matched_pending = False
    try:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        while (loop.time() - start_time) < ttl_seconds:
            await asyncio.sleep(0.5)
            if hasattr(gateway, "fetch_order_fills"):
                fills = await gateway.fetch_order_fills(provider_order_id, token_id)
                if fills:
                    filled = True
                    return sub_res.model_copy(update={"maker_status": "FILLED"})
    except Exception as exc:
        logger.warning("gtc_ttl_wait_error", order_id=provider_order_id, error=str(exc))
    finally:
        if not filled and hasattr(gateway, "cancel_order"):
            try:
                await gateway.cancel_order(provider_order_id)
            except Exception as cancel_err:
                logger.warning("gtc_ttl_cancel_failed", order_id=provider_order_id, error=str(cancel_err))

    if hasattr(gateway, "fetch_order_fills"):
        try:
            if await gateway.fetch_order_fills(provider_order_id, token_id):
                return sub_res.model_copy(update={"maker_status": "FILLED"})
        except Exception as exc:
            logger.warning(
                "gtc_ttl_final_fill_lookup_failed",
                order_id=provider_order_id,
                error=str(exc),
            )

    # Polymarket can report an order as MATCHED before its trade appears in
    # list_account_trades (and before settlement is CONFIRMED).  Treating that
    # short interval as a timeout loses a real fill: the worker then marks the
    # request REJECTED and never gets a chance to reconcile the provider order.
    # Keep the request in RECONCILING until the fill endpoint catches up.
    if hasattr(gateway, "get_order"):
        try:
            order_state = await gateway.get_order(provider_order_id)
            provider_status = str(order_state.provider_status or "").upper()
            matched_pending = provider_status in {
                "MATCHED",
                "PARTIALLY_MATCHED",
                "PARTIALLY_FILLED",
            }
        except Exception as exc:
            logger.warning(
                "gtc_ttl_final_order_lookup_failed",
                order_id=provider_order_id,
                error=str(exc),
            )

    if matched_pending:
        return sub_res.model_copy(
            update={
                "accepted": True,
                "provider_status": "MATCHED",
                "settlement_state": "PENDING",
                "maker_status": "MATCHED_PENDING_SETTLEMENT",
                "error_message": (
                    "GTC order matched; awaiting provider fill/settlement reconciliation"
                ),
            }
        )

    return sub_res.model_copy(update={
        "accepted": False, "provider_status": "NO_LIQUIDITY_TTL_EXPIRED",
        "maker_status": "TIMEOUT",
        "error_message": f"GTC order expired after {ttl_seconds}s without fill",
    })

async def execute_fak_retry(
    gateway: Any,
    order: GatewayOrder,
    api_client: Any = None,
    max_attempts: int = DEFAULT_FAK_RETRY_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_FAK_RETRY_DELAY_SEC,
    max_acceptable_price: Decimal | None = None,
) -> SubmissionResult:
    """
    Выполняет попытки FAK-ордера. Если NO_LIQUIDITY — выдерживает паузу,
    обновляет котировку из стакана и повторяет запрос до max_attempts раз.
    """
    last_result: Optional[SubmissionResult] = None
    current_order = order
    effective_max_price = _decimal(max_acceptable_price)
    if effective_max_price is None:
        effective_max_price = _decimal(order.max_acceptable_price)
    if (
        effective_max_price is not None
        and order.max_acceptable_price != effective_max_price
    ):
        current_order = order.model_copy(
            update={"max_acceptable_price": effective_max_price}
        )
    price_cap_blocked: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            sub_res = await gateway.submit(current_order)
        except Exception as e:
            logger.warning("fak_retry_submit_exception", attempt=attempt, error=str(e))
            if attempt == max_attempts:
                if last_result:
                    return last_result
                return SubmissionResult(
                    accepted=False,
                    provider_status="NETWORK_ERROR",
                    error_message=str(e),
                )
            await asyncio.sleep(delay_seconds)
            continue

        last_result = sub_res

        status_text = " ".join(
            str(value or "")
            for value in (sub_res.provider_status, sub_res.error_message)
        ).lower()
        is_no_liquidity = sub_res.provider_status == "NO_LIQUIDITY_FAK" or (
            sub_res.error_message and "NO_LIQUIDITY_FAK" in sub_res.error_message
        )
        is_transient = any(
            marker in status_text
            for marker in (
                "cloudflare",
                "502",
                "503",
                "504",
                "429",
                "timeout",
                "network",
                "temporarily unavailable",
                "connection reset",
            )
        )

        if sub_res.accepted and not is_no_liquidity:
            logger.info(
                "fak_retry_success",
                attempt=attempt,
                order_id=str(current_order.attempt_id),
            )
            return sub_res

        if is_no_liquidity or is_transient or not sub_res.accepted:
            if attempt < max_attempts:
                logger.info(
                    "fak_retry_no_liquidity_waiting",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay=delay_seconds,
                    order_id=str(current_order.attempt_id),
                )
                await asyncio.sleep(delay_seconds)
                prices = None
                quote_provider = getattr(gateway, "quote_provider", None)
                if callable(quote_provider):
                    try:
                        prices = await asyncio.wait_for(
                            quote_provider(current_order.token_id),
                            timeout=2.0,
                        )
                    except Exception as refresh_err:
                        logger.warning(
                            "fak_retry_quote_provider_failed",
                            attempt=attempt,
                            error=str(refresh_err),
                        )
                if (
                    not prices
                    and api_client
                    and hasattr(api_client, "get_market_prices")
                ):
                    try:
                        prices = await asyncio.wait_for(
                            api_client.get_market_prices(current_order.token_id),
                            timeout=2.0,
                        )
                    except Exception as refresh_err:
                        logger.warning(
                            "fak_retry_price_refresh_failed",
                            attempt=attempt,
                            error=str(refresh_err),
                        )
                if prices:
                    quote_key = (
                        "best_ask"
                        if current_order.side.upper() == "BUY"
                        else "best_bid"
                    )
                    new_price = _decimal(prices.get(quote_key))
                    cap = effective_max_price
                    if new_price is not None and new_price > 0:
                        if (
                            current_order.side.upper() == "BUY"
                            and cap is not None
                            and new_price > cap
                        ):
                            price_cap_blocked = (
                                f"Fresh ask {new_price} exceeds max acceptable "
                                f"price {cap}"
                            )
                            logger.info(
                                "fak_retry_price_above_cap",
                                attempt=attempt,
                                fresh_price=str(new_price),
                                max_acceptable_price=str(cap),
                            )
                        else:
                            updates: dict[str, Any] = {
                                "limit_price": new_price
                            }
                            if (
                                current_order.side.upper() == "BUY"
                                and current_order.max_spend_usdc is not None
                            ):
                                updates["requested_shares"] = (
                                    current_order.max_spend_usdc / new_price
                                )
                            current_order = current_order.model_copy(
                                update=updates
                            )
                            price_cap_blocked = None
                            logger.info(
                                "fak_retry_price_refreshed",
                                attempt=attempt,
                                new_price=str(new_price),
                                requested_shares=str(
                                    current_order.requested_shares
                                ),
                            )
                continue

    logger.warning(
        "fak_retry_exhausted",
        max_attempts=max_attempts,
        order_id=str(order.attempt_id),
        last_status=last_result.provider_status if last_result else "unknown",
        error=last_result.error_message if last_result else None,
    )
    if price_cap_blocked and last_result:
        return last_result.model_copy(
            update={
                "provider_status": "PRICE_MOVED",
                "rejection_code": "MAX_ACCEPTABLE_PRICE_EXCEEDED",
                "error_message": price_cap_blocked,
            }
        )
    return last_result or SubmissionResult(
        accepted=False,
        provider_status="NO_LIQUIDITY_FAK",
        error_message=f"FAK retry exhausted after {max_attempts} attempts",
    )
