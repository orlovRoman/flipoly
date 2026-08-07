import asyncio
from decimal import Decimal
from typing import Any, Optional
import structlog
from polyflip.execution.contracts import GatewayOrder, SubmissionResult

logger = structlog.get_logger(__name__)

DEFAULT_GTC_TTL_SECONDS = 5.0
DEFAULT_FAK_RETRY_MAX_ATTEMPTS = 3
DEFAULT_FAK_RETRY_DELAY_SEC = 0.75


async def execute_gtc_ttl(
    gateway: Any,
    order: GatewayOrder,
    ttl_seconds: float = DEFAULT_GTC_TTL_SECONDS,
) -> SubmissionResult:
    """
    Отправляет GTD/GTC ордер в стакан, ждет ttl_seconds секунд.
    При истечении времени отменяет ордер и возвращает результат.
    """
    try:
        sub_res = await gateway.submit(order, order_type="GTC")
    except TypeError:
        sub_res = await gateway.submit(order)

    if not sub_res.accepted or not sub_res.provider_order_id:
        return sub_res

    provider_order_id = sub_res.provider_order_id
    token_id = order.token_id
    filled = False

    try:
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < ttl_seconds:
            await asyncio.sleep(0.5)
            if hasattr(gateway, "fetch_order_fills"):
                fills = await gateway.fetch_order_fills(provider_order_id, token_id)
                if fills:
                    filled = True
                    logger.info("gtc_ttl_filled_before_timeout", order_id=provider_order_id, fills_count=len(fills))
                    return sub_res
    except Exception as e:
        logger.warning("gtc_ttl_wait_error", order_id=provider_order_id, error=str(e))
    finally:
        if not filled:
            try:
                if hasattr(gateway, "cancel_order"):
                    await gateway.cancel_order(provider_order_id)
                    logger.info("gtc_ttl_cancelled_on_timeout", order_id=provider_order_id, ttl=ttl_seconds)
            except Exception as cancel_err:
                logger.warning("gtc_ttl_cancel_failed", order_id=provider_order_id, error=str(cancel_err))

    if hasattr(gateway, "fetch_order_fills"):
        try:
            final_fills = await gateway.fetch_order_fills(provider_order_id, token_id)
            if final_fills:
                return sub_res
        except Exception:
            pass

    return SubmissionResult(
        accepted=False,
        provider_order_id=provider_order_id,
        provider_status="NO_LIQUIDITY_TTL_EXPIRED",
        error_message=f"GTC order expired after {ttl_seconds}s without fill",
    )


async def execute_fak_retry(
    gateway: Any,
    order: GatewayOrder,
    api_client: Any = None,
    max_attempts: int = DEFAULT_FAK_RETRY_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_FAK_RETRY_DELAY_SEC,
) -> SubmissionResult:
    """
    Выполняет попытки FAK-ордера. Если NO_LIQUIDITY — выдерживает паузу,
    обновляет котировку из стакана и повторяет запрос до max_attempts раз.
    """
    last_result: Optional[SubmissionResult] = None
    current_order = order

    for attempt in range(1, max_attempts + 1):
        try:
            sub_res = await gateway.submit(current_order)
        except Exception as e:
            logger.warning("fak_retry_submit_exception", attempt=attempt, error=str(e))
            if attempt == max_attempts and last_result:
                return last_result
            raise e

        last_result = sub_res

        is_no_liquidity = (
            sub_res.provider_status == "NO_LIQUIDITY_FAK"
            or (sub_res.error_message and "NO_LIQUIDITY_FAK" in sub_res.error_message)
        )

        if sub_res.accepted and not is_no_liquidity:
            logger.info("fak_retry_success", attempt=attempt, order_id=str(current_order.attempt_id))
            return sub_res

        if is_no_liquidity or not sub_res.accepted:
            if attempt < max_attempts:
                logger.info(
                    "fak_retry_no_liquidity_waiting",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay=delay_seconds,
                    order_id=str(current_order.attempt_id),
                )
                await asyncio.sleep(delay_seconds)

                if api_client and hasattr(api_client, "get_market_prices"):
                    try:
                        prices = await asyncio.wait_for(
                            api_client.get_market_prices(current_order.token_id),
                            timeout=2.0,
                        )
                        if prices and prices.get("best_ask"):
                            new_price = Decimal(str(prices["best_ask"]))
                            if new_price > 0:
                                current_order = current_order.model_copy(
                                    update={"limit_price": new_price}
                                )
                                logger.info("fak_retry_price_refreshed", attempt=attempt, new_price=float(new_price))
                    except Exception as refresh_err:
                        logger.warning("fak_retry_price_refresh_failed", attempt=attempt, error=str(refresh_err))
                continue

    return last_result or SubmissionResult(
        accepted=False,
        provider_status="NO_LIQUIDITY_FAK",
        error_message=f"FAK retry exhausted after {max_attempts} attempts",
    )
