import httpx
from decimal import Decimal
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
import structlog
import json
from dataclasses import dataclass

from polyflip.db.models import TradeHistory, LiveMarket
from polyflip.constants import HTTP_TIMEOUT_SEC
from polyflip.execution.trade_lifecycle import mark_trade_resolved

logger = structlog.get_logger(__name__)


class LivePositionNotFound(Exception):
    pass


class MarketNotResolved(Exception):
    pass


class GammaApiError(Exception):
    pass


@dataclass
class ResolutionResult:
    final_outcome: str  # "YES" or "NO"
    resolution_source: str


async def fetch_polymarket_market(market_id: str) -> dict:
    """Получает данные рынка напрямую из Gamma API Polymarket."""
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(
                    "gamma_api_market_fetch_failed",
                    status=resp.status_code,
                    market_id=market_id,
                )
                raise GammaApiError(f"HTTP {resp.status_code}")
        except Exception as e:
            if not isinstance(e, GammaApiError):
                logger.error(
                    "gamma_api_market_fetch_error", error=str(e), market_id=market_id
                )
                raise GammaApiError(str(e))
            raise


def parse_confirmed_resolution(market: dict) -> Optional[ResolutionResult]:
    """Определяет результат по outcomes + outcomePrices,
    а не доверяет winning_outcome."""
    if not market:
        return None

    is_closed = market.get("closed", False)
    if not is_closed:
        return None

    outcomes = market.get("outcomes", [])
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            pass

    outcome_prices = market.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except Exception:
            pass

    if outcomes and outcome_prices and len(outcomes) == len(outcome_prices):
        try:
            for i, p in enumerate(outcome_prices):
                if float(p) >= 0.99:
                    winner = outcomes[i]
                    wo = str(winner).upper()
                    if wo in ("UP", "YES", "1"):
                        return ResolutionResult(
                            final_outcome="YES", resolution_source="GAMMA_API"
                        )
                    elif wo in ("DOWN", "NO", "0"):
                        return ResolutionResult(
                            final_outcome="NO", resolution_source="GAMMA_API"
                        )
                    else:
                        return ResolutionResult(
                            final_outcome="INVALID", resolution_source="GAMMA_API"
                        )
        except Exception:
            pass

    winning_outcome = market.get("winning_outcome")
    if winning_outcome:
        wo = str(winning_outcome).upper()
        if wo in ("UP", "YES", "1"):
            return ResolutionResult(final_outcome="YES", resolution_source="GAMMA_API")
        elif wo in ("DOWN", "NO", "0"):
            return ResolutionResult(final_outcome="NO", resolution_source="GAMMA_API")
        elif wo == "INVALID":
            return ResolutionResult(
                final_outcome="INVALID", resolution_source="GAMMA_API"
            )

    return None


async def save_market_resolution(
    db: AsyncSession, market_id: str, resolution: ResolutionResult
) -> None:
    market = await db.scalar(
        select(LiveMarket).where(LiveMarket.market_id == market_id).with_for_update()
    )
    if market:
        market.trading_status = "RESOLVED"
        market.resolution_status = (
            "RESOLVED" if resolution.final_outcome != "INVALID" else "INVALID"
        )
        market.final_outcome = resolution.final_outcome
        market.resolution_source = resolution.resolution_source
        market.resolved_at = datetime.now(timezone.utc)
        market.resolution_checked_at = datetime.now(timezone.utc)
        market.accepting_orders = False
        await db.flush()


async def refresh_market_trading_state(
    db: AsyncSession, market: LiveMarket, gamma_data: dict
) -> None:
    market.resolution_checked_at = datetime.now(timezone.utc)

    if gamma_data.get("closed"):
        market.trading_status = "RESOLVED"
        market.accepting_orders = False
    else:
        accepting = bool(gamma_data.get("acceptingOrders"))
        market.accepting_orders = accepting
        market.trading_status = "TRADABLE" if accepting else "CLOSED"

    await db.flush()


async def reconcile_live_resolution(db: AsyncSession, trade_id: int) -> TradeHistory:
    """Проверяет, завершился ли рынок, и если да — обновляет позицию
    до RESOLVED_REDEEMABLE или RESOLVED_LOST."""
    trade = await db.scalar(
        select(TradeHistory).where(TradeHistory.id == trade_id).with_for_update()
    )

    if not trade or trade.mode != "LIVE":
        raise LivePositionNotFound("Позиция не найдена или не является LIVE-позицией")

    if trade.position_status in (
        "RESOLVED_REDEEMABLE",
        "RESOLVED_LOST",
        "REDEEMED",
        "CLOSED",
    ):
        # Idempotency
        return trade

    market_data = await fetch_polymarket_market(trade.market_id)

    market = await db.scalar(
        select(LiveMarket)
        .where(LiveMarket.market_id == trade.market_id)
        .with_for_update()
    )
    if market:
        await refresh_market_trading_state(db, market, market_data)

    resolution = parse_confirmed_resolution(market_data)

    if resolution is None:
        raise MarketNotResolved("Рынок ещё не закрыт или результат неизвестен")

    if market:
        await save_market_resolution(db, trade.market_id, resolution)

    is_win = trade.outcome_bought == resolution.final_outcome
    trade.settlement_outcome = resolution.final_outcome
    mark_trade_resolved(trade, is_win=is_win)

    if is_win:
        payout = Decimal(str(trade.remaining_shares))
        trade.expected_payout_usdc = payout
        trade.redeemable_shares = trade.remaining_shares
        entry_basis = Decimal(str(trade.entry_cost_usdc or 0))
        trade.realized_pnl_usdc = payout - entry_basis
        trade.pnl = float(trade.realized_pnl_usdc)
    else:
        entry_basis = Decimal(str(trade.entry_cost_usdc or 0))
        trade.realized_pnl_usdc = -entry_basis
        trade.pnl = float(-entry_basis)

    return trade
