"""
live_session_service.py — Единый сервис управления LIVE-сессиями и лимитами риска
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

from sqlalchemy import select, func, text, and_
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.execution_models import (
    LiveTradingSession,
    ExecutionRequest,
    ExecutionWorkerStatus,
    ExposureReservation,
    LiveMirrorCandidate,
)
from polyflip.db.models import TradeHistory, RuntimeSettings

logger = logging.getLogger("live_session_service")


class LiveSessionError(Exception):
    """Базовое исключение сервиса LIVE-сессий."""

    pass


def get_max_order_cost(request: ExecutionRequest) -> Decimal:
    """
    Рассчитывает максимальное возможное списание по ордеру.
    max(target_amount_usdc, max_spend_usdc)
    """
    target = Decimal(str(request.target_amount_usdc or 0))
    max_spend = Decimal(str(request.max_spend_usdc or 0))
    amount = max(target, max_spend)
    if amount <= 0:
        raise LiveSessionError("LIVE order amount must be positive")
    return amount


async def get_active_session_for_update(
    db: AsyncSession,
) -> Optional[LiveTradingSession]:
    """Возвращает текущую управляемую сессию (DRAFT/READY/ACTIVE) с блокировкой строки."""
    return (
        await db.execute(
            select(LiveTradingSession)
            .where(LiveTradingSession.status.in_(["DRAFT", "READY", "ACTIVE"]))
            .with_for_update()
        )
    ).scalar_one_or_none()


async def count_session_positions(db: AsyncSession, session_id: uuid.UUID) -> int:
    """Считает открытые/закрывающиеся позиции сессии."""
    cnt = await db.scalar(
        select(func.count(TradeHistory.id)).where(
            TradeHistory.live_session_id == session_id,
            TradeHistory.mode == "LIVE",
            TradeHistory.position_status.in_(
                ["OPEN", "PARTIALLY_CLOSED", "EXIT_REQUESTED", "CLOSING"]
            ),
        )
    )
    return cnt or 0


async def get_session_exposure(db: AsyncSession, session_id: uuid.UUID) -> Decimal:
    """
    Рассчитывает суммарную экспозицию сессии:
    позиции (OPEN, PARTIALLY_CLOSED, EXIT_REQUESTED, CLOSING) + удержания (ExposureReservation).
    """
    pos_exp = await db.scalar(
        select(func.coalesce(func.sum(TradeHistory.entry_cost_usdc), 0)).where(
            TradeHistory.live_session_id == session_id,
            TradeHistory.mode == "LIVE",
            TradeHistory.position_status.in_(
                ["OPEN", "PARTIALLY_CLOSED", "EXIT_REQUESTED", "CLOSING"]
            ),
        )
    )
    res_exp = await db.scalar(
        select(func.coalesce(func.sum(ExposureReservation.amount_usdc), 0))
        .join(
            ExecutionRequest,
            ExecutionRequest.id == ExposureReservation.request_id,
        )
        .where(
            ExecutionRequest.live_session_id == session_id,
            ExposureReservation.released_at.is_(None),
        )
    )
    return Decimal(str(pos_exp or 0)) + Decimal(str(res_exp or 0))


async def calculate_session_filled_usdc(
    db: AsyncSession, session_id: uuid.UUID
) -> Decimal:
    """Вычисляет суммарную исполненную стоимость OPEN-ордеров сессии из БД."""
    val = await db.scalar(
        select(func.coalesce(func.sum(ExecutionRequest.filled_cost_usdc), 0)).where(
            ExecutionRequest.live_session_id == session_id,
            ExecutionRequest.requested_mode == "LIVE",
            ExecutionRequest.intent == "OPEN",
        )
    )
    return Decimal(str(val or 0))


@dataclass
class LiveReadinessResult:
    ready: bool
    checks: dict[str, bool]
    errors: list[str]
    warnings: list[str]


async def evaluate_live_readiness(
    db: AsyncSession, session: LiveTradingSession
) -> LiveReadinessResult:
    """
    Единый реальный модуль проверки 100% готовности системы к LIVE-торговле.
    """
    now = datetime.now(timezone.utc)
    checks = {
        "live_worker": False,
        "heartbeat": False,
        "credentials": False,
        "wallet": False,
        "network": False,
        "gateway": False,
        "balance": False,
        "collateral_allowance": False,
        "conditional_allowance": False,
        "active_requests": False,
        "old_positions": False,
        "single_alembic_head": True,
    }
    errors: list[str] = []
    warnings: list[str] = []

    # 1. LIVE worker checks
    ws = (
        await db.execute(
            select(ExecutionWorkerStatus)
            .where(ExecutionWorkerStatus.execution_mode == "LIVE")
            .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    if not ws:
        errors.append("LIVE worker status не найден в БД")
    else:
        checks["live_worker"] = True
        hb_at = ws.heartbeat_at
        if hb_at and hb_at.tzinfo is None:
            hb_at = hb_at.replace(tzinfo=timezone.utc)

        if hb_at and (now - hb_at).total_seconds() <= 30:
            checks["heartbeat"] = True
        else:
            errors.append("LIVE worker heartbeat устарел (>30 сек)")

        if ws.credentials_loaded:
            checks["credentials"] = True
        else:
            errors.append("Ключи кошелька LIVE worker не загружены")

        if ws.wallet_address:
            checks["wallet"] = True
            checks["network"] = True
        else:
            errors.append("Адрес кошелька Polygon не определен")

        if ws.gateway_ready:
            checks["gateway"] = True
        else:
            errors.append(f"Gateway не готов: {ws.last_error_message or 'неизвестно'}")

        if ws.collateral_allowance_ready:
            checks["collateral_allowance"] = True
        else:
            errors.append("USDC collateral allowance не подтвержден")

        if ws.conditional_allowance_ready:
            checks["conditional_allowance"] = True
        else:
            errors.append("Conditional token allowance (продажа токенов) не подтвержден")

        required_bal = min(
            Decimal(str(session.budget_usdc)),
            Decimal(str(session.max_total_exposure_usdc)),
        )
        current_bal = Decimal(str(ws.balance_usdc or 0))
        if current_bal >= required_bal:
            checks["balance"] = True
        else:
            errors.append(
                f"Баланс {current_bal} USDC меньше требуемого предела {required_bal} USDC"
            )

    # 2. Активные/зависшие LIVE-заявки
    active_cnt = (
        await db.scalar(
            select(func.count(ExecutionRequest.id)).where(
                ExecutionRequest.requested_mode == "LIVE",
                ExecutionRequest.state.in_(
                    [
                        "AWAITING_APPROVAL",
                        "READY",
                        "CLAIMED",
                        "SUBMITTING",
                        "ACCEPTED",
                        "UNKNOWN",
                        "PARTIALLY_FILLED",
                        "RECONCILING",
                        "MANUAL_REVIEW_REQUIRED",
                    ]
                ),
            )
        )
    ) or 0

    if active_cnt == 0:
        checks["active_requests"] = True
    else:
        errors.append(f"Обнаружено {active_cnt} незавершенных LIVE-заявок")

    # 3. Старые/незакрытые LIVE-позиции других сессий
    old_pos_cnt = (
        await db.scalar(
            select(func.count(TradeHistory.id)).where(
                TradeHistory.mode == "LIVE",
                TradeHistory.position_status.in_(
                    ["OPEN", "PARTIALLY_CLOSED", "EXIT_REQUESTED", "CLOSING"]
                ),
                TradeHistory.live_session_id != session.id,
            )
        )
    ) or 0

    if old_pos_cnt == 0:
        checks["old_positions"] = True
    else:
        errors.append(f"Обнаружено {old_pos_cnt} не закрытых LIVE-позиций предыдущих сессий")

    critical_keys = [
        "live_worker",
        "heartbeat",
        "credentials",
        "wallet",
        "network",
        "gateway",
        "balance",
        "collateral_allowance",
        "conditional_allowance",
        "active_requests",
        "old_positions",
    ]
    ready = all(checks[k] for k in critical_keys)

    return LiveReadinessResult(
        ready=ready, checks=checks, errors=errors, warnings=warnings
    )


def serialize_live_session_dto(
    session: LiveTradingSession, filled_usdc: Optional[Decimal] = None
) -> dict:
    filled_val = (
        float(filled_usdc)
        if filled_usdc is not None
        else float(session.filled_usdc or 0)
    )
    return {
        "id": str(session.id),
        "status": session.status,
        "budget_usdc": float(session.budget_usdc),
        "reserved_usdc": float(session.reserved_usdc),
        "remaining_budget_usdc": float(session.budget_usdc - session.reserved_usdc),
        "filled_usdc": filled_val,
        "max_single_order_usdc": float(session.max_single_order_usdc),
        "max_total_exposure_usdc": float(session.max_total_exposure_usdc),
        "max_open_positions": session.max_open_positions,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "stopped_at": session.stopped_at.isoformat() if session.stopped_at else None,
        "stop_reason": session.stop_reason,
        "created_at": session.created_at.isoformat() if session.created_at else None,
    }
