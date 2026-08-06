"""
live_session_service.py — Единый сервис управления LIVE-сессиями, бюджетом и лимитами риска
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import select, func, or_, case
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.execution_models import (
    LiveTradingSession,
    ExecutionRequest,
    ExecutionWorkerStatus,
    ExposureReservation,
)
from polyflip.db.models import TradeHistory

ACTIVE_TRADABLE_STATUSES = frozenset(["OPENING", "OPEN", "CLOSING", "RECONCILING"])


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
    позиции (OPEN, PARTIALLY_CLOSED, EXIT_REQUESTED, CLOSING)
    + невыпущенные удержания (ExposureReservation).
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
    diff_expr = ExposureReservation.amount_usdc - func.coalesce(
        ExecutionRequest.filled_cost_usdc, 0
    )
    unfilled_expr = case((diff_expr > 0, diff_expr), else_=0)

    res_exp = await db.scalar(
        select(func.coalesce(func.sum(unfilled_expr), 0))
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


@dataclass(frozen=True)
class SessionBudgetSnapshot:
    filled_usdc: Decimal
    reserved_usdc: Decimal
    committed_usdc: Decimal
    remaining_usdc: Decimal


async def get_session_budget_snapshot(
    db: AsyncSession,
    session_obj: LiveTradingSession,
) -> SessionBudgetSnapshot:
    """
    Рассчитывает динамический снимок бюджета сессии:
    filled = фактически исполненная стоимость OPEN-ордеров
    reserved = неисполненный остаток активных ExposureReservation
    committed = filled + reserved
    remaining = max(0, budget_usdc - committed)
    """
    diff_expr = ExposureReservation.amount_usdc - func.coalesce(
        ExecutionRequest.filled_cost_usdc, 0
    )
    unfilled_expr = case((diff_expr > 0, diff_expr), else_=0)

    filled_sq = (
        select(func.coalesce(func.sum(TradeHistory.entry_cost_usdc), 0))
        .where(
            TradeHistory.live_session_id == session_obj.id,
            TradeHistory.mode == "LIVE",
            TradeHistory.position_status.in_(ACTIVE_TRADABLE_STATUSES),
        )
        .scalar_subquery()
    )

    reserved_sq = (
        select(func.coalesce(func.sum(unfilled_expr), 0))
        .join(
            ExecutionRequest,
            ExecutionRequest.id == ExposureReservation.request_id,
        )
        .where(
            ExecutionRequest.live_session_id == session_obj.id,
            ExposureReservation.released_at.is_(None),
        )
        .scalar_subquery()
    )

    row = (await db.execute(select(filled_sq, reserved_sq))).one()
    filled = Decimal(str(row[0] or 0))
    reserved = Decimal(str(row[1] or 0))
    committed = filled + reserved
    remaining = max(
        Decimal("0"),
        Decimal(str(session_obj.budget_usdc)) - committed,
    )
    return SessionBudgetSnapshot(
        filled_usdc=filled,
        reserved_usdc=reserved,
        committed_usdc=committed,
        remaining_usdc=remaining,
    )


@dataclass
class LiveReadinessResult:
    ready: bool
    checks: dict[str, bool]
    errors: list[str]
    warnings: list[str]


async def get_latest_live_worker_status(
    db: AsyncSession,
) -> ExecutionWorkerStatus | None:
    return await db.scalar(
        select(ExecutionWorkerStatus)
        .where(ExecutionWorkerStatus.execution_mode == "LIVE")
        .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
        .limit(1)
    )


async def evaluate_live_readiness(
    db: AsyncSession,
    session: LiveTradingSession,
    *,
    worker_status: ExecutionWorkerStatus | None = None,
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
    }
    errors: list[str] = []
    warnings: list[str] = []

    # 1. LIVE worker checks
    # NOTE: transport_failure must be initialised before the `if not ws` branch
    # to prevent UnboundLocalError when no LIVE worker row exists in the DB.
    transport_failure: bool = False

    if worker_status is None:
        worker_status = await get_latest_live_worker_status(db)
    ws = worker_status

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
        else:
            errors.append("Адрес кошелька Polygon не определен")

        if ws.network_chain_id == 137:
            checks["network"] = True
        else:
            errors.append(
                f"Неверная сеть: chain_id={ws.network_chain_id}, ожидается Polygon Mainnet (137)"
            )

        if ws.gateway_ready:
            checks["gateway"] = True
        else:
            errors.append(f"Gateway не готов: {ws.last_error_message or 'неизвестно'}")
        required_bal = min(
            Decimal(str(session.budget_usdc)),
            Decimal(str(session.max_total_exposure_usdc)),
        )

        transport_failure = getattr(ws, "last_error_code", None) in {
            "READINESS_TIMEOUT",
            "TLS_TRANSPORT_ERROR",
            "NETWORK_TRANSPORT_ERROR",
        }

        if transport_failure:
            errors.append(
                f"Polymarket временно недоступен: " f"{ws.last_error_message}"
            )

            checks["balance"] = (
                ws.balance_usdc is not None
                and Decimal(str(ws.balance_usdc)) >= required_bal
            )
            checks["collateral_allowance"] = bool(ws.collateral_allowance_ready)
            checks["conditional_allowance"] = bool(ws.conditional_allowance_ready)

            warnings.append(
                "Баланс и approvals показаны по последней " "успешной проверке"
            )
        else:
            if getattr(ws, "balance_usdc", None) is None:
                errors.append("Баланс Polymarket пока не получен")
            else:
                current_bal = Decimal(str(ws.balance_usdc))
                if current_bal >= required_bal:
                    checks["balance"] = True
                else:
                    errors.append(
                        f"Баланс {current_bal} USDC меньше требуемого предела {required_bal} USDC"
                    )

            if getattr(ws, "collateral_allowance_ready", False):
                checks["collateral_allowance"] = True
            else:
                errors.append("USDC collateral allowance не подтвержден")

            if getattr(ws, "conditional_allowance_ready", False):
                checks["conditional_allowance"] = True
            else:
                errors.append(
                    "Conditional token allowance (продажа токенов) не подтвержден"
                )

    # 2. Активные/зависшие LIVE-заявки и 3. Старые/незакрытые LIVE-позиции
    active_requests_sq = (
        select(func.count(ExecutionRequest.id))
        .where(
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
        .scalar_subquery()
    )
    old_positions_sq = (
        select(func.count(TradeHistory.id))
        .where(
            TradeHistory.mode == "LIVE",
            TradeHistory.position_status.in_(
                ["OPEN", "PARTIALLY_CLOSED", "EXIT_REQUESTED", "CLOSING"]
            ),
            or_(
                TradeHistory.live_session_id.is_(None),
                TradeHistory.live_session_id != session.id,
            ),
        )
        .scalar_subquery()
    )

    counts_row = (await db.execute(select(active_requests_sq, old_positions_sq))).one()
    active_cnt, old_pos_cnt = counts_row[0] or 0, counts_row[1] or 0

    if active_cnt == 0:
        checks["active_requests"] = True
    else:
        errors.append(f"Обнаружено {active_cnt} незавершенных LIVE-заявок")

    if old_pos_cnt == 0:
        checks["old_positions"] = True
    else:
        errors.append(
            f"Обнаружено {old_pos_cnt} не закрытых LIVE-позиций "
            "прошлых сессий или без сессии"
        )

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
    # transport_failure already forces checks["gateway"]=False inside the else-branch;
    # no need to repeat here. ready requires all critical checks to pass.
    ready = not transport_failure and all(checks[key] for key in critical_keys)

    return LiveReadinessResult(
        ready=ready, checks=checks, errors=errors, warnings=warnings
    )


def serialize_live_session_dto(
    session: LiveTradingSession, budget_snapshot: Optional[SessionBudgetSnapshot] = None
) -> dict:
    if budget_snapshot is not None:
        filled_val = float(budget_snapshot.filled_usdc)
        reserved_val = float(budget_snapshot.reserved_usdc)
        committed_val = float(budget_snapshot.committed_usdc)
        remaining_val = float(budget_snapshot.remaining_usdc)
    else:
        filled_val = float(session.filled_usdc or 0)
        reserved_val = float(session.reserved_usdc or 0)
        committed_val = reserved_val + filled_val
        remaining_val = max(0.0, float(session.budget_usdc - session.reserved_usdc))

    return {
        "id": str(session.id),
        "status": session.status,
        "budget_usdc": float(session.budget_usdc),
        "reserved_usdc": reserved_val,
        "committed_usdc": committed_val,
        "remaining_budget_usdc": remaining_val,
        "filled_usdc": filled_val,
        "max_single_order_usdc": float(session.max_single_order_usdc),
        "order_amount_usdc": (
            float(session.order_amount_usdc)
            if getattr(session, "order_amount_usdc", None) is not None
            else None
        ),
        "max_total_exposure_usdc": float(session.max_total_exposure_usdc),
        "max_open_positions": session.max_open_positions,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "stopped_at": session.stopped_at.isoformat() if session.stopped_at else None,
        "stop_reason": session.stop_reason,
        "created_at": session.created_at.isoformat() if session.created_at else None,
    }
