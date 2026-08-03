"""
release_gate.py — Этап 8: шлюз выпуска LiveMirrorCandidate → LIVE TradeHistory
                    + LIVE ExecutionRequest

Архитектура:
    mirror_worker                 release_gate              execution_worker_live
    ──────────────────────        ──────────────────────    ──────────────────────
    PAPER OPEN FILLED ──►         candidate(NEW/ELIGIBLE)   picks up LIVE OPEN READY
    LiveMirrorCandidate(NEW)      ──► atomically creates   ──► sends order to exchange
                                  TradeHistory(LIVE)
                                  ExecutionRequest(LIVE,READY)
                                  candidate.state=RELEASED

Инварианты (принудительные):
    - Никогда не изменяет PAPER TradeHistory или PAPER ExecutionRequest.
    - Атомарная транзакция: либо всё создаётся, либо ничего.
    - Кандидат переходит в RELEASED только внутри той же транзакции.
    - Если release_gate падает после INSERT TradeHistory, но до RELEASED —
      при следующем запуске он не увидит кандидата как RELEASED
      и попытается снова. Уникальный индекс uq_live_trade_source_paper
      заблокирует дубль INSERT → кандидат остаётся ELIGIBLE, оператор
      получает ошибку в логах.

Управление (читается из RuntimeSettings в БД, не из env):
    LIVE_RELEASE_MODE:
        DISABLED  — не делаем ничего (спим)
        MANUAL    — обрабатываем только кандидатов в состоянии ELIGIBLE
                    (оператор выставил через /api/execution/candidates/{id}/release)
        AUTO      — обрабатываем кандидатов в состоянии NEW и ELIGIBLE
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from polyflip.db.execution_models import (
    ExecutionRequest,
    LiveMirrorCandidate,
    ExecutionWorkerStatus,
    LiveTradingSession,
)
from polyflip.db.models import RuntimeSettings, TradeHistory
from polyflip.execution.live_mirror_worker import (
    PAPER_MIRRORABLE_STATES,
    _build_signal_snapshot,
    _compute_hash,
)
from polyflip.execution.risk_checks import check_risk_limits
from polyflip.execution.config import LIVE_MIN_GROSS_BUY_USDC

logger = logging.getLogger("release_gate")


@dataclass(frozen=True)
class LiveReleasePlan:
    order_amount_usdc: Decimal
    max_spend_usdc: Decimal
    session: LiveTradingSession | None


def calculate_live_order_amount(
    paper_request: ExecutionRequest,
    session: LiveTradingSession,
) -> Decimal:
    from polyflip.execution.live_session_service import get_max_order_cost

    if session.order_amount_usdc is not None:
        live_amount = Decimal(str(session.order_amount_usdc))
    else:
        # Fallback (old behavior) for old sessions where order_amount_usdc is NULL
        source_amount = get_max_order_cost(paper_request)
        live_amount = max(
            source_amount,
            LIVE_MIN_GROSS_BUY_USDC,
        )

    if live_amount > Decimal(str(session.max_single_order_usdc)):
        raise ReleaseDeferred(
            f"LIVE-сумма {live_amount} USDC превышает лимит "
            f"сессии {session.max_single_order_usdc} USDC"
        )

    return live_amount


# ── Настройки ────────────────────────────────────────────────────────────────

DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
POLL_INTERVAL: float = float(os.environ.get("RELEASE_GATE_POLL_INTERVAL", "5"))

_shutdown = False


class ReleaseRejected(Exception):
    """Кандидат забракован окончательно — переходит в REJECTED."""

    pass


class ReleaseDeferred(Exception):
    """Временная неготовность системы — кандидат остаётся ELIGIBLE/NEW."""

    pass


def _handle_sigterm(signum: int, frame: object) -> None:
    global _shutdown
    logger.info("Получен сигнал %s, остановка release_gate...", signum)
    _shutdown = True


# ── Вспомогательные функции ───────────────────────────────────────────────────


async def _get_release_mode(session: AsyncSession) -> str:
    """Читает LIVE_RELEASE_MODE из RuntimeSettings (DISABLED | MANUAL | AUTO)."""
    row = (
        await session.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == "LIVE_RELEASE_MODE")
        )
    ).scalar_one_or_none()
    return (row.value if row else "DISABLED").upper()


async def get_candidate_ids(
    session: AsyncSession,
    release_mode: str,
    target_mode: str,
    batch_size: int = 10,
) -> list[uuid.UUID]:
    """
    Возвращает ID кандидатов, готовых к выпуску, в зависимости от режима:
      AUTO   — NEW и ELIGIBLE
      MANUAL — только ELIGIBLE
      DISABLED — []
    """
    if release_mode == "DISABLED":
        return []

    eligible_states = ["ELIGIBLE"] if release_mode == "MANUAL" else ["NEW", "ELIGIBLE"]

    stmt = (
        select(LiveMirrorCandidate.id)
        .where(
            LiveMirrorCandidate.state.in_(eligible_states),
            LiveMirrorCandidate.target_mode == target_mode,
        )
        .order_by(LiveMirrorCandidate.created_at)
        .limit(batch_size)
    )
    return list((await session.scalars(stmt)).all())


async def release_batch(
    session: AsyncSession,
    target_mode: str,
) -> int:
    """
    Выпускает очередную порцию кандидатов.
    Возвращает количество успешно выпущенных.
    """
    release_mode = await _get_release_mode(session)
    if release_mode == "DISABLED":
        return 0

    candidate_ids = await get_candidate_ids(session, release_mode, target_mode)
    if not candidate_ids:
        return 0

    released = 0

    for candidate_id in candidate_ids:
        try:
            success = await release_candidate_by_id(session, candidate_id, target_mode)
            if success:
                released += 1
            else:
                await session.rollback()
        except Exception as exc:
            await session.rollback()
            logger.error(
                "release_failed for candidate %s: %s",
                candidate_id,
                exc,
                exc_info=True,
            )

    if released:
        logger.info(
            "release_gate: выпущено %d кандидатов (mode=%s, target=%s)",
            released,
            release_mode,
            target_mode,
        )
    return released


async def release_candidate_by_id(
    session: AsyncSession,
    candidate_id: uuid.UUID,
    target_mode: str,
) -> bool:
    """
    Блокирует кандидата с помощью advisory_xact_lock и FOR UPDATE, проверяет состояние и выпускает.
    """
    from sqlalchemy import text

    conn = await session.connection()
    if conn.dialect.name == "postgresql":
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
            {"key": f"release:{target_mode}"},
        )

    candidate = await session.scalar(
        select(LiveMirrorCandidate)
        .where(LiveMirrorCandidate.id == candidate_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )

    if candidate is None or candidate.state not in ("NEW", "ELIGIBLE"):
        return False

    return await _release_one_locked(session, candidate, target_mode)


async def _release_one_locked(
    session: AsyncSession,
    candidate: LiveMirrorCandidate,
    target_mode: str,
) -> bool:
    """
    Валидирует и атомарно выпускает заблокированного кандидата.
    Возвращает True если кандидат выпущен, False если задефферен или отклонен.
    """
    from polyflip.db.execution_models import ExposureReservation

    now = datetime.now(timezone.utc)

    # 1. Загружаем PAPER-источники
    paper_request = (
        await session.execute(
            select(ExecutionRequest).where(
                ExecutionRequest.id == candidate.source_paper_request_id
            )
        )
    ).scalar_one_or_none()

    if paper_request is None:
        candidate.state = "REJECTED"
        candidate.rejection_reason = "source_paper_request not found"
        await session.commit()
        logger.warning(
            "Rejected candidate %s: source_paper_request not found", candidate.id
        )
        return False

    paper_trade = (
        await session.execute(
            select(TradeHistory).where(
                TradeHistory.id == candidate.source_paper_trade_id
            )
        )
    ).scalar_one_or_none()

    if paper_trade is None:
        candidate.state = "REJECTED"
        candidate.rejection_reason = "source_paper_trade not found"
        await session.commit()
        logger.warning(
            "Rejected candidate %s: source_paper_trade not found", candidate.id
        )
        return False

    # 2. Валидация выпуска
    try:
        release_plan = await validate_live_release(
            session, candidate, paper_request, paper_trade, target_mode
        )
    except ReleaseDeferred as e:
        logger.info("Deferred release for candidate %s: %s", candidate.id, e)
        return False
    except ReleaseRejected as e:
        logger.warning("Rejected candidate %s: %s", candidate.id, e)
        candidate.state = "REJECTED"
        candidate.rejection_reason = str(e)
        await session.commit()
        return False

    # 3. Достаем активную LIVE-сессию для связки (если target_mode == LIVE)
    active_session = (
        release_plan.session if isinstance(release_plan, LiveReleasePlan) else None
    )

    # 4. Создаём LIVE-строки
    live_trade = _build_live_trade(
        candidate,
        paper_trade,
        now,
        target_mode,
        order_amount_usdc=(
            release_plan.order_amount_usdc
            if isinstance(release_plan, LiveReleasePlan)
            else release_plan
        ),
    )
    if active_session:
        live_trade.live_session_id = active_session.id
    session.add(live_trade)
    await session.flush()

    live_request = _build_live_request(
        candidate,
        paper_request,
        live_trade,
        now,
        target_mode,
        order_amount_usdc=(
            release_plan.order_amount_usdc
            if isinstance(release_plan, LiveReleasePlan)
            else release_plan
        ),
        max_spend_usdc=(
            release_plan.max_spend_usdc
            if isinstance(release_plan, LiveReleasePlan)
            else release_plan
        ),
    )
    if active_session:
        live_request.live_session_id = active_session.id
    session.add(live_request)
    await session.flush()

    # 5. Резервируем экспозицию (ExposureReservation) и бюджет сессии
    exposure_amount = (
        release_plan.max_spend_usdc
        if isinstance(release_plan, LiveReleasePlan)
        else release_plan
    )
    exposure_res = ExposureReservation(
        id=uuid.uuid4(),
        request_id=live_request.id,
        trade_history_id=live_trade.id,
        market_id=live_request.market_id,
        amount_usdc=exposure_amount,
        expires_at=live_request.expires_at or (now + timedelta(seconds=30)),
        created_at=now,
    )
    session.add(exposure_res)

    # Примечание: session.reserved_usdc не накапливается накопительно.
    # Источник истины — динамический get_session_budget_snapshot().

    # 6. Обновляем кандидата
    candidate.state = "RELEASED"
    candidate.released_at = now
    candidate.released_trade_id = live_trade.id
    candidate.released_request_id = live_request.id

    await session.commit()

    logger.info(
        "release_gate: candidate=%s → live_trade=%d live_request=%s exposure_res=%s",
        candidate.id,
        live_trade.id,
        live_request.id,
        exposure_res.id,
    )
    return True


async def validate_live_release(
    session: AsyncSession,
    candidate: LiveMirrorCandidate,
    paper_request: ExecutionRequest,
    paper_trade: TradeHistory,
    target_mode: str,
) -> LiveReleasePlan | Decimal:
    """
    Проверяет, можно ли сейчас выпустить кандидата в LIVE-исполнение (или SHADOW).
    Возвращает live_amount (Decimal), который нужно использовать для создания заявок.
    При невосстановимой ошибке бросает ReleaseRejected (кандидат отклоняется).
    При временной неготовности системы бросает ReleaseDeferred (откладывается).
    """
    now = datetime.now(timezone.utc)

    # 1. Проверка источника
    if candidate.target_mode != target_mode:
        raise ReleaseRejected("candidate target_mode mismatch")

    if paper_request.requested_mode != "PAPER":
        raise ReleaseRejected("source request is not PAPER")

    if paper_request.intent != "OPEN":
        raise ReleaseRejected("source request is not OPEN")

    if paper_request.state not in PAPER_MIRRORABLE_STATES:
        raise ReleaseRejected(
            f"source request state is {paper_request.state!r}, not finally filled"
        )

    if paper_trade.mode != "PAPER":
        raise ReleaseRejected("source trade is not PAPER")

    if paper_request.trade_history_id != paper_trade.id:
        raise ReleaseRejected("source linkage mismatch")

    # 2. Совпадение snapshot signal_hash
    current_snapshot = _build_signal_snapshot(paper_request, paper_trade)
    if _compute_hash(current_snapshot) != candidate.signal_hash:
        raise ReleaseRejected("signal snapshot hash mismatch")

    # 3. Возраст сигнала: не старше 30 секунд от updated_at/created_at
    created_or_updated = paper_request.updated_at or paper_request.created_at
    if created_or_updated:
        if created_or_updated.tzinfo is None:
            created_or_updated = created_or_updated.replace(tzinfo=timezone.utc)
        age_sec = (now - created_or_updated).total_seconds()
        if age_sec > 30:
            raise ReleaseRejected(f"Signal is too old ({age_sec:.1f}s > 30s)")

    # 4. Проверка окончания рынка
    if paper_trade.market_end_time:
        end_time = paper_trade.market_end_time
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        if end_time <= now or (end_time - now).total_seconds() < 120:
            raise ReleaseRejected("Market is closed or ending soon (< 120s)")

    # 4.1 Проверка минимального размера ордера
    order_amount = Decimal(str(paper_request.target_amount_usdc or 0))
    if order_amount < Decimal("1.00"):
        raise ReleaseRejected(
            f"Сумма ордера {order_amount} USDC ниже минимальной суммы Polymarket 1.00 USDC"
        )

    # 5. Проверки для LIVE режима (kill-switch, worker, gateway, allowance, balance, session)
    if target_mode == "LIVE":
        live_enabled = await session.scalar(
            select(RuntimeSettings.value).where(
                RuntimeSettings.key == "LIVE_TRADING_ENABLED"
            )
        )
        if live_enabled is None or live_enabled.strip().lower() != "true":
            raise ReleaseDeferred("LIVE kill switch is off")

        # 5.1 Проверка активной LIVE-сессии и лимитов сессии
        from polyflip.db.execution_models import LiveTradingSession
        from polyflip.execution.live_session_service import (
            count_session_positions,
            get_session_exposure,
        )

        active_session = (
            await session.execute(
                select(LiveTradingSession)
                .where(LiveTradingSession.status == "ACTIVE")
                .with_for_update()
            )
        ).scalar_one_or_none()

        if active_session is None:
            raise ReleaseDeferred("No active LIVE trading session")

        # 5.2 Проверка стоимости ордера max(target_amount_usdc, max_spend_usdc)
        live_amount = calculate_live_order_amount(paper_request, active_session)
        order_amount = live_amount

        # Лимит бюджета сессии через SessionBudgetSnapshot
        from polyflip.execution.live_session_service import get_session_budget_snapshot

        budget_snap = await get_session_budget_snapshot(session, active_session)
        if order_amount > budget_snap.remaining_usdc:
            raise ReleaseDeferred(
                "LIVE session budget exhausted "
                f"(remaining {budget_snap.remaining_usdc} USDC "
                f"< {order_amount} USDC)"
            )

        # Лимит количества позиций сессии
        open_positions = await count_session_positions(session, active_session.id)
        if open_positions >= active_session.max_open_positions:
            raise ReleaseDeferred(
                "Session open-position limit reached "
                f"({open_positions} >= "
                f"{active_session.max_open_positions})"
            )

        # Лимит экспозиции сессии
        current_exposure = await get_session_exposure(session, active_session.id)
        max_exposure = Decimal(str(active_session.max_total_exposure_usdc))
        if current_exposure + order_amount > max_exposure:
            raise ReleaseDeferred(
                "Session exposure limit reached "
                f"({current_exposure} + {order_amount} "
                f"> {max_exposure})"
            )

        ws = (
            await session.execute(
                select(ExecutionWorkerStatus)
                .where(ExecutionWorkerStatus.execution_mode == "LIVE")
                .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if ws is None:
            raise ReleaseDeferred("No LIVE worker status found")

        hb_at = ws.heartbeat_at
        if hb_at and hb_at.tzinfo is None:
            hb_at = hb_at.replace(tzinfo=timezone.utc)
        if not hb_at or (now - hb_at).total_seconds() > 60:
            raise ReleaseDeferred("LIVE worker heartbeat is stale (>60s)")

        if not ws.gateway_ready:
            raise ReleaseDeferred(
                f"LIVE worker gateway not ready: {ws.last_error_message}"
            )

        if not ws.collateral_allowance_ready:
            raise ReleaseDeferred("LIVE worker collateral allowance not ready")

        if not ws.conditional_allowance_ready:
            raise ReleaseDeferred("LIVE worker conditional token allowance not ready")

        required_balance = min(
            Decimal(str(active_session.budget_usdc)),
            Decimal(str(active_session.max_total_exposure_usdc)),
        )
        if Decimal(str(ws.balance_usdc or 0)) < required_balance:
            raise ReleaseDeferred(
                f"LIVE worker balance USDC ({ws.balance_usdc}) "
                f"is less than required ({required_balance})"
            )

    # 6. Риск-лимиты
    risk_error = await check_risk_limits(
        session,
        intent="OPEN",
        max_spend_usdc=order_amount,
        requested_mode=target_mode,
    )
    if risk_error:
        raise ReleaseRejected(f"Risk check failed: {risk_error}")

    if target_mode == "LIVE":
        return LiveReleasePlan(
            order_amount_usdc=order_amount,
            max_spend_usdc=order_amount,
            session=active_session,
        )

    return order_amount


def _build_live_trade(
    candidate: LiveMirrorCandidate,
    paper_trade: TradeHistory,
    now: datetime,
    target_mode: str,
    order_amount_usdc: Decimal,
) -> TradeHistory:
    """
    Создаёт новый TradeHistory(mode=target_mode) на основе PAPER-снимка.
    """
    return TradeHistory(
        market_id=paper_trade.market_id,
        asset=paper_trade.asset,
        outcome_bought=paper_trade.outcome_bought,
        amount_usdc=order_amount_usdc,
        executed_price=0.0,  # заполняется средней ценой после fill
        predicted_flip_prob=paper_trade.predicted_flip_prob,
        active_features=paper_trade.active_features,
        model_version=paper_trade.model_version,
        status="PENDING",
        mode=target_mode,
        edge=paper_trade.edge,
        market_role=paper_trade.market_role,
        strategy_type=paper_trade.strategy_type,
        p_flip_effective=paper_trade.p_flip_effective,
        p_win_effective=paper_trade.p_win_effective,
        stop_loss_pct=paper_trade.stop_loss_pct,
        stop_loss_price=paper_trade.stop_loss_price,
        take_profit_enabled=paper_trade.take_profit_enabled,
        take_profit_multiplier=paper_trade.take_profit_multiplier,
        take_profit_price=paper_trade.take_profit_price,
        position_status="OPENING",
        entry_filled_shares=Decimal("0"),
        entry_cost_usdc=Decimal("0"),
        remaining_shares=Decimal("0"),
        position_accounting_version=0,  # инициализируется после FILLED
        model_key=paper_trade.model_key,
        confirm_model_key=paper_trade.confirm_model_key,
        confirm_model_version=paper_trade.confirm_model_version,
        model_attribution_source=paper_trade.model_attribution_source,
        direction_model_key=paper_trade.direction_model_key,
        direction_model_version=paper_trade.direction_model_version,
        entry_model_key=paper_trade.entry_model_key,
        entry_model_version=paper_trade.entry_model_version,
        entry_model_source=paper_trade.entry_model_source,
        p_candidate_win=paper_trade.p_candidate_win,
        gross_edge=paper_trade.gross_edge,
        cost_buffer=paper_trade.cost_buffer,
        net_edge=paper_trade.net_edge,
        decision_run_id=paper_trade.decision_run_id,
        config_snapshot=paper_trade.config_snapshot,
        market_end_time=paper_trade.market_end_time,
        source_paper_trade_id=paper_trade.id,
        created_at=now,
        updated_at=now,
    )


def _build_live_request(
    candidate: LiveMirrorCandidate,
    paper_request: ExecutionRequest,
    live_trade: TradeHistory,
    now: datetime,
    target_mode: str,
    order_amount_usdc: Decimal,
    max_spend_usdc: Decimal,
) -> ExecutionRequest:
    """
    Создаёт новый ExecutionRequest(requested_mode=target_mode, state='READY').
    Заполняет ttl_seconds (макс 30с) и expires_at.
    """
    ttl_seconds = min(paper_request.ttl_seconds or 30, 30)
    expires_at = now + timedelta(seconds=ttl_seconds)

    return ExecutionRequest(
        id=uuid.uuid4(),
        idempotency_key=f"{target_mode}-OPEN-mirror-{candidate.id}",
        requested_mode=target_mode,
        trade_history_id=live_trade.id,
        intent="OPEN",
        trigger_reason="MIRROR",
        market_id=paper_request.market_id,
        asset=paper_request.asset,
        outcome_to_buy=paper_request.outcome_to_buy,
        target_amount_usdc=order_amount_usdc,
        requested_shares=None,
        limit_price=paper_request.limit_price,
        max_slippage_pct=paper_request.max_slippage_pct,
        max_spend_usdc=max_spend_usdc,
        max_acceptable_price=paper_request.max_acceptable_price,
        ttl_seconds=ttl_seconds,
        expires_at=expires_at,
        state="READY",
        source_paper_request_id=paper_request.id,
        created_at=now,
        updated_at=now,
    )


# ── Главный цикл ──────────────────────────────────────────────────────────────


async def run_gate(target_mode: str) -> None:
    """Главный цикл release_gate."""

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    logger.info(
        "release_gate запущен. target_mode=%s POLL_INTERVAL=%ss",
        target_mode,
        POLL_INTERVAL,
    )

    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    while not _shutdown:
        try:
            async with Session() as session:
                await release_batch(session, target_mode)
        except Exception:
            logger.exception(
                "Ошибка в release_batch, продолжаем через %ss", POLL_INTERVAL
            )
        await asyncio.sleep(POLL_INTERVAL)

    logger.info("release_gate остановлен.")
    await engine.dispose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Release gate: LiveMirrorCandidate → LIVE ExecutionRequest"
    )
    parser.add_argument(
        "--mode",
        choices=["SHADOW", "LIVE"],
        default="SHADOW",
        help="Целевой режим для LIVE-заявок (SHADOW или LIVE)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        asyncio.run(run_gate(args.mode))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
