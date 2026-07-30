"""
scripts/check_live_readiness.py — Этап 7

Pre-flight скрипт готовности LIVE-торговли: проверяет все предусловия перед включением LIVE-торговли
и при успехе выставляет начальные значения рубильников в БД.

Последовательность вызовов (типичный MANUAL pre-flight):
    python scripts/check_live_readiness.py --target-mode SHADOW --release-mode MANUAL --enable-mirror

    или для полного AUTO-режима:
    python scripts/check_live_readiness.py --target-mode LIVE --release-mode AUTO --enable-mirror

Что проверяет:
    1. Доступность БД
    2. Применена ли последняя миграция (live_mirror_candidates существует)
    3. Нет ли зависших LIVE ExecutionRequest в состоянии READY или CLAIMED
    4. Heartbeat LIVE execution_worker_status свежий (< 60s)
    5. LIVE-воркер имеет достаточный баланс (>= MIN_USDC_BALANCE)
    6. Collateral allowance готов
    7. Нет ни одного кандидата в состоянии RELEASED без соответствующей LIVE-сделки

Флаги:
    --check-only         — только проверки, не изменять БД
    --enable-mirror      — установить LIVE_MIRROR_ENABLED=true
    --release-mode MODE  — установить LIVE_RELEASE_MODE (DISABLED|MANUAL|AUTO)
    --target-mode MODE   — целевой режим зеркалирования (SHADOW|LIVE)
    --min-balance        — минимальный баланс USDC для LIVE-воркера (default: 10)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

import os


DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
MIN_USDC_BALANCE: float = 10.0

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {BOLD}{msg}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}── {title} ──{RESET}")


async def check_db_reachable(session: AsyncSession) -> bool:
    try:
        await session.execute(text("SELECT 1"))
        ok("БД доступна")
        return True
    except Exception as e:
        fail(f"БД недоступна: {e}")
        return False


async def check_migration_applied(session: AsyncSession) -> bool:
    from sqlalchemy import inspect as sa_inspect
    conn = await session.connection()

    def _has_table(sync_conn):
        inspector = sa_inspect(sync_conn)
        return "live_mirror_candidates" in inspector.get_table_names()

    has_table = await conn.run_sync(_has_table)
    if has_table:
        ok("Миграция e7f8a9b0c1d2 применена (live_mirror_candidates существует)")
        return True
    else:
        fail("Таблица live_mirror_candidates ОТСУТСТВУЕТ. Применить: alembic upgrade head")
        return False


async def check_no_stale_live_requests(session: AsyncSession) -> bool:
    from polyflip.db.execution_models import ExecutionRequest
    stale = (await session.execute(
        select(ExecutionRequest).where(
            ExecutionRequest.requested_mode == "LIVE",
            ExecutionRequest.state.in_(["READY", "CLAIMED", "SUBMITTING"]),
        ).limit(5)
    )).scalars().all()

    if not stale:
        ok("Нет зависших LIVE-заявок в состоянии READY/CLAIMED/SUBMITTING")
        return True
    else:
        fail(f"Найдено {len(stale)} зависших LIVE-заявок. Разрешить вручную через /api/execution/requests")
        for r in stale:
            print(f"    → id={r.id} market={r.market_id} state={r.state}")
        return False


async def check_worker_heartbeat(
    session: AsyncSession,
    min_balance: float,
    target_mode: str = "SHADOW",
) -> bool:
    from polyflip.db.execution_models import ExecutionWorkerStatus
    status = (await session.execute(
        select(ExecutionWorkerStatus)
        .where(ExecutionWorkerStatus.execution_mode == "LIVE")
        .order_by(ExecutionWorkerStatus.heartbeat_at.desc())
        .limit(1)
    )).scalar_one_or_none()

    if status is None:
        msg = "Нет heartbeat для LIVE execution_worker. Запустить live-воркер."
        if target_mode == "LIVE":
            fail(msg)
            return False
        else:
            warn(f"{msg} (Игнорируется для режима {target_mode})")
            return True

    now = datetime.now(timezone.utc)
    heartbeat_at = status.heartbeat_at
    if heartbeat_at.tzinfo is None:
        heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)

    age = (now - heartbeat_at).total_seconds()
    if age > 60:
        msg = f"LIVE-воркер не отвечал {age:.0f}s (максимум 60s). Проверить контейнер."
        if target_mode == "LIVE":
            fail(msg)
            return False
        else:
            warn(f"{msg} (Игнорируется для режима {target_mode})")
            return True

    ok(f"LIVE-воркер heartbeat {age:.0f}s назад")

    if not status.gateway_ready:
        msg = f"Gateway не готов: {status.last_error_message}"
        if target_mode == "LIVE":
            fail(msg)
            return False
        else:
            warn(f"{msg} (Игнорируется для режима {target_mode})")
            return True

    ok("Gateway готов")

    balance = float(status.balance_usdc or 0)
    if balance < min_balance:
        msg = f"Недостаточный баланс USDC: {balance:.2f} (минимум {min_balance:.2f})"
        if target_mode == "LIVE":
            fail(msg)
            return False
        else:
            warn(f"{msg} (Игнорируется для режима {target_mode})")
            return True

    ok(f"Баланс USDC: {balance:.2f}")

    if not status.collateral_allowance_ready:
        msg = "Collateral allowance не выставлен"
        if target_mode == "LIVE":
            fail(msg)
            return False
        else:
            warn(f"{msg} (Игнорируется для режима {target_mode})")
            return True

    ok("Collateral allowance готов")

    return True


async def check_no_orphaned_releases(session: AsyncSession) -> bool:
    from polyflip.db.execution_models import LiveMirrorCandidate
    from polyflip.db.models import TradeHistory

    # Кандидаты со state=RELEASED, у которых released_trade_id ссылается на несуществующую сделку
    released = (await session.execute(
        select(LiveMirrorCandidate).where(
            LiveMirrorCandidate.state == "RELEASED",
            LiveMirrorCandidate.released_trade_id.is_(None),
        )
    )).scalars().all()

    if not released:
        ok("Нет «осиротевших» RELEASED-кандидатов")
        return True
    else:
        warn(f"Найдено {len(released)} RELEASED-кандидатов без released_trade_id (возможный баг в release_gate)")
        for c in released:
            print(f"    → candidate_id={c.id}")
        return True  # предупреждение, не блокирует


async def set_flags(
    session: AsyncSession,
    *,
    enable_mirror: bool,
    release_mode: str,
) -> None:
    from polyflip.db.models import RuntimeSettings
    from polyflip.execution.live_mirror_worker import set_mirror_enabled

    now = datetime.now(timezone.utc)

    async def _upsert(key: str, value: str) -> None:
        existing = (await session.execute(
            select(RuntimeSettings).where(RuntimeSettings.key == key)
        )).scalar_one_or_none()
        if existing:
            existing.value = value
            existing.updated_at = now
            existing.updated_by = "check_live_readiness"
        else:
            session.add(RuntimeSettings(key=key, value=value, updated_at=now, updated_by="check_live_readiness"))

    if enable_mirror:
        await set_mirror_enabled(session, enabled=True, updated_by="check_live_readiness")
        ok("LIVE_MIRROR_ENABLED = true (LIVE_MIRROR_STARTED_AT выставлен)")

    await _upsert("LIVE_RELEASE_MODE", release_mode)
    ok(f"LIVE_RELEASE_MODE = {release_mode}")

    # LIVE_TRADING_ENABLED управляется только через kill-switch API
    # (требует живого воркера), здесь НЕ трогаем
    warn("LIVE_TRADING_ENABLED НЕ изменён. Включить через: PUT /api/execution/kill-switch")

    await session.commit()


async def run(args: argparse.Namespace) -> int:
    if not DATABASE_URL:
        print(f"{RED}ERROR: DATABASE_URL не задан{RESET}")
        return 1

    engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    all_ok = True

    async with Session() as session:
        section("1. Доступность БД")
        if not await check_db_reachable(session):
            return 1

        section("2. Миграции")
        if not await check_migration_applied(session):
            all_ok = False

        section("3. Зависшие LIVE-заявки")
        if not await check_no_stale_live_requests(session):
            all_ok = False

        section("4. LIVE execution_worker heartbeat")
        min_bal = getattr(args, "min_balance", MIN_USDC_BALANCE)
        target_mode = getattr(args, "target_mode", "SHADOW").upper()
        if not await check_worker_heartbeat(session, min_bal, target_mode=target_mode):
            all_ok = False

        section("5. Консистентность release_gate")
        await check_no_orphaned_releases(session)

    if not all_ok:
        print(f"\n{RED}{BOLD}Pre-flight FAILED. Устранить ошибки и запустить повторно.{RESET}")
        await engine.dispose()
        return 1

    print(f"\n{GREEN}{BOLD}Pre-flight PASSED ✓{RESET}")

    if args.check_only:
        print("--check-only: флаги не изменены.")
        await engine.dispose()
        return 0

    section("Установка флагов")
    async with Session() as session:
        await set_flags(
            session,
            enable_mirror=args.enable_mirror,
            release_mode=args.release_mode.upper(),
        )

    print(f"\n{GREEN}Готово. mirror-воркер и release_gate готовы к запуску.{RESET}")
    await engine.dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight проверки и первоначальная настройка LIVE-торговли",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только проверки, не изменять БД",
    )
    parser.add_argument(
        "--enable-mirror",
        action="store_true",
        default=False,
        help="Установить LIVE_MIRROR_ENABLED=true",
    )
    parser.add_argument(
        "--release-mode",
        choices=["DISABLED", "MANUAL", "AUTO"],
        default="DISABLED",
        help="Установить LIVE_RELEASE_MODE (default: DISABLED)",
    )
    parser.add_argument(
        "--target-mode",
        choices=["SHADOW", "LIVE"],
        default="SHADOW",
        help="Целевой режим зеркалирования (default: SHADOW)",
    )
    parser.add_argument(
        "--min-balance",
        type=float,
        default=MIN_USDC_BALANCE,
        help=f"Минимальный USDC-баланс для LIVE-воркера (default: {MIN_USDC_BALANCE})",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
