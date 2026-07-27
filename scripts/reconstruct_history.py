"""
Скрипт восстановления истории сделок на основе on-chain результатов рынка.

Алгоритм:
1. Найти PAPER/SHADOW позиции в активных статусах.
2. Для каждого market_id запросить финальный итог через Gamma API.
3. Восстановить nullable-поля (basis-реконструкция) если нужно.
4. Вызвать settle_resolved_position() из settlement_service.
5. Вывести diff.
6. Если --apply — commit. Иначе rollback.

Примечания:
- LIVE-позиции пропускаются: они требуют on-chain redemption.
- Никакой собственной формулы PnL в этом скрипте нет.
"""

import argparse
import asyncio
import aiohttp
from decimal import Decimal
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from polyflip.execution.settlement_service import (
    settle_resolved_position,
    AccountingInvariantError,
)
from polyflip.execution.states import ACTIVE_POSITION_STATES
from polyflip.collector.resolver import extract_final_outcome

OUTCOME_ALIASES = {"UP": "YES", "DOWN": "NO", "1": "YES", "0": "NO"}


def normalize_outcome(outcome: str) -> str:
    if not outcome:
        return ""
    out = outcome.upper()
    return OUTCOME_ALIASES.get(out, out)


async def fetch_market(http_session, market_id: str) -> dict | None:
    url = "https://gamma-api.polymarket.com/markets"
    params = {"condition_id": market_id}
    async with http_session.get(url, params=params) as resp:
        if resp.status == 200:
            data = await resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
    return None


async def main(apply: bool):
    print(f"Running reconstruct_history... (Apply={apply})")
    async with async_session() as db:
        res = await db.execute(
            select(TradeHistory).where(
                TradeHistory.position_status.in_(ACTIVE_POSITION_STATES),
                # Только PAPER и SHADOW — LIVE нельзя закрывать без on-chain redemption
                TradeHistory.mode.in_(("PAPER", "SHADOW")),
            )
        )
        trades = res.scalars().all()
        print(f"Found {len(trades)} active PAPER/SHADOW trades.")

        if not trades:
            return

        updates: list[tuple] = []
        errors: list[tuple] = []

        async with aiohttp.ClientSession() as http_session:
            for trade in trades:
                market = await fetch_market(http_session, trade.market_id)
                if not market:
                    continue

                if not (market.get("closed") or market.get("active") is False):
                    continue

                # Единый авторитетный определитель исхода — тот же, что использует scheduler.
                # resolvedBy НЕ используется: это адрес resolver-контракта, не исход.
                outcome = extract_final_outcome(market)
                if outcome is None:
                    # Рынок закрыт, но итог ещё не верифицирован
                    continue

                if outcome == "INVALID":
                    payout = Decimal("0.5")
                else:
                    # outcome = "YES" или "NO" — сравниваем с позицией трейдера
                    payout = (
                        Decimal("1")
                        if normalize_outcome(trade.outcome_bought or "") == outcome
                        else Decimal("0")
                    )

                try:
                    old_status = trade.position_status
                    await settle_resolved_position(
                        db,
                        trade_id=trade.id,
                        winning_outcome=outcome,
                        payout_per_share=payout,
                        settlement_fee_usdc=Decimal("0"),
                    )
                    if outcome == "INVALID":
                        trade.status = "INVALID"
                    updates.append(
                        (
                            trade.id,
                            trade.mode,
                            trade.market_id,
                            old_status,
                            "CLOSED",
                            outcome,
                        )
                    )
                except AccountingInvariantError as exc:
                    errors.append((trade.id, str(exc)))

                await asyncio.sleep(0.1)

        print("\n--- Proposed Changes ---")
        print(
            f"{'ID':<6} | {'Mode':<6} | {'Market ID':<44} | {'Old':<20} | {'New':<10} | {'Winner':<10}"
        )
        print("-" * 120)
        for u in updates:
            print(
                f"{u[0]:<6} | {u[1]:<6} | {u[2]:<44} | {u[3]:<20} | {u[4]:<10} | {u[5]:<10}"
            )

        if errors:
            print(f"\n--- Accounting Errors ({len(errors)}) ---")
            for trade_id, err in errors:
                print(f"  Trade {trade_id}: {err}")

        if apply and updates:
            await db.commit()
            print(f"\nCommitted {len(updates)} updated trades to database.")
        elif not apply and updates:
            await db.rollback()
            print("\nDRY RUN: Run with --apply to commit changes.")
        else:
            print("\nNo trades were updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct trade history for PAPER/SHADOW trades using settlement_service"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply changes to the database"
    )
    args = parser.parse_args()

    asyncio.run(main(apply=args.apply))
