"""
Точечное восстановление PnL для конкретного списка сделок.

Отличие от reconstruct_history.py:
- НЕ загружает все активные позиции из БД (нет OOM).
- Принимает явный список trade_id через --ids или авто-запрос через
  --from-query / --from-query-cat-b.
- Обрабатывает батчами с паузами (rate limiting Gamma API).
- Запускать вне основного контейнера: python scripts/targeted_pnl_recovery.py --dry-run

Endpoint Gamma API: GET /markets/{numeric_market_id}
  trade_history.market_id — числовой ID (например 3103319), НЕ condition_id (hex-хэш).

Группы:
  Cat A (--from-query)
      OPEN PAPER/SHADOW с entry_filled_shares > 0 и pnl = NULL/0.
      Это позиции, застрявшие в OPEN после разрешения рынка. Восстанавливаются.

  Cat B (--from-query-cat-b)
      CLOSED PAPER/SHADOW с pnl=0 и entry_filled_shares > 0.
      ⚠  У всех этих сделок remaining_shares=0 (баг rebuild_trade_accounting).
      expected_delta = 0, PnL после apply НЕ изменится. Флаг оставлен для диагностики.

Примеры:
  # Dry-run Cat A (показывает expected_delta до apply):
  python scripts/targeted_pnl_recovery.py --from-query --dry-run

  # Apply Cat A:
  python scripts/targeted_pnl_recovery.py --from-query --apply

  # Диагностика Cat B (не меняет PnL):
  python scripts/targeted_pnl_recovery.py --from-query-cat-b --dry-run
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
from polyflip.collector.resolver import extract_final_outcome

# GET https://gamma-api.polymarket.com/markets/{numeric_market_id}
# trade_history.market_id — числовой ID, НЕ condition_id (hex-хэш).
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
BATCH_SIZE = 10        # сделок за один батч
BATCH_PAUSE_SEC = 1.0  # пауза между батчами (rate limiting)
REQUEST_TIMEOUT = 10   # секунд на один HTTP-запрос

OUTCOME_ALIASES = {"UP": "YES", "DOWN": "NO", "1": "YES", "0": "NO"}


def normalize_outcome(outcome: str) -> str:
    if not outcome:
        return ""
    out = outcome.upper().strip()
    return OUTCOME_ALIASES.get(out, out)


def compute_expected_delta(trade: TradeHistory, payout_per_share: Decimal) -> Decimal | None:
    """
    Рассчитывает ожидаемое изменение PnL ДО вызова settle_resolved_position.

    Формула (как в settle_service):
        payout      = remaining_shares × payout_per_share
        cost_basis  = remaining_shares × (entry_cost_usdc / entry_filled_shares)
        delta       = payout - cost_basis

    Возвращает None если данных недостаточно для расчёта.
    Используется в dry-run для информирования без изменений в БД.
    """
    try:
        remaining = Decimal(str(trade.remaining_shares))
        shares = Decimal(str(trade.entry_filled_shares))
        cost = Decimal(str(trade.entry_cost_usdc))
        if shares == 0:
            return Decimal("0")
        cost_per_share = cost / shares
        payout = remaining * payout_per_share
        cost_basis = remaining * cost_per_share
        return payout - cost_basis
    except Exception:
        return None


async def fetch_market(http_session: aiohttp.ClientSession, market_id: str) -> dict | None:
    """
    GET /markets/{market_id} где market_id — числовой ID из trade_history.market_id.

    Не использовать ?condition_id=X — condition_id это hex-хэш (0x...), числовой
    market_id передаётся только как path-параметр.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with http_session.get(
            f"{GAMMA_API_URL}/{market_id}",
            timeout=timeout,
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"  [WARN] Gamma API {resp.status} for market {market_id}")
    except asyncio.TimeoutError:
        print(f"  [WARN] Timeout: market {market_id}")
    except Exception as exc:
        print(f"  [WARN] Error fetching market {market_id}: {exc}")
    return None


async def load_target_trades(db, trade_ids: list[int]) -> list[TradeHistory]:
    """Загружает только указанные ID — без скана всей таблицы."""
    if not trade_ids:
        return []
    result = await db.execute(
        select(TradeHistory).where(TradeHistory.id.in_(trade_ids))
    )
    return result.scalars().all()


async def load_query_trades(db) -> list[TradeHistory]:
    """Cat A: OPEN PAPER/SHADOW с entry_filled_shares > 0 и нулевым/NULL PnL.

    Намеренно фильтрует position_status == 'OPEN', чтобы Cat B (CLOSED с pnl=0)
    не попали в обработку и не тратили API-запросы впустую.
    """
    result = await db.execute(
        select(TradeHistory).where(
            TradeHistory.position_status == "OPEN",
            TradeHistory.mode.in_(("PAPER", "SHADOW")),
            TradeHistory.entry_filled_shares.isnot(None),
            TradeHistory.entry_filled_shares > 0,
            TradeHistory.realized_pnl_usdc.is_(None)
            | (TradeHistory.realized_pnl_usdc == Decimal("0")),
        )
    )
    return result.scalars().all()


async def load_query_trades_cat_b(db) -> list[TradeHistory]:
    """Cat B: CLOSED PAPER/SHADOW с pnl=0 и entry_filled_shares > 0.

    ⚠  Диагностический режим. У всех этих сделок remaining_shares=0 из-за бага
    rebuild_trade_accounting, поэтому expected_delta всегда будет 0. Apply не
    изменит PnL. Используй --dry-run для проверки, --apply не рекомендуется.
    """
    result = await db.execute(
        select(TradeHistory).where(
            TradeHistory.position_status == "CLOSED",
            TradeHistory.mode.in_(("PAPER", "SHADOW")),
            TradeHistory.entry_filled_shares.isnot(None),
            TradeHistory.entry_filled_shares > 0,
            TradeHistory.realized_pnl_usdc == Decimal("0"),
        )
    )
    return result.scalars().all()


async def process_trade(
    http_session, db, trade: TradeHistory, apply: bool, cat_b: bool = False
) -> dict:
    """
    Обрабатывает одну сделку. Возможные результаты:
      SKIP_NOT_CLOSED   – рынок ещё активен
      SKIP_NO_OUTCOME   – рынок закрыт, итог не верифицирован
      SKIP_MARKET_ERROR – Gamma API не ответил
      MANUAL_REVIEW     – AccountingInvariantError
      SETTLED           – успешно закрыто

    cat_b=True: сделка уже CLOSED с pnl=0 — временно сбрасываем в OPEN,
    чтобы settle_resolved_position смогла её пересчитать.
    expected_delta в этом случае будет 0 (remaining_shares=0).
    """
    base = {
        "id": trade.id, "mode": trade.mode,
        "market_id": trade.market_id, "old_status": trade.position_status,
        "result": None, "outcome": None,
        "payout_per_share": None, "expected_delta": None, "error": None,
    }

    market = await fetch_market(http_session, trade.market_id)
    if not market:
        return {**base, "result": "SKIP_MARKET_ERROR"}

    is_closed = market.get("closed") or (market.get("active") is False)
    if not is_closed:
        return {**base, "result": "SKIP_NOT_CLOSED"}

    outcome = extract_final_outcome(market)
    if outcome is None:
        return {**base, "result": "SKIP_NO_OUTCOME"}

    if outcome == "INVALID":
        payout_per_share = Decimal("0.5")
    else:
        normalized_bought = normalize_outcome(str(trade.outcome_bought or ""))
        payout_per_share = Decimal("1") if normalized_bought == outcome else Decimal("0")

    # Рассчитываем ожидаемую дельту PnL ДО вызова settle.
    # Это позволяет видеть в dry-run, что реально изменится (или не изменится).
    expected_delta = compute_expected_delta(trade, payout_per_share)

    # Cat B: settle_resolved_position пропускает CLOSED записи (идемпотентная защита).
    # Временно переводим в OPEN, чтобы сервис мог пересчитать PnL.
    patched_status = None
    if cat_b and trade.position_status == "CLOSED":
        patched_status = trade.position_status
        trade.position_status = "OPEN"

    try:
        await settle_resolved_position(
            db,
            trade_id=trade.id,
            winning_outcome=outcome,
            payout_per_share=payout_per_share,
            settlement_fee_usdc=Decimal("0"),
        )
        return {
            **base,
            "result": "SETTLED",
            "outcome": outcome,
            "payout_per_share": str(payout_per_share),
            "expected_delta": expected_delta,
        }
    except AccountingInvariantError as exc:
        # Откатываем патч статуса при ошибке
        if patched_status is not None:
            trade.position_status = patched_status
        return {**base, "result": "MANUAL_REVIEW", "error": str(exc)}


def print_report(results: list[dict], apply: bool) -> None:
    from collections import Counter
    counts = Counter(r["result"] for r in results)

    print("\n" + "=" * 114)
    print(f"{'TARGETED PnL RECOVERY — ' + ('APPLIED' if apply else 'DRY RUN'):^114}")
    print("=" * 114)

    settled = [r for r in results if r["result"] == "SETTLED"]

    if settled:
        # Разбиваем на реально восстановленные и нулевые (remaining=0)
        recovered = [r for r in settled if r["expected_delta"] is not None and r["expected_delta"] != 0]
        zero_delta = [r for r in settled if r["expected_delta"] is None or r["expected_delta"] == 0]

        header = f"{'ID':<8} {'Mode':<8} {'Market ID':<14} {'Old Status':<22} {'Outcome':<10} {'Payout/sh':<12} Expected Δ PnL"
        divider = "-" * 114

        if recovered:
            print(f"\n--- ✅ SETTLED with real PnL change ({len(recovered)}) ---")
            print(header)
            print(divider)
            for r in recovered:
                delta_str = f"{r['expected_delta']:+.6f}" if r["expected_delta"] is not None else "?"
                print(
                    f"{r['id']:<8} {r['mode']:<8} {r['market_id']:<14} "
                    f"{r['old_status']:<22} {r['outcome']:<10} {r['payout_per_share']:<12} {delta_str}"
                )

        if zero_delta:
            print(f"\n--- ⚠  SETTLED but delta_pnl=0 ({len(zero_delta)}) — remaining_shares=0, PnL не изменится ---")
            print(header)
            print(divider)
            for r in zero_delta:
                print(
                    f"{r['id']:<8} {r['mode']:<8} {r['market_id']:<14} "
                    f"{r['old_status']:<22} {r['outcome'] or '?':<10} {r['payout_per_share'] or '?':<12} 0.000000"
                )

    manual = [r for r in results if r["result"] == "MANUAL_REVIEW"]
    if manual:
        print(f"\n--- MANUAL_REVIEW_REQUIRED ({len(manual)}) — needs human attention ---")
        for r in manual:
            print(f"  Trade {r['id']}: {r['error']}")

    print("\n--- Summary ---")
    real_recovered = sum(
        1 for r in results
        if r["result"] == "SETTLED"
        and r.get("expected_delta") is not None
        and r["expected_delta"] != 0
    )
    if real_recovered:
        print(f"  {'SETTLED (with PnL change)':<30} : {real_recovered}")
    zero_count = counts.get("SETTLED", 0) - real_recovered
    if zero_count:
        print(f"  {'SETTLED (delta=0, no change)':<30} : {zero_count}")
    for status, count in sorted(counts.items()):
        if status != "SETTLED":
            print(f"  {status:<30} : {count}")

    if not apply:
        print("\n⚠  DRY RUN — no changes committed. Re-run with --apply to persist.")
    else:
        print(f"\n✅ APPLIED — {real_recovered} trades with real PnL change, "
              f"{zero_count} settled with delta=0.")


async def main(
    trade_ids: list[int], use_query: bool, use_query_cat_b: bool, apply: bool
) -> None:
    if use_query_cat_b:
        source_label = "query Cat B (CLOSED pnl=0 with shares — diagnostic only)"
    elif use_query:
        source_label = "query Cat A (OPEN with fills)"
    else:
        source_label = f"{len(trade_ids)} explicit IDs"
    print(f"\nTargeted PnL Recovery | apply={apply} | source={source_label}")

    async with async_session() as db:
        if use_query_cat_b:
            trades = await load_query_trades_cat_b(db)
        elif use_query:
            trades = await load_query_trades(db)
        else:
            trades = await load_target_trades(db, trade_ids)

        print(f"Loaded {len(trades)} trades to process.")
        if not trades:
            print("Nothing to do.")
            return

        results: list[dict] = []

        async with aiohttp.ClientSession() as http_session:
            for batch_start in range(0, len(trades), BATCH_SIZE):
                batch = trades[batch_start : batch_start + BATCH_SIZE]
                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (len(trades) + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"  Batch {batch_num}/{total_batches} ({len(batch)} trades)...")

                for trade in batch:
                    r = await process_trade(
                        http_session, db, trade, apply, cat_b=use_query_cat_b
                    )
                    results.append(r)

                if batch_start + BATCH_SIZE < len(trades):
                    await asyncio.sleep(BATCH_PAUSE_SEC)

        settled_count = sum(1 for r in results if r["result"] == "SETTLED")
        if apply and settled_count > 0:
            await db.commit()
        else:
            await db.rollback()

        print_report(results, apply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Targeted PnL recovery. Run outside the API container."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", nargs="+", type=int, metavar="ID",
                       help="Explicit trade IDs, e.g. --ids 18473 18481")
    group.add_argument("--from-query", action="store_true",
                       help="Cat A: OPEN PAPER/SHADOW trades with fills and missing PnL")
    group.add_argument("--from-query-cat-b", action="store_true",
                       help="Cat B: CLOSED pnl=0 with shares (diagnostic only, won't change PnL)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")

    args = parser.parse_args()
    asyncio.run(main(
        trade_ids=args.ids or [],
        use_query=args.from_query,
        use_query_cat_b=args.from_query_cat_b,
        apply=args.apply,
    ))
