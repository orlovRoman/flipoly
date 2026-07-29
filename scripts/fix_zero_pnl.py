"""
Fix 59 PAPER trades closed with zero PnL.
Calls Gamma API per market_id, then sets close_price + realized_pnl_usdc.
"""
import asyncio
import argparse
import aiohttp
from decimal import Decimal
from sqlalchemy import text
from polyflip.db.connection import async_session

GAMMA_URL = "https://gamma-api.polymarket.com/markets"

OUTCOME_ALIASES = {"UP": "YES", "DOWN": "NO", "1": "YES", "0": "NO"}


def normalize(outcome: str) -> str:
    return OUTCOME_ALIASES.get(outcome.upper(), outcome.upper())


async def fetch_market(session, market_id: str) -> dict | None:
    try:
        async with session.get(
            GAMMA_URL,
            params={"id": market_id},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.status == 200:
                data = await r.json()
                if isinstance(data, list) and data:
                    return data[0]
    except Exception as e:
        print(f"  [WARN] API error for {market_id}: {e}")
    return None


def get_winning_outcome(market: dict) -> str | None:
    """Returns YES, NO, INVALID or None if not resolved."""
    if not (market.get("closed") or market.get("active") is False):
        return None

    outcome_prices = market.get("outcomePrices")
    outcomes = market.get("outcomes")

    if outcome_prices and outcomes:
        try:
            prices = [float(p) for p in outcome_prices]
            for i, price in enumerate(prices):
                if price >= 0.99:
                    return normalize(outcomes[i])
        except Exception:
            pass

    resolved_by = market.get("resolvedBy")
    if resolved_by:
        return normalize(str(resolved_by))

    return None


async def main(apply: bool):
    print(f"=== fix_zero_pnl.py (apply={apply}) ===")

    async with async_session() as db:
        result = await db.execute(
            text("""
                SELECT id, market_id, outcome_bought,
                       entry_cost_usdc, entry_filled_shares, remaining_shares
                FROM trade_history
                WHERE position_status = 'CLOSED'
                  AND close_price IS NULL
                  AND realized_pnl_usdc = 0
                  AND entry_cost_usdc IS NOT NULL
                ORDER BY id
            """)
        )
        trades = result.mappings().all()

    print(f"Found {len(trades)} trades to process.")

    market_cache: dict[str, dict | None] = {}
    updates = []

    async with aiohttp.ClientSession() as http:
        for trade in trades:
            mid = trade["market_id"]
            if mid not in market_cache:
                print(f"  Fetching market {mid}...")
                market_cache[mid] = await fetch_market(http, mid)
                await asyncio.sleep(0.15)

            market = market_cache[mid]
            if not market:
                print(f"  [SKIP] id={trade['id']}: market {mid} not found in API")
                continue

            winner = get_winning_outcome(market)
            if winner is None:
                print(f"  [SKIP] id={trade['id']}: market {mid} not resolved yet")
                continue

            our_side = (trade["outcome_bought"] or "").upper()
            sold_shares = (
                Decimal(str(trade["entry_filled_shares"]))
                - Decimal(str(trade["remaining_shares"]))
            )
            entry_cost = Decimal(str(trade["entry_cost_usdc"]))

            if winner == "INVALID":
                close_price = Decimal("0.5")
            elif our_side == winner:
                close_price = Decimal("1.0")
            else:
                close_price = Decimal("0.0")

            pnl = sold_shares * close_price - entry_cost

            updates.append(
                {
                    "id": trade["id"],
                    "market_id": mid,
                    "outcome_bought": our_side,
                    "winner": winner,
                    "close_price": float(close_price),
                    "realized_pnl_usdc": float(pnl),
                }
            )

    print("\n--- Proposed changes ---")
    print(f"{'ID':<7} {'Market':<10} {'Bought':<6} {'Winner':<8} {'close_price':<13} {'PnL'}")
    print("-" * 65)
    for u in updates:
        print(
            f"{u['id']:<7} {u['market_id']:<10} {u['outcome_bought']:<6} "
            f"{u['winner']:<8} {u['close_price']:<13.4f} {u['realized_pnl_usdc']:.4f}"
        )

    wins = sum(1 for u in updates if u["realized_pnl_usdc"] > 0)
    losses = sum(1 for u in updates if u["realized_pnl_usdc"] < 0)
    print(f"\nTotal: {len(updates)} | Wins: {wins} | Losses: {losses}")

    if apply and updates:
        async with async_session() as db:
            for u in updates:
                await db.execute(
                    text("""
                        UPDATE trade_history
                        SET close_price = :cp, realized_pnl_usdc = :pnl
                        WHERE id = :id
                    """),
                    {"cp": u["close_price"], "pnl": u["realized_pnl_usdc"], "id": u["id"]},
                )
            await db.commit()
        print(f"Committed {len(updates)} rows.")
    elif not apply:
        print("\nDRY RUN — run with --apply to commit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(apply=args.apply))
