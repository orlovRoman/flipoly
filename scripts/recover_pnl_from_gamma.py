import asyncio
import httpx
from decimal import Decimal
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from polyflip.collector.resolver import extract_final_outcome
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recover_pnl")

OUTCOME_ALIASES = {"UP": "YES", "DOWN": "NO", "1": "YES", "0": "NO"}

def normalize_outcome(outcome: str) -> str:
    if not outcome:
        return ""
    out = outcome.upper()
    return OUTCOME_ALIASES.get(out, out)

async def fetch_market(http_session, market_id: str) -> dict | None:
    url = "https://gamma-api.polymarket.com/markets"
    params = {"condition_id": market_id}
    try:
        resp = await http_session.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
    except Exception as e:
        logger.error(f"Error fetching {market_id}: {e}")
    return None

async def main():
    async with async_session() as db:
        res = await db.execute(
            select(TradeHistory).where(
                TradeHistory.status == 'SUCCESS',
                TradeHistory.position_status == 'CLOSED',
                TradeHistory.mode == 'PAPER',
                TradeHistory.pnl == 0,
                TradeHistory.amount_usdc > 0,
                TradeHistory.executed_price > 0
            )
        )
        trades = res.scalars().all()
        logger.info(f"Found {len(trades)} CLOSED PAPER trades with 0 PnL to recover.")

        if not trades:
            return

        updates = 0
        async with httpx.AsyncClient() as http_session:
            for i, trade in enumerate(trades):
                market = await fetch_market(http_session, trade.market_id)
                if not market:
                    continue

                if not (market.get("closed") or market.get("active") is False):
                    continue

                outcome = extract_final_outcome(market)
                if outcome is None:
                    continue

                amount = Decimal(str(trade.amount_usdc))
                price = Decimal(str(trade.executed_price))
                
                if outcome == "INVALID":
                    # Возврат ставки
                    trade.pnl = 0.0
                    trade.realized_pnl_usdc = 0.0
                else:
                    if normalize_outcome(trade.outcome_bought or "") == outcome:
                        # Win
                        payout = amount / price
                        pnl = payout - amount
                    else:
                        # Loss
                        pnl = -amount

                    trade.pnl = float(pnl)
                    trade.realized_pnl_usdc = float(pnl)
                
                # Также проставим closed_at если он NULL
                if not trade.closed_at:
                    trade.closed_at = trade.updated_at
                
                db.add(trade)
                updates += 1
                
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(trades)}... Committing intermediate batch.")
                    await db.commit()
                
                await asyncio.sleep(0.05) # Rate limit protection

        await db.commit()
        logger.info(f"Finished! Successfully recovered PnL for {updates} trades.")

if __name__ == "__main__":
    asyncio.run(main())
