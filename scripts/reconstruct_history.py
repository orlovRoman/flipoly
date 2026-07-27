import argparse
import asyncio
import aiohttp
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from datetime import datetime, timezone

async def fetch_market(session, market_id):
    url = f"https://gamma-api.polymarket.com/markets"
    params = {"condition_id": market_id}
    async with session.get(url, params=params) as resp:
        if resp.status == 200:
            data = await resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
        return None

async def main(apply: bool):
    print(f"Running reconstruct_history... (Apply={apply})")
    async with async_session() as db:
        res = await db.execute(select(TradeHistory).where(TradeHistory.position_status.in_(['OPEN', 'CLOSING', 'PARTIALLY_CLOSED'])))
        trades = res.scalars().all()
        print(f"Found {len(trades)} active trades.")
        
        if not trades:
            return

        updates = []
        async with aiohttp.ClientSession() as session:
            for trade in trades:
                if trade.mode == 'LIVE':
                    continue
                    
                market = await fetch_market(session, trade.market_id)
                if not market:
                    continue
                
                if market.get('closed') or market.get('active') is False:
                    tokens = market.get('tokens', [])
                    winner_token = next((t for t in tokens if t.get('winner')), None)
                    
                    if winner_token:
                        winner_outcome = winner_token.get('outcome', '')
                        
                        size = float(trade.entry_filled_shares) if trade.entry_filled_shares is not None else (float(trade.amount_usdc) / float(trade.executed_price) if trade.executed_price else 0)
                        cost = float(trade.entry_cost_usdc) if trade.entry_cost_usdc is not None else float(trade.amount_usdc)
                        
                        if winner_outcome.upper() == str(trade.outcome_bought).upper():
                            pnl = size - cost
                            close_price = 1.0
                        else:
                            pnl = -cost
                            close_price = 0.0
                        
                        old_status = trade.position_status
                        
                        if apply:
                            trade.position_status = 'CLOSED'
                            trade.status = 'SUCCESS'
                            trade.pnl = pnl
                            trade.realized_pnl_usdc = pnl
                            trade.close_price = close_price
                            trade.remaining_shares = 0
                            trade.closed_at = datetime.now(timezone.utc)
                            
                        updates.append((trade.id, trade.mode, trade.market_id, old_status, "CLOSED", pnl, winner_outcome))
                        
                await asyncio.sleep(0.1)
        
        print("\n--- Proposed Changes ---")
        print(f"{'ID':<6} | {'Mode':<6} | {'Market ID':<44} | {'Old':<10} | {'New':<10} | {'PnL':<8} | {'Winner':<10}")
        print("-" * 115)
        for u in updates:
            print(f"{u[0]:<6} | {u[1]:<6} | {u[2]:<44} | {u[3]:<10} | {u[4]:<10} | {u[5]:<8.2f} | {u[6]:<10}")
            
        if apply and updates:
            await db.commit()
            print(f"\nCommitted {len(updates)} updated trades to database.")
        elif not apply and updates:
            print("\nDRY RUN: Run with --apply to commit changes.")
        else:
            print("\nNo trades were updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct trade history for PAPER trades")
    parser.add_argument("--apply", action="store_true", help="Apply changes to the database")
    args = parser.parse_args()
    
    asyncio.run(main(apply=args.apply))
