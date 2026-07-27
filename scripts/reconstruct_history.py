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

async def main():
    async with async_session() as db:
        res = await db.execute(select(TradeHistory).where(TradeHistory.position_status.in_(['OPEN', 'CLOSING', 'PARTIALLY_CLOSED'])))
        trades = res.scalars().all()
        print(f"Found {len(trades)} active trades.")
        
        if not trades:
            return

        updated_count = 0
        async with aiohttp.ClientSession() as session:
            for trade in trades:
                if trade.mode == 'LIVE':
                    print(f"Ignoring LIVE trade ID {trade.id} - should be handled by execution worker via blockchain")
                    continue
                    
                market = await fetch_market(session, trade.market_id)
                if not market:
                    print(f"Market {trade.market_id} not found on Gamma API.")
                    continue
                
                # Check if resolved
                if market.get('closed') or market.get('active') is False:
                    tokens = market.get('tokens', [])
                    winner_token = next((t for t in tokens if t.get('winner')), None)
                    
                    if winner_token:
                        winner_outcome = winner_token.get('outcome', '')
                        
                        # Calculate size and cost robustly
                        size = float(trade.entry_filled_shares) if trade.entry_filled_shares is not None else (float(trade.amount_usdc) / float(trade.executed_price) if trade.executed_price else 0)
                        cost = float(trade.entry_cost_usdc) if trade.entry_cost_usdc is not None else float(trade.amount_usdc)
                        
                        if winner_outcome.upper() == str(trade.outcome_bought).upper():
                            pnl = size - cost
                            close_price = 1.0
                        else:
                            pnl = -cost
                            close_price = 0.0
                        
                        trade.position_status = 'CLOSED'
                        trade.status = 'SUCCESS'
                        trade.pnl = pnl
                        trade.realized_pnl_usdc = pnl
                        trade.close_price = close_price
                        trade.remaining_shares = 0
                        trade.closed_at = datetime.now(timezone.utc)
                        
                        print(f"Trade {trade.id} ({trade.mode}) closed. Winner: {winner_outcome}, PnL: {pnl:.2f}")
                        updated_count += 1
                    else:
                        print(f"Market {trade.market_id} is closed but no winner token found.")
                
                # Sleep briefly to avoid rate limits
                await asyncio.sleep(0.1)
        
        if updated_count > 0:
            await db.commit()
            print(f"Committed {updated_count} updated trades.")
        else:
            print("No trades were updated.")

if __name__ == "__main__":
    asyncio.run(main())
