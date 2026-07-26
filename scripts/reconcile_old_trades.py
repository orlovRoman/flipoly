import asyncio
import aiohttp
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory

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
        res = await db.execute(select(TradeHistory).where(TradeHistory.position_status == 'OPEN'))
        trades = res.scalars().all()
        print(f"Found {len(trades)} OPEN trades.")
        
        if not trades:
            return

        updated_count = 0
        async with aiohttp.ClientSession() as session:
            for trade in trades:
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
                        if winner_outcome.upper() == str(trade.outcome_bought).upper():
                            pnl = float(trade.size or 0) - float(trade.amount_usdc or 0)
                        else:
                            pnl = -float(trade.amount_usdc or 0)
                        
                        trade.position_status = 'CLOSED'
                        trade.status = 'CLOSED'
                        trade.realized_pnl_usdc = pnl
                        
                        print(f"Trade {trade.id} closed. Winner: {winner_outcome}, PnL: {pnl}")
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
