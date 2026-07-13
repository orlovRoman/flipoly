import asyncio
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory, MarketSnapshot
import pandas as pd

async def analyze():
    print("Connecting to the database...")
    try:
        async with async_session() as session:
            print("Session opened, executing query...")
            # Get all successful trades
            stmt = select(TradeHistory).where(TradeHistory.status == 'SUCCESS')
            result = await session.execute(stmt)
            trades = result.scalars().all()
            print(f"Query completed, got {len(trades)} trades")

            # Get outcomes for these markets
            market_ids = list({t.market_id for t in trades})
            print(f"Fetching outcomes for {len(market_ids)} unique markets...")
            
            # chunking market_ids if too many
            outcomes = {}
            chunk_size = 500
            for i in range(0, len(market_ids), chunk_size):
                chunk = market_ids[i:i+chunk_size]
                out_stmt = select(MarketSnapshot.market_id, MarketSnapshot.final_outcome).where(
                    MarketSnapshot.market_id.in_(chunk)
                ).distinct()
                out_res = await session.execute(out_stmt)
                for mid, outc in out_res.all():
                    # We only care if it resolved to YES or NO or INVALID, not PENDING
                    if outc != "PENDING":
                        outcomes[mid] = outc
                        
            # Map trades and outcomes
            rows = []
            for t in trades:
                rows.append((t, outcomes.get(t.market_id)))
    except Exception as e:
        print(f"ERROR during query: {e}")
        import traceback
        traceback.print_exc()
        return

    if not rows:
        print("No successful trades found.")
        return

    data = []
    for trade, final_outcome in rows:
        data.append({
            "id": trade.id,
            "market_id": trade.market_id,
            "asset": trade.asset,
            "mode": trade.mode,
            "outcome_bought": trade.outcome_bought,
            "amount_usdc": trade.amount_usdc,
            "executed_price": trade.executed_price,
            "stop_loss_status": trade.stop_loss_status,
            "stop_loss_pct": trade.stop_loss_pct,
            "stop_loss_price": trade.stop_loss_price,
            "stop_loss_sell_price": getattr(trade, 'stop_loss_sell_price', None),
            "pnl": trade.pnl,
            "final_outcome": final_outcome
        })

    df = pd.DataFrame(data)
    
    df['market_resolved'] = df['final_outcome'].notna() & df['final_outcome'].isin(['YES', 'NO'])
    df['was_correct'] = (df['outcome_bought'] == df['final_outcome'])

    print("=== STOP LOSS ANALYSIS ===")
    print(f"Total trades: {len(df)}")
    print(f"Resolved trades: {df['market_resolved'].sum()}")
    
    # Analyze triggered stop losses
    triggered = df[df['stop_loss_status'] == 'TRIGGERED']
    print(f"\nTotal trades closed by stop-loss: {len(triggered)}")
    
    resolved_triggered = triggered[triggered['market_resolved']]
    print(f"Resolved trades closed by stop-loss: {len(resolved_triggered)}")
    
    if len(resolved_triggered) > 0:
        correct_but_stopped = resolved_triggered[resolved_triggered['was_correct']]
        incorrect_and_stopped = resolved_triggered[~resolved_triggered['was_correct']]
        
        print(f"\n1. Stop-Loss was a 'MISTAKE' (Market outcome turned out to be CORRECT in the end):")
        pct_mistake = len(correct_but_stopped) / len(resolved_triggered) * 100
        print(f"   Count: {len(correct_but_stopped)} ({pct_mistake:.1f}%)")
        print(f"   (We lost money by stopping out, but would have won if we held to the end)")
        
        print(f"\n2. Stop-Loss was 'EFFECTIVE' (Market outcome turned out to be INCORRECT):")
        pct_effective = len(incorrect_and_stopped) / len(resolved_triggered) * 100
        print(f"   Count: {len(incorrect_and_stopped)} ({pct_effective:.1f}%)")
        print(f"   (We correctly stopped out and saved part of the deposit, because we would have lost 100% anyway)")
        
        print(f"\n--- PnL Impact of Stop Losses (Resolved trades) ---")
        total_pnl = resolved_triggered['pnl'].sum()
        print(f"Total PnL on these triggered stop losses: {total_pnl:.2f} USDC")
        
        # Calculate theoretical PnL if we didn't use stop loss
        # If mistake: we would have won. Win PnL = (1 - executed_price) * shares = (1/executed_price - 1) * amount_usdc
        theoretical_win_pnl = ((1 / correct_but_stopped['executed_price'] - 1) * correct_but_stopped['amount_usdc']).sum()
        
        # If effective: we would have lost 100%. Loss PnL = -amount_usdc
        theoretical_loss_pnl = -incorrect_and_stopped['amount_usdc'].sum()
        
        theoretical_total_pnl = theoretical_win_pnl + theoretical_loss_pnl
        print(f"Theoretical PnL if we DID NOT use stop losses: {theoretical_total_pnl:.2f} USDC")
        print(f"Difference (Value of Stop Loss feature): {total_pnl - theoretical_total_pnl:.2f} USDC")

    print("\n--- By Mode ---")
    if not triggered.empty:
        print(triggered.groupby('mode').size().to_string())

if __name__ == "__main__":
    asyncio.run(analyze())
