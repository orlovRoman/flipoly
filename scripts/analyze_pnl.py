import asyncio
from sqlalchemy import select
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
import pandas as pd

async def analyze():
    print("Connecting to database...")
    async with async_session() as session:
        stmt = select(TradeHistory).where(TradeHistory.status == 'SUCCESS').where(TradeHistory.pnl.is_not(None))
        result = await session.execute(stmt)
        trades = result.scalars().all()

    if not trades:
        print("No trades with PnL found.")
        return

    data = []
    for t in trades:
        af = (t.active_features or "").upper()
        
        # Determine strategy type
        if "OUTSIDER" in af:
            strategy = "OUTSIDER"
        elif "FAVORITE" in af:
            strategy = "FAVORITE"
        elif "CRYPTO" in af:
            strategy = "CRYPTO"
        else:
            strategy = "OTHER"

        # Base asset (e.g., BTC_15m -> BTC, ETH_15m -> ETH, or just asset if it's already BTC)
        # Polymarket bot uses something like "BTC" or "ETH"
        base_asset = t.asset.split('_')[0]

        data.append({
            "asset": base_asset,
            "mode": t.mode,
            "strategy": strategy,
            "pnl": t.pnl,
            "amount_usdc": t.amount_usdc
        })

    df = pd.DataFrame(data)

    print("\n=== PNL BY ASSET AND STRATEGY ===")
    
    # Analyze by strategy
    for strategy in ["FAVORITE", "OUTSIDER"]:
        print(f"\n--- Strategy: {strategy} ---")
        df_strat = df[df['strategy'] == strategy]
        
        if df_strat.empty:
            print("No trades for this strategy.")
            continue
            
        print(f"Total trades: {len(df_strat)}")
        print(f"Total PnL: {df_strat['pnl'].sum():.2f} USDC")
        
        # Group by asset
        grouped = df_strat.groupby('asset').agg(
            trades=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            avg_pnl=('pnl', 'mean'),
            win_rate=('pnl', lambda x: (x > 0).mean() * 100)
        ).round(2)
        
        print("\nBy Asset:")
        print(grouped.to_string())

    print("\n--- Summary All Strategies ---")
    summary = df.groupby(['strategy', 'asset']).agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100)
    ).round(2)
    print(summary.to_string())

if __name__ == "__main__":
    asyncio.run(analyze())
