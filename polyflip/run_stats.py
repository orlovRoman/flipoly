import asyncio
import pandas as pd
import numpy as np
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from sqlalchemy import select, and_

async def run():
    async with async_session() as s:
        stmt = select(TradeHistory).where(
            and_(TradeHistory.status == "SUCCESS", TradeHistory.pnl.is_not(None))
        )
        trades = (await s.execute(stmt)).scalars().all()
        if not trades:
            print("No trades")
            return

        df = pd.DataFrame([{
            "id": t.id,
            "asset": t.asset,
            "edge": t.edge,
            "pnl": t.pnl,
            "price": t.executed_price,
            "is_win": 1 if t.pnl > 0 else 0,
            "features": str(t.active_features or "")
        } for t in trades])

        print(f"=== OVERALL ===")
        print(f"Trades: {len(df)} | Total PnL: ${df['pnl'].sum():.2f} | WinRate: {df['is_win'].mean()*100:.1f}%\n")

        # Correlation
        df_edge = df.dropna(subset=["edge"]).copy()
        if not df_edge.empty:
            corr_pnl = df_edge["edge"].corr(df_edge["pnl"])
            corr_win = df_edge["edge"].corr(df_edge["is_win"])
            sp_pnl = df_edge["edge"].corr(df_edge["pnl"], method="spearman")
            sp_win = df_edge["edge"].corr(df_edge["is_win"], method="spearman")
            print(f"Pearson Edge vs PnL: {corr_pnl:.4f} | Spearman: {sp_pnl:.4f}")
            print(f"Pearson Edge vs WinRate: {corr_win:.4f} | Spearman: {sp_win:.4f}\n")

            # Bins
            bins = [-999, 0.0, 0.03, 0.05, 0.10, 0.20, 999]
            labels = ["< 0%", "0-3%", "3-5%", "5-10%", "10-20%", ">= 20%"]
            df_edge["edge_bin"] = pd.cut(df_edge["edge"], bins=bins, labels=labels)
            
            grp = df_edge.groupby("edge_bin", observed=False).agg(
                cnt=("id", "count"),
                win_rate=("is_win", lambda x: round(x.mean()*100, 1) if len(x) > 0 else 0),
                total_pnl=("pnl", lambda x: round(x.sum(), 2)),
                avg_pnl=("pnl", lambda x: round(x.mean(), 3))
            )
            print("=== EDGE BUCKETS ===")
            print(grp.to_string())
            print()

        # Strategy
        def get_strat(r):
            f = r["features"].lower()
            if "аутсайдер" in f or "outsider" in f:
                return "Outsider"
            elif "фаворит" in f or "favorite" in f:
                return "Favorite"
            else:
                return "Outsider" if (r["price"] is not None and r["price"] < 0.5) else "Favorite"

        df["strat"] = df.apply(get_strat, axis=1)
        s_grp = df.groupby("strat").agg(
            cnt=("id", "count"),
            win_rate=("is_win", lambda x: round(x.mean()*100, 1)),
            total_pnl=("pnl", lambda x: round(x.sum(), 2)),
            avg_pnl=("pnl", lambda x: round(x.mean(), 3)),
            avg_edge=("edge", lambda x: round(x.mean(), 4))
        )
        print("=== STRATEGY BUCKETS ===")
        print(s_grp.to_string())
        print()

        # Assets
        a_grp = df.groupby("asset").agg(
            cnt=("id", "count"),
            win_rate=("is_win", lambda x: round(x.mean()*100, 1)),
            total_pnl=("pnl", lambda x: round(x.sum(), 2)),
            avg_pnl=("pnl", lambda x: round(x.mean(), 3)),
            avg_edge=("edge", lambda x: round(x.mean(), 4))
        )
        print("=== ASSET BUCKETS ===")
        print(a_grp.to_string())

if __name__ == "__main__":
    asyncio.run(run())
