import asyncio
import pandas as pd
import numpy as np
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from sqlalchemy import select, and_

async def run_analysis():
    async with async_session() as session:
        stmt = select(
            TradeHistory.id,
            TradeHistory.asset,
            TradeHistory.active_features,
            TradeHistory.predicted_flip_prob,
            TradeHistory.executed_price,
            TradeHistory.edge,
            TradeHistory.amount_usdc,
            TradeHistory.pnl,
            TradeHistory.created_at
        ).where(
            and_(
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None)
            )
        )
        
        rows = (await session.execute(stmt)).all()
        if not rows:
            print("No trades found.")
            return

        df = pd.DataFrame([
            {
                "id": r.id,
                "asset": r.asset,
                "features": str(r.active_features or ""),
                "prob": r.predicted_flip_prob,
                "price": r.executed_price,
                "edge": r.edge,
                "amount": r.amount_usdc,
                "pnl": r.pnl,
                "is_win": 1 if r.pnl > 0 else 0
            }
            for r in rows
        ])

        print(f"Total trades analyzed: {len(df)}")
        print(f"Total PnL: ${df['pnl'].sum():.2f}")
        print(f"Win Rate: {df['is_win'].mean()*100:.1f}%\n")

        # Корреляции
        df_edge = df.dropna(subset=["edge"]).copy()
        if not df_edge.empty:
            print("--- CORRELATIONS ---")
            print(f"Pearson (Edge vs PnL): {df_edge['edge'].corr(df_edge['pnl']):.4f}")
            print(f"Spearman (Edge vs PnL): {df_edge['edge'].corr(df_edge['pnl'], method='spearman'):.4f}")
            print(f"Pearson (Edge vs Win): {df_edge['edge'].corr(df_edge['is_win']):.4f}")
            print(f"Spearman (Edge vs Win): {df_edge['edge'].corr(df_edge['is_win'], method='spearman'):.4f}\n")

            # Бакеты по Edge
            bins = [-np.inf, 0.0, 0.02, 0.05, 0.10, 0.20, np.inf]
            labels = ["<0%", "0-2%", "2-5%", "5-10%", "10-20%", ">=20%"]
            df_edge["edge_bucket"] = pd.cut(df_edge["edge"], bins=bins, labels=labels)
            
            b_stats = df_edge.groupby("edge_bucket", observed=False).agg(
                trades=("id", "count"),
                win_rate=("is_win", lambda x: round(x.mean()*100, 1)),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean")
            )
            print("--- EDGE BUCKETS ---")
            print(b_stats.to_string())
            print()

        # Стратегия (Outsider vs Favorite)
        def strat_name(r):
            f = r["features"].lower()
            if "аутсайдер" in f or "outsider" in f:
                return "Outsider"
            elif "фаворит" in f or "favorite" in f:
                return "Favorite"
            elif r["price"] is not None and r["price"] >= 0.5:
                return "Favorite"
            else:
                return "Outsider"

        df["strat"] = df.apply(strat_name, axis=1)
        s_stats = df.groupby("strat").agg(
            trades=("id", "count"),
            win_rate=("is_win", lambda x: round(x.mean()*100, 1)),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_edge=("edge", "mean")
        )
        print("--- STRATEGY STATS ---")
        print(s_stats.to_string())
        print()

        # Актив
        a_stats = df.groupby("asset").agg(
            trades=("id", "count"),
            win_rate=("is_win", lambda x: round(x.mean()*100, 1)),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_edge=("edge", "mean")
        )
        print("--- ASSET STATS ---")
        print(a_stats.to_string())

if __name__ == "__main__":
    asyncio.run(run_analysis())
