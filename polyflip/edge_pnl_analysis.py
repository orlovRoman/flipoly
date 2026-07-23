import asyncio
import numpy as np
import pandas as pd
from polyflip.db.connection import async_session
from polyflip.db.models import TradeHistory
from sqlalchemy import select, and_

async def analyze_edge_vs_pnl():
    async with async_session() as session:
        stmt = select(
            TradeHistory.id,
            TradeHistory.asset,
            TradeHistory.strategy_type,
            TradeHistory.active_features,
            TradeHistory.predicted_flip_prob,
            TradeHistory.executed_price,
            TradeHistory.edge,
            TradeHistory.amount_usdc,
            TradeHistory.pnl,
            TradeHistory.status,
            TradeHistory.created_at
        ).where(
            and_(
                TradeHistory.status == "SUCCESS",
                TradeHistory.pnl.is_not(None)
            )
        )
        
        result = await session.execute(stmt)
        rows = result.all()

        if not rows:
            print("No finished trade records found.")
            return

        df = pd.DataFrame([
            {
                "id": r.id,
                "asset": r.asset,
                "strategy": r.strategy_type or "UNKNOWN",
                "active_features": r.active_features or "",
                "prob": r.predicted_flip_prob,
                "price": r.executed_price,
                "edge": r.edge,
                "amount": r.amount_usdc,
                "pnl": r.pnl,
                "is_win": 1 if r.pnl > 0 else 0,
                "created_at": r.created_at
            }
            for r in rows
        ])

        print(f"=== ОБЩАЯ СТАТИСТИКА (Всего сделок: {len(df)}) ===")
        print(f"Суммарный PnL: ${df['pnl'].sum():.2f}")
        print(f"Общий Win Rate: {df['is_win'].mean()*100:.1f}%")
        print(f"Средний PnL на сделку: ${df['pnl'].mean():.4f}")
        print()

        # Чистим данные с edge для корреляций
        df_edge = df.dropna(subset=["edge"]).copy()
        print(f"Сделок с записанным edge: {len(df_edge)}")

        if len(df_edge) > 0:
            pearson_pnl = df_edge["edge"].corr(df_edge["pnl"], method="pearson")
            spearman_pnl = df_edge["edge"].corr(df_edge["pnl"], method="spearman")
            pearson_win = df_edge["edge"].corr(df_edge["is_win"], method="pearson")
            spearman_win = df_edge["edge"].corr(df_edge["is_win"], method="spearman")

            print(f"--- Корреляции (Edge vs PnL / WinRate) ---")
            print(f"Pearson (Edge vs PnL): {pearson_pnl:.4f}")
            print(f"Spearman (Edge vs PnL): {spearman_pnl:.4f}")
            print(f"Pearson (Edge vs Win): {pearson_win:.4f}")
            print(f"Spearman (Edge vs Win): {spearman_win:.4f}")
            print()

            # Бакетизация по диапазонам Edge
            bins = [-np.inf, 0.0, 0.02, 0.05, 0.10, 0.20, np.inf]
            labels = ["< 0%", "0% - 2%", "2% - 5%", "5% - 10%", "10% - 20%", ">= 20%"]
            df_edge["edge_bucket"] = pd.cut(df_edge["edge"], bins=bins, labels=labels)

            bucket_stats = df_edge.groupby("edge_bucket", observed=False).agg(
                trades=("id", "count"),
                win_rate=("is_win", lambda x: round(x.mean() * 100, 1) if len(x) > 0 else 0),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                avg_amount=("amount", "mean")
            )
            print("--- Анализ по бакетам EDGE ---")
            print(bucket_stats.to_string())
            print()

        # Группировка по активам
        print("--- Анализ по АКТИВАМ ---")
        asset_stats = df.groupby("asset").agg(
            trades=("id", "count"),
            win_rate=("is_win", lambda x: round(x.mean() * 100, 1)),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_edge=("edge", "mean")
        )
        print(asset_stats.to_string())
        print()

        # Разделение по стратегиям (Аутсайдер vs Фаворит)
        def determine_strategy(row):
            feats = str(row["active_features"]).lower()
            if "аутсайдер" in feats or "outsider" in feats:
                return "Outsider (NO)"
            elif "фаворит" in feats or "favorite" in feats:
                return "Favorite (YES)"
            elif row["price"] is not None and row["price"] >= 0.5:
                return "Favorite (YES)"
            else:
                return "Outsider (NO)"

        df["strat_type"] = df.apply(determine_strategy, axis=1)

        print("--- Анализ по СТРАТЕГИЯМ ---")
        strat_stats = df.groupby("strat_type").agg(
            trades=("id", "count"),
            win_rate=("is_win", lambda x: round(x.mean() * 100, 1)),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_edge=("edge", "mean")
        )
        print(strat_stats.to_string())
        print()

        # Диапазоны цен входа (Executed Price)
        price_bins = [0.0, 0.20, 0.40, 0.50, 0.60, 0.80, 1.0]
        price_labels = ["< 0.20", "0.20-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.80", "> 0.80"]
        df["price_bucket"] = pd.cut(df["price"], bins=price_bins, labels=price_labels)

        print("--- Анализ по ЦЕНАМ ВХОДА (Executed Price) ---")
        price_stats = df.groupby("price_bucket", observed=False).agg(
            trades=("id", "count"),
            win_rate=("is_win", lambda x: round(x.mean() * 100, 1)),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean")
        )
        print(price_stats.to_string())

if __name__ == "__main__":
    asyncio.run(analyze_edge_vs_pnl())
