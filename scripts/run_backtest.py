"""
Запускает backtest на реальных данных из БД.
Выводит PnL / Sharpe / win-rate / edge-rate в stdout и сохраняет CSV.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbols BTCUSDT --interval 15m
    python scripts/run_backtest.py --min-edge 0.03
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

# Добавляем корень проекта в путь (нужно когда запускаем напрямую, не через -m)
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import polyflip.constants as C
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


async def main(symbols: list[str], interval: str, min_edge: float) -> None:
    # Переопределяем из CLI (константа используется в backtester при импорте)
    C.BACKTEST_MIN_EDGE = min_edge

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL не задан. Задайте переменную окружения.")
        sys.exit(1)

    engine = create_async_engine(db_url)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    results = []
    async with Session() as session:
        for symbol in symbols:
            print(f"\nBacktesting {symbol} {interval}...")
            candles = await get_recent_candles(session, symbol, interval, limit=10_000)
            if len(candles) < 600:
                print(f"  WARN: Мало свечей: {len(candles)} < 600. Пропускаем.")
                print(f"  Подсказка: запустите python scripts/backfill_candles.py --days 90")
                continue

            df = build_features(candles)
            result = run_backtest(df, symbol)
            print(f"  {result.summary()}")
            results.append({
                "symbol":           result.symbol,
                "n_candles_total":  result.n_candles_total,
                "n_candles_test":   result.n_candles_test,
                "n_trades":         result.n_trades,
                "win_rate":         round(result.win_rate, 4),
                "total_return":     round(result.total_return, 5),
                "total_return_net": round(result.total_return_net, 5),
                "sharpe_ratio":     round(result.sharpe_ratio, 3),
                "max_drawdown":     round(result.max_drawdown, 5),
                "edge_rate":        round(result.edge_rate, 4),
                "epsilon":          round(result.epsilon, 6),
                "train_auc":        round(result.train_auc, 4),
            })

    if results:
        df_out = pd.DataFrame(results)
        out_path = Path(__file__).parent / "backtest_results.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nРезультаты сохранены в {out_path}")
        print(df_out.to_string(index=False))

        # Интерпретация
        print("\n--- Интерпретация ---")
        for r in results:
            sharpe = r["sharpe_ratio"]
            win    = r["win_rate"]
            ret    = r["total_return_net"]
            sym    = r["symbol"]
            if sharpe > 0.7 and win > 0.54:
                print(f"  {sym}: РАБОЧИЙ EDGE (Sharpe={sharpe}, WinRate={win:.1%})")
            elif sharpe > 0.3 or win > 0.51:
                print(f"  {sym}: СЛАБЫЙ EDGE (Sharpe={sharpe}, WinRate={win:.1%}) — требует дотюнинга")
            else:
                print(f"  {sym}: НЕТ EDGE (Sharpe={sharpe}, WinRate={win:.1%}) — модель не даёт преимущества")
    else:
        print("\nНет данных для бэктеста")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest LightGBM crypto model")
    parser.add_argument("--symbols",   nargs="+", default=SYMBOLS)
    parser.add_argument("--interval",  default="15m")
    parser.add_argument("--min-edge",  type=float, default=0.04, dest="min_edge")
    args = parser.parse_args()
    asyncio.run(main(args.symbols, args.interval, args.min_edge))
