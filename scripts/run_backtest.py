#!/usr/bin/env python3
"""
CLI для крипто-бэктеста на локальных данных из БД.

Использование:
    python scripts/run_backtest.py --symbol BTCUSDT --interval 15m --days 60
    python scripts/run_backtest.py --symbol ETHUSDT --interval 5m  --days 30 --min-edge 0.08
    python scripts/run_backtest.py --symbol BTCUSDT --interval 15m --days 90 \
        --features ret_1 ret_3 vol_6 vol_24 rsi_14   # эксперимент с подмножеством фич
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в путь (запуск из корня репозитория)
sys.path.insert(0, str(Path(__file__).parent.parent))

from polyflip.db.connection import async_session
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.trainer import CRYPTO_FEATURES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Крипто-бэктест из командной строки")
    p.add_argument("--symbol",   default="BTCUSDT",   choices=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--interval", default="15m",       choices=["5m", "15m"])
    p.add_argument("--days",     default=60,  type=int, help="Глубина истории (дней)")
    p.add_argument("--min-edge", default=None, type=float, help="Мин. edge для сигнала (0.01–0.49)")
    p.add_argument(
        "--features", nargs="+", default=None,
        help=f"Список фич. По умолчанию: {CRYPTO_FEATURES}"
    )
    p.add_argument("--no-pnl-curve", action="store_true", help="Не печатать кривую PnL")
    return p.parse_args()


async def main(args: argparse.Namespace) -> None:
    candles_per_day = {"5m": 288, "15m": 96}
    limit = args.days * candles_per_day[args.interval]

    print(f"\n📊 Запуск бэктеста: {args.symbol} / {args.interval} / {args.days} дней")
    print(f"   Загружаем до {limit} свечей из БД...")

    async with async_session() as db:
        candles = await get_recent_candles(db, args.symbol, args.interval, limit=limit)

    if len(candles) < 500:
        print(f"❌ Недостаточно свечей: {len(candles)} < 500")
        print("   Запустите: python scripts/load_history.py")
        sys.exit(1)

    print(f"   Загружено: {len(candles)} свечей")

    df = build_features(candles)

    # Переопределяем фичи для эксперимента
    features_to_use = args.features or CRYPTO_FEATURES
    if args.features:
        unknown = set(args.features) - set(df.columns)
        if unknown:
            print(f"❌ Неизвестные фичи: {unknown}")
            print(f"   Доступные: {sorted(df.columns.tolist())}")
            sys.exit(1)
        print(f"   🧪 Эксперимент: используем {len(features_to_use)} фич вместо {len(CRYPTO_FEATURES)}")
        # Ограничиваем df только нужными фичами + служебные колонки
        service_cols = [c for c in df.columns if c not in CRYPTO_FEATURES]
        df = df[service_cols + features_to_use]

    print("   Обучаем модель и считаем метрики...\n")

    result = run_backtest(df, args.symbol, min_edge=args.min_edge, features=args.features)

    # Вывод результата
    print("=" * 60)
    print(result.summary())
    print("=" * 60)
    print(f"  Всего свечей:     {result.n_candles_total}")
    print(f"  Тест-период:      {result.n_candles_test} свечей")
    print(f"  Сделок:           {result.n_trades}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    print(f"  Return (брутто):  {result.total_return:.2%}")
    print(f"  Return (нетто):   {result.total_return_net:.2%}")
    print(f"  Sharpe:           {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:     {result.max_drawdown:.2%}")
    print(f"  Edge Rate:        {result.edge_rate:.1%}")
    print(f"  Epsilon:          {result.epsilon:.5f}")
    print(f"  Train AUC:        {result.train_auc:.4f}")
    print(f"  Profitable:       {'✅ ДА' if result.is_profitable() else '❌ НЕТ'}")

    if not args.no_pnl_curve and result.pnl_curve:
        print(f"\n  PnL curve ({len(result.pnl_curve)} точек):")
        for pt in result.pnl_curve[-5:]:   # последние 5 точек
            print(f"    {pt['time']}  →  {pt['pnl']:+.2f}%")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
