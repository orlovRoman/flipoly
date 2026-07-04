#!/usr/bin/env python3
"""
CLI для крипто-бэктеста без Docker.

Примеры:
  python scripts/run_backtest.py --symbol BTCUSDT --interval 15m --days 60
  python scripts/run_backtest.py --symbol ETHUSDT --days 30 --min-edge 0.08
  python scripts/run_backtest.py --symbol BTCUSDT --days 90 \
      --features ret_1 ret_3 vol_ratio rsi_14 taker_buy_ratio
"""
from __future__ import annotations
import argparse, asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polyflip.db.connection import async_session
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features, CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.backtester import run_backtest
from polyflip.crypto.trainer import CRYPTO_FEATURES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Крипто-бэктест из CLI")
    p.add_argument("--symbol",    default="BTCUSDT", choices=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--interval",  default="15m",     choices=["5m", "15m"])
    p.add_argument("--days",      default=60,  type=int)
    p.add_argument("--min-edge",  default=None, type=float)
    p.add_argument("--features",  nargs="+", default=None,
                   help=f"Подмножество фич. Доступные: {CRYPTO_FEATURES}")
    return p.parse_args()


async def main(args: argparse.Namespace) -> None:
    candles_per_day = {"5m": 288, "15m": 96}
    limit = args.days * candles_per_day[args.interval]

    print(f"\n📊 {args.symbol} / {args.interval} / {args.days}d  ({limit} candles)")

    async with async_session() as db:
        candles = await get_recent_candles(db, args.symbol, args.interval, limit=limit)

    if len(candles) < 500:
        print(f"❌ Мало свечей: {len(candles)} < 500")
        print("   Запустите: docker compose exec api python scripts/load_history.py")
        sys.exit(1)

    print(f"✅ Загружено: {len(candles)} свечей")

    # Валидируем фичи если переданы вручную
    if args.features:
        unknown = set(args.features) - set(CRYPTO_FEATURE_COLUMNS)
        if unknown:
            print(f"❌ Неизвестные фичи: {unknown}")
            print(f"   Доступные: {CRYPTO_FEATURE_COLUMNS}")
            sys.exit(1)
        features_to_use = args.features
        print(f"🔧 Эксперимент с фичами: {features_to_use}")
    else:
        features_to_use = None
        print(f"🔧 Стандартные фичи ({len(CRYPTO_FEATURES)}): {CRYPTO_FEATURES}")

    df = build_features(candles)
    result = run_backtest(
        df, symbol=args.symbol,
        min_edge=args.min_edge,
        features=features_to_use,
    )

    print(f"\n{'='*60}")
    print(result.summary())
    print(f"{'='*60}")
    print(f"  Candles total  : {result.n_candles_total}")
    print(f"  Candles test   : {result.n_candles_test}")
    print(f"  Trades         : {result.n_trades}")
    print(f"  Win rate       : {result.win_rate:.1%}")
    print(f"  Return (gross) : {result.total_return:.2%}")
    print(f"  Return (net)   : {result.total_return_net:.2%}")
    print(f"  Sharpe         : {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown   : {result.max_drawdown:.2%}")
    print(f"  Edge rate      : {result.edge_rate:.1%}")
    print(f"  Train AUC      : {result.train_auc:.4f}")
    print(f"  Epsilon        : {result.epsilon:.5f}")
    print(f"  Profitable     : {'✅ YES' if result.is_profitable() else '❌ NO'}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
