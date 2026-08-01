# polyflip/api/crypto_backtest_api.py
"""
API для крипто-бэктестера (LightGBM на OHLCV-свечах).
Отдельный роутер — не смешиваем с prediction-рынками.

PnL-режимы:
  - binance    (default) — лог-доход следующей свечи Binance
  - polymarket — бинарная PnL-логика с реальными ценами MarketSnapshot
"""
from __future__ import annotations

import asyncio
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from polyflip.db.connection import get_db_session
from polyflip.db.models import CryptoCandle
from polyflip.api.auth import verify_api_key
from polyflip.crypto.candle_repository import get_recent_candles
from polyflip.crypto.feature_builder import build_features
from polyflip.crypto.backtester import run_backtest, BacktestResult
from polyflip.crypto.polymarket_join import join_polymarket_prices

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/crypto/backtest", tags=["Crypto Backtest"])


class CryptoBacktestRequest(BaseModel):
    symbol:   Literal["BTCUSDT", "ETHUSDT"] = "BTCUSDT"
    interval: Literal["5m", "15m"]           = "15m"
    days:     int = Field(default=60, ge=7, le=180)
    min_edge: float | None = Field(default=None, ge=0.01, le=0.49)
    features: list[str] | None = None
    # Новые поля для Polymarket-режима
    pnl_mode: Literal["binance", "polymarket"] = "binance"
    asset:    Literal["BTC", "ETH"] = "BTC"


@router.post("/run", dependencies=[Depends(verify_api_key)])
async def run_crypto_backtest(
    req: CryptoBacktestRequest,
    db:  AsyncSession = Depends(get_db_session),
):
    """
    Запускает крипто-бэктест на исторических свечах из БД.

    pnl_mode="binance"    — legacy, симулирует Binance-торговлю по лог-доходам свечей.
    pnl_mode="polymarket" — бинарная логика с реальными ценами Polymarket из market_snapshots.
                            Требует накопленных снапшотов; при coverage < 30% возвращает
                            предупреждение в поле "coverage_warning".
    """
    candles_per_day = {"5m": 288, "15m": 96}
    limit = req.days * candles_per_day[req.interval]

    candles = await get_recent_candles(db, req.symbol, req.interval, limit=limit)
    if len(candles) < 500:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough candles: {len(candles)} < 500. Run historical_loader first."
        )

    df = build_features(candles)

    features_to_use = None
    if req.features:
        from polyflip.crypto.trainer import CRYPTO_FEATURES as DEFAULT_FEATURES
        unknown = set(req.features) - set(df.columns)
        if unknown:
            raise HTTPException(422, detail=f"Unknown features: {unknown}")
        features_to_use = req.features

    # ── Polymarket join (только если нужен режим polymarket) ──
    polymarket_prices = None
    if req.pnl_mode == "polymarket":
        polymarket_prices = await join_polymarket_prices(
            session=db,
            df_candles=df,
            asset=req.asset,
        )

    result: BacktestResult = await asyncio.to_thread(
        run_backtest,
        df,
        req.symbol,
        min_edge=req.min_edge,
        features=features_to_use,
        pnl_mode=req.pnl_mode,
        polymarket_prices=polymarket_prices,
    )

    coverage_warning = None
    if req.pnl_mode == "polymarket" and result.coverage_pct < 30.0:
        coverage_warning = (
            f"Low Polymarket coverage: {result.coverage_pct}% of candles matched a snapshot. "
            "Collect more market_snapshots data (>30 days) for reliable results."
        )

    response = {
        "symbol":              result.symbol,
        "interval":            req.interval,
        "days_requested":      req.days,
        "pnl_mode":            result.pnl_mode,
        "n_candles":           result.n_candles_total,
        "n_candles_test":      result.n_candles_test,
        "n_trades":            result.n_trades,
        "win_rate":            round(result.win_rate, 4),
        "total_return":        round(result.total_return, 4),
        "total_return_net":    round(result.total_return_net, 4),
        "sharpe_ratio":        round(result.sharpe_ratio, 3),
        "max_drawdown":        round(result.max_drawdown, 4),
        "edge_rate":           round(result.edge_rate, 4),
        "epsilon":             round(result.epsilon, 6),
        "train_auc":           round(result.train_auc, 4),
        "is_profitable":       result.is_profitable(),
        "summary":             result.summary(),
        "pnl_curve":           result.pnl_curve,
    }

    # Дополнительные поля только для polymarket-режима
    if req.pnl_mode == "polymarket":
        response["n_polymarket_matched"] = result.n_polymarket_matched
        response["avg_buy_price"]        = round(result.avg_buy_price, 4) if result.avg_buy_price else None
        response["coverage_pct"]         = result.coverage_pct
        if coverage_warning:
            response["coverage_warning"] = coverage_warning

    return response


@router.get("/candle-stats", dependencies=[Depends(verify_api_key)])
async def get_candle_stats(db: AsyncSession = Depends(get_db_session)):
    """
    Статистика по свечам в БД.
    Показывает: сколько свечей есть, какой период покрыт, хватает ли для обучения.
    """
    rows = (await db.execute(
        select(
            CryptoCandle.symbol,
            CryptoCandle.interval,
            func.count().label("count"),
            func.min(CryptoCandle.open_time).label("oldest"),
            func.max(CryptoCandle.open_time).label("newest"),
        ).group_by(CryptoCandle.symbol, CryptoCandle.interval)
        .order_by(CryptoCandle.symbol, CryptoCandle.interval)
    )).all()

    return {
        "candles": [
            {
                "symbol":   r.symbol,
                "interval": r.interval,
                "count":    r.count,
                "days_covered": round(
                    (r.newest - r.oldest).total_seconds() / 86400, 1
                ) if r.oldest and r.newest else 0,
                "oldest":   r.oldest.isoformat() if r.oldest else None,
                "newest":   r.newest.isoformat() if r.newest else None,
                "ready_for_training": r.count >= 500,
            }
            for r in rows
        ]
    }
