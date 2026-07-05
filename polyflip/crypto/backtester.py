"""
Walk-forward backtester для крипто-модели LightGBM.

Метрики:
  - total_return     — суммарный лог-доход за период
  - sharpe_ratio     — аннуализированный Sharpe (252 * 96 периодов/год)
  - win_rate         — доля прибыльных сделок
  - n_trades         — количество сделок (когда edge >= MIN_EDGE)
  - edge_rate        — доля свечей, на которых модель даёт сигнал
  - max_drawdown     — максимальная просадка (пик→дно) на кумулятивном PnL

PnL-режимы:
  - "binance"    (legacy) — pnl = direction × ret_next (лог-доход свечи)
  - "polymarket" (новый)  — бинарная логика:
      угадал → pnl = (1 - buy_price) / buy_price
      не угадал → pnl = -1.0
      комиссия: POLYMARKET_FEE_RATE / buy_price

Walk-forward: обучение на первых BACKTEST_TRAIN_RATIO данных,
тест на оставшихся. Нет data leakage.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from polyflip.constants import (
    BACKTEST_COMMISSION,
    BACKTEST_MIN_EDGE,
    BACKTEST_SHARPE_ANNUALIZE,
    BACKTEST_TRAIN_RATIO,
    CANDLE_EPSILON_QUANTILE,
    CV_N_SPLITS,
    POLYMARKET_FEE_RATE,
)
from polyflip.crypto.trainer import (
    CRYPTO_FEATURES,
    _build_target,
    _fit_lgbm_and_serialize,
)


@dataclass
class BacktestResult:
    symbol:           str
    n_candles_total:  int
    n_candles_test:   int
    n_trades:         int
    win_rate:         float
    total_return:     float        # лог-доход без комиссии
    total_return_net: float        # лог-доход / бинарный PnL с комиссией
    sharpe_ratio:     float
    max_drawdown:     float
    edge_rate:        float        # доля свечей с сигналом
    epsilon:          float
    train_auc:        float
    pnl_mode:         str = "binance"       # "binance" | "polymarket"
    n_polymarket_matched: int = 0           # сделок с ценой Polymarket
    avg_buy_price:    float | None = None   # средняя цена входа YES/NO
    coverage_pct:     float = 0.0           # % свечей с совпавшим снапшотом
    pnl_curve:        list[dict] = field(default_factory=list)

    def is_profitable(self, min_sharpe: float = 0.5) -> bool:
        return self.sharpe_ratio >= min_sharpe and self.win_rate >= 0.52

    def summary(self) -> str:
        sign = "[OK]" if self.is_profitable() else "[--]"
        mode_tag = f"[{self.pnl_mode}]"
        return (
            f"{sign}{mode_tag} {self.symbol} | "
            f"Trades={self.n_trades} | "
            f"WinRate={self.win_rate:.1%} | "
            f"Return(net)={self.total_return_net:.2%} | "
            f"Sharpe={self.sharpe_ratio:.2f} | "
            f"MaxDD={self.max_drawdown:.2%} | "
            f"EdgeRate={self.edge_rate:.1%} | "
            f"TrainAUC={self.train_auc:.3f}"
        )


def _empty_result(
    symbol: str,
    n_candles_total: int,
    n_candles_test: int,
    epsilon: float,
    pnl_mode: str = "binance",
) -> BacktestResult:
    return BacktestResult(
        symbol=symbol,
        n_candles_total=n_candles_total,
        n_candles_test=n_candles_test,
        n_trades=0,
        win_rate=0.0,
        total_return=0.0,
        total_return_net=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        edge_rate=0.0,
        epsilon=epsilon,
        train_auc=0.0,
        pnl_mode=pnl_mode,
        pnl_curve=[],
    )


def run_backtest(
    df_features: pd.DataFrame,
    symbol: str,
    min_edge: float | None = None,
    commission: float | None = None,
    features: list[str] | None = None,
    epsilon_quantile: float | None = None,
    lgbm_params: dict | None = None,
    pnl_mode: Literal["binance", "polymarket"] = "binance",
    polymarket_prices: pd.DataFrame | None = None,  # output of join_polymarket_prices()
) -> BacktestResult:
    """
    Принимает DataFrame с фичами (выход build_features()).
    Сам разбивает train/test по времени, обучает модель на train,
    считает метрики на test.

    df_features должен быть отсортирован по времени (open_time ASC).

    pnl_mode="polymarket" требует polymarket_prices (из join_polymarket_prices).
    При недостаточном покрытии (<30% свечей имеют снапшот) возвращает предупреждение
    через BacktestResult.coverage_pct.
    """
    n_total = len(df_features)
    n_train = int(n_total * BACKTEST_TRAIN_RATIO)

    df_train_raw = df_features.iloc[:n_train].copy()
    df_test_raw  = df_features.iloc[n_train:].copy()

    # Epsilon вычисляем только из train — без leakage
    _eps_q = epsilon_quantile if epsilon_quantile is not None else CANDLE_EPSILON_QUANTILE
    epsilon = float(df_train_raw["ret_1"].abs().quantile(_eps_q))

    df_train = _build_target(df_train_raw, epsilon)
    df_test  = _build_target(df_test_raw, epsilon)

    feature_list = features if features is not None else CRYPTO_FEATURES
    available = [f for f in feature_list if f in df_train.columns]

    if len(df_train) < 300 or len(available) == 0:
        return _empty_result(symbol, n_total, len(df_test), epsilon, pnl_mode)

    # ── Vol-режим: обучаем 2 модели ──────────────────────────────
    vol_median = float(df_train["vol_ratio"].median())

    models: dict[str, Any] = {}
    train_aucs: list[float] = []

    for regime, mask in [
        ("low_vol",  df_train["vol_ratio"] <= vol_median),
        ("high_vol", df_train["vol_ratio"] >  vol_median),
    ]:
        df_r = df_train[mask]
        if len(df_r) < 150:
            continue
        n_splits = min(CV_N_SPLITS, 3)
        _lgbm = lgbm_params or {}
        model_bytes, auc, *_ = _fit_lgbm_and_serialize(
            df_r[available], df_r["target"], n_splits=n_splits, **_lgbm
        )
        models[regime] = pickle.loads(model_bytes)
        train_aucs.append(auc)

    if not models:
        return _empty_result(symbol, n_total, len(df_test), epsilon, pnl_mode)

    train_auc = float(np.mean(train_aucs))

    # ── Инференс: выбираем модель по режиму каждой свечи ─────────
    X_test = df_test[available]
    probas = np.full(len(df_test), 0.5)

    low_mask  = df_test["vol_ratio"] <= vol_median
    high_mask = ~low_mask

    if "low_vol" in models and low_mask.any():
        probas[low_mask.values] = models["low_vol"].predict_proba(X_test[low_mask])[:, 1]

    if "high_vol" in models:
        if high_mask.any():
            probas[high_mask.values] = models["high_vol"].predict_proba(X_test[high_mask])[:, 1]
    elif "low_vol" in models and high_mask.any():
        probas[high_mask.values] = models["low_vol"].predict_proba(X_test[high_mask])[:, 1]

    import logging
    if "low_vol" not in models:
        logging.warning(f"[backtest:{symbol}] no low_vol model, fallback 0.5")
    if "high_vol" not in models:
        logging.warning(f"[backtest:{symbol}] no high_vol model, fallback")

    _min_edge   = min_edge if min_edge is not None else BACKTEST_MIN_EDGE
    _commission = commission if commission is not None else BACKTEST_COMMISSION

    df_test = df_test.copy()
    df_test["prob_up"] = probas
    df_test["edge"]    = probas - 0.5
    df_test["signal"]  = df_test["edge"].abs() >= _min_edge
    df_test["ret_next"] = df_test["ret_1"].shift(-1)
    df_test = df_test.dropna(subset=["ret_next"])

    trades = df_test[df_test["signal"]].copy()

    if len(trades) == 0:
        return BacktestResult(
            symbol=symbol,
            n_candles_total=n_total,
            n_candles_test=len(df_test),
            n_trades=0,
            win_rate=0.0,
            total_return=0.0,
            total_return_net=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            edge_rate=float(df_test["signal"].mean()),
            epsilon=epsilon,
            train_auc=train_auc,
            pnl_mode=pnl_mode,
            pnl_curve=[],
        )

    trades = trades.copy()
    trades["direction"] = np.where(trades["edge"] > 0, 1, -1)

    # ══════════════════════════════════════════════════
    # PnL-РЕЖИМ: POLYMARKET (бинарная логика)
    # ══════════════════════════════════════════════════
    if pnl_mode == "polymarket" and polymarket_prices is not None:
        # Нормализуем open_time для join
        pm = polymarket_prices.copy()
        pm["open_time"] = pd.to_datetime(pm["open_time"], utc=True)
        pm = pm[["open_time", "pm_yes_price", "pm_outcome", "pm_market_id"]].dropna(subset=["pm_yes_price", "pm_outcome"])
        pm = pm[pm["pm_outcome"].isin(["YES", "NO"])]  # INVALID уже отфильтрован в join, но на всякий случай
        pm = pm.sort_values("open_time").reset_index(drop=True)

        trades["open_time"] = pd.to_datetime(trades["open_time"], utc=True)
        trades = trades.sort_values("open_time").reset_index(drop=True)

        # Объединяем сделки с ценами Polymarket
        tolerance = pd.Timedelta(seconds=450)
        trades = pd.merge_asof(
            trades,
            pm,
            on="open_time",
            direction="nearest",
            tolerance=tolerance,
        )

        # Отфильтровываем сделки без совпадения (нет снапшота)
        matched = trades.dropna(subset=["pm_yes_price", "pm_outcome"]).copy()
        n_matched   = len(matched)
        coverage    = round(n_matched / len(df_test) * 100, 1) if len(df_test) > 0 else 0.0
        avg_buy_p   = float(matched["pm_yes_price"].mean()) if n_matched > 0 else None

        if n_matched == 0:
            return BacktestResult(
                symbol=symbol,
                n_candles_total=n_total,
                n_candles_test=len(df_test),
                n_trades=0,
                win_rate=0.0,
                total_return=0.0,
                total_return_net=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                edge_rate=float(df_test["signal"].mean()),
                epsilon=epsilon,
                train_auc=train_auc,
                pnl_mode=pnl_mode,
                n_polymarket_matched=0,
                avg_buy_price=None,
                coverage_pct=coverage,
                pnl_curve=[],
            )

        # direction=1 → покупаем YES, direction=-1 → покупаем NO
        buy_price = np.where(
            matched["direction"] == 1,
            matched["pm_yes_price"],
            1.0 - matched["pm_yes_price"],
        )
        # Защита от граничных значений (0.01..0.99)
        buy_price = np.clip(buy_price, 0.01, 0.99)

        # Угадали ли направление
        won = np.where(
            matched["direction"] == 1,
            matched["pm_outcome"] == "YES",
            matched["pm_outcome"] == "NO",
        )

        # Бинарный PnL: выиграл / проиграл всю ставку
        pnl_raw = np.where(won, (1.0 - buy_price) / buy_price, -1.0)

        # Комиссия Polymarket (2% от выплаты): fee / buy_price
        fee_per_trade = POLYMARKET_FEE_RATE / buy_price
        pnl_net_arr   = pnl_raw - fee_per_trade

        matched = matched.copy()
        matched["pnl"]     = pnl_raw
        matched["pnl_net"] = pnl_net_arr
        win_rate = float((matched["pnl"] > 0).mean())

        total_return = float(matched["pnl"].sum())
        total_net    = float(matched["pnl_net"].sum())

        pnl_std = matched["pnl_net"].std()
        if pnl_std > 0:
            trade_fraction = len(matched) / len(df_test)
            ann_factor = trade_fraction * BACKTEST_SHARPE_ANNUALIZE
            sharpe = float(
                matched["pnl_net"].mean() / pnl_std * np.sqrt(ann_factor)
            )
        else:
            sharpe = 0.0

        cum_pnl  = matched["pnl_net"].cumsum()
        roll_max = cum_pnl.cummax()
        max_dd   = float((cum_pnl - roll_max).min())
        edge_rate = float(df_test["signal"].mean())

        pnl_curve = _build_pnl_curve(matched, cum_pnl)

        return BacktestResult(
            symbol=symbol,
            n_candles_total=n_total,
            n_candles_test=len(df_test),
            n_trades=n_matched,
            win_rate=win_rate,
            total_return=total_return,
            total_return_net=total_net,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            edge_rate=edge_rate,
            epsilon=epsilon,
            train_auc=train_auc,
            pnl_mode=pnl_mode,
            n_polymarket_matched=n_matched,
            avg_buy_price=avg_buy_p,
            coverage_pct=coverage,
            pnl_curve=pnl_curve,
        )

    # ══════════════════════════════════════════════════
    # PnL-РЕЖИМ: BINANCE (legacy — лог-доход свечи)
    # ══════════════════════════════════════════════════
    trades["pnl"]     = trades["direction"] * trades["ret_next"]
    trades["pnl_net"] = trades["pnl"] - _commission * 2

    significant_trades = trades[trades["ret_next"].abs() >= epsilon]
    if len(significant_trades) > 0:
        win_rate = float((significant_trades["pnl"] > 0).mean())
    else:
        win_rate = float((trades["pnl"] > 0).mean())

    total_return = float(trades["pnl"].sum())
    total_net    = float(trades["pnl_net"].sum())

    pnl_std = trades["pnl_net"].std()
    if pnl_std > 0:
        trade_fraction = len(trades) / len(df_test)
        ann_factor = trade_fraction * BACKTEST_SHARPE_ANNUALIZE
        sharpe = float(
            trades["pnl_net"].mean() / pnl_std * np.sqrt(ann_factor)
        )
    else:
        sharpe = 0.0

    cum_pnl  = trades["pnl_net"].cumsum()
    roll_max = cum_pnl.cummax()
    max_dd   = float((cum_pnl - roll_max).min())
    edge_rate = float(df_test["signal"].mean())
    pnl_curve = _build_pnl_curve(trades, cum_pnl)

    return BacktestResult(
        symbol=symbol,
        n_candles_total=n_total,
        n_candles_test=len(df_test),
        n_trades=len(trades),
        win_rate=win_rate,
        total_return=total_return,
        total_return_net=total_net,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        edge_rate=edge_rate,
        epsilon=epsilon,
        train_auc=train_auc,
        pnl_mode="binance",
        pnl_curve=pnl_curve,
    )


def _build_pnl_curve(trades: pd.DataFrame, cum_pnl: pd.Series) -> list[dict]:
    """Строит кривую PnL для графика (max 100 точек)."""
    pnl_curve = []
    if len(trades) == 0:
        return pnl_curve
    step = max(1, len(trades) // 100)
    for i in range(0, len(trades), step):
        t_str = _format_time(trades, i)
        pnl_curve.append({"time": t_str, "pnl": round(float(cum_pnl.iloc[i]) * 100, 2)})
    if len(trades) % step != 0:
        t_str = _format_time(trades, -1)
        pnl_curve.append({"time": t_str, "pnl": round(float(cum_pnl.iloc[-1]) * 100, 2)})
    return pnl_curve


def _format_time(trades: pd.DataFrame, i: int) -> str:
    if "open_time" in trades.columns:
        t_val = trades["open_time"].iloc[i]
        return t_val.isoformat() if hasattr(t_val, "isoformat") else str(t_val)
    return str(trades.index[i])
