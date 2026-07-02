"""
Метрики и расчет PnL для бэктеста.
"""
from __future__ import annotations
import pandas as pd
from polyflip.backtesting.simulated_trader import SimulatedTrade
from polyflip.backtesting.market_replay import MarketReplay


def compute_trade_pnl(trade: SimulatedTrade, replay: MarketReplay) -> float:
    """
    Вычисляет профит/убыток одной сделки после разрешения рынка.
    Если рынок INVALID или не завершен, возвращает 0.0.
    """
    if replay.final_outcome not in ("YES", "NO"):
        return 0.0  # Возврат ставки или неизвестно

    # Polymarket: выигрышный токен стоит $1, проигрышный $0
    won = (trade.decision.action == "BUY_YES" and replay.final_outcome == "YES") or \
          (trade.decision.action == "BUY_NO" and replay.final_outcome == "NO")

    if won:
        revenue = 1.0 * trade.shares
        gross_profit = revenue - trade.bet_size
        fee = max(gross_profit, 0) * 0.02
        profit = gross_profit - fee
            
        return profit
    else:
        # Потеряли всю ставку
        return -trade.bet_size


def compute_metrics(trades: list[SimulatedTrade], replays: dict[str, MarketReplay], initial_capital: float = 1000.0) -> dict:
    """Считает общую статистику по бэктесту."""
    if not trades:
        return {"total_trades": 0, "net_profit": 0.0, "roi_pct": 0.0}

    results = []
    for t in trades:
        replay = replays.get(t.market_id)
        if not replay:
            continue
            
        pnl = compute_trade_pnl(t, replay)
        results.append({
            "market_id": t.market_id,
            "asset": t.asset,
            "bet_size": t.bet_size,
            "pnl": pnl,
            "won": pnl > 0,
            "strategy": t.decision.strategy_type
        })

    if not results:
        return {"total_trades": 0, "net_profit": 0.0, "roi_pct": 0.0}

    df = pd.DataFrame(results)
    
    total_trades = len(df)
    total_invested = df["bet_size"].sum()
    net_profit = df["pnl"].sum()
    win_rate = df["won"].mean() * 100
    roi = (net_profit / total_invested * 100) if total_invested > 0 else 0.0

    # Группировка по стратегиям
    strat_stats = {}
    for strat, group in df.groupby("strategy"):
        strat_stats[strat] = {
            "trades": len(group),
            "pnl": float(group["pnl"].sum()),
            "win_rate": float(group["won"].mean() * 100)
        }

    # Equity curve для drawdown
    # В реальном бэктесте мы бы считали PnL по времени, но здесь просто кумулятивная сумма всех сделок
    # Для целей метрики это подойдет, если таблица отсортирована (или хотя бы дает оценку)
    cumulative = df["pnl"].cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    
    # Мы используем переданный initial_capital для расчета max_drawdown_pct
    max_drawdown_pct = float(abs(drawdown.min()) / initial_capital * 100) if initial_capital > 0 else 0.0

    # Sharpe Ratio proxy
    pnl_std = df["pnl"].std()
    avg_pnl = df["pnl"].mean()
    n_trades = len(df)

    import math
    information_ratio = float(avg_pnl / pnl_std) if pd.notna(pnl_std) and pnl_std > 0 else None
    sharpe_annualized_proxy = (
        float(information_ratio * math.sqrt(n_trades))
        if information_ratio is not None else None
    )

    # Profit Factor
    gross_profit = df.loc[df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else None

    return {
        "total_trades": total_trades,
        "total_invested": float(total_invested),
        "net_profit": float(net_profit),
        "roi_pct": float(roi),
        "win_rate_pct": float(win_rate),
        "strategies": strat_stats,
        "max_drawdown_pct": max_drawdown_pct,
        "information_ratio": information_ratio,
        "sharpe_ratio": sharpe_annualized_proxy,
        "sharpe_note": "proxy: IR * sqrt(N), не истинный annualized Sharpe",
        "profit_factor": profit_factor
    }
