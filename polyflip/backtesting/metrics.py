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
        # PnL = (1.0 - executed_price) * shares
        revenue = 1.0 * trade.shares
        profit = revenue - trade.bet_size
        
        # Комиссия Polymarket (например 2% с профита, если в плюсе)
        # Упрощенно: если profit > 0, вычитаем 2%
        if profit > 0:
            profit *= 0.98
            
        return profit
    else:
        # Потеряли всю ставку
        return -trade.bet_size


def compute_metrics(trades: list[SimulatedTrade], replays: dict[str, MarketReplay]) -> dict:
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

    return {
        "total_trades": total_trades,
        "total_invested": float(total_invested),
        "net_profit": float(net_profit),
        "roi_pct": float(roi),
        "win_rate_pct": float(win_rate),
        "strategies": strat_stats
    }
