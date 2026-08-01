"""
Симуляция исполнения сделок (простейшая модель).
Принимает решения (TradeDecision) от decision_logic и записывает "как бы" исполнение.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

from polyflip.trading.decision_logic import TradeDecision


@dataclass
class SimulatedTrade:
    market_id: str
    asset: str
    decision: TradeDecision
    executed_price: float
    slippage: float
    bet_size: float
    shares: float
    timestamp: datetime
    p_flip: float


class SimulatedTrader:
    """
    Ведёт учёт "бумажных" сделок в памяти.
    Симулирует проскальзывание (slippage).
    """

    def __init__(self, slippage_pct: float = 0.005) -> None:
        self.slippage_pct = slippage_pct  # 0.5% по умолчанию
        self.trades: list[SimulatedTrade] = []

    def execute_trade(
        self,
        market_id: str,
        asset: str,
        decision: TradeDecision,
        timestamp: datetime,
        p_flip: float,
    ) -> SimulatedTrade:
        
        # Симулируем исполнение по цене + slippage
        # В реальности на Polymarket slippage может двигать цену вверх
        expected_price = decision.buy_price
        
        # Увеличиваем цену на % проскальзывания, но не выше 0.99
        executed_price = min(expected_price * (1.0 + self.slippage_pct), 0.99)
        slippage_abs = executed_price - expected_price
        
        shares = decision.bet_size_usdc / executed_price if executed_price > 0 else 0.0
        
        trade = SimulatedTrade(
            market_id=market_id,
            asset=asset,
            decision=decision,
            executed_price=executed_price,
            slippage=slippage_abs,
            bet_size=decision.bet_size_usdc,
            shares=shares,
            timestamp=timestamp,
            p_flip=p_flip
        )
        self.trades.append(trade)
        return trade

    @property
    def total_invested(self) -> float:
        return sum(t.bet_size for t in self.trades)
