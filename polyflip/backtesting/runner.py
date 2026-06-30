"""
Ядро бэктеста. Объединяет MarketReplay, ML-модель, decision_logic и SimulatedTrader.
"""
from __future__ import annotations
import pickle
import pandas as pd
from typing import Any

from polyflip.backtesting.market_replay import MarketReplay
from polyflip.backtesting.simulated_trader import SimulatedTrader
from polyflip.trading.decision_logic import decide_favorite, decide_ml_trend, decide_outsider
from polyflip.trading.feature_builder import build_feature_vector, FEATURE_COLUMNS


class BacktestRunner:
    def __init__(self, config: dict, model_blob: bytes, features: str):
        self.config = config
        self.model = pickle.loads(model_blob) if model_blob and len(model_blob) > 0 else None
        self.features = [f.strip() for f in features.split(',')] if features else []
        self.trader = SimulatedTrader(slippage_pct=float(config.get("SLIPPAGE_PCT", 0.005)))
        
        _tof = config.get("TRADE_ON_FLIP", False)
        self.trade_on_flip = _tof if isinstance(_tof, bool) else str(_tof).lower() == "true"
        
        self.strategy_mode = config.get("STRATEGY_MODE", "ML")  # ML or PURE_FAVORITE
        
        self.bet_sizing_mode = config.get("BET_SIZING_MODE", "fixed")
        self.base_bet = float(config.get("TRADE_BET_SIZE_USDC", 5.0))
        self.max_bet = float(config.get("MAX_BET_SIZE_USDC", 50.0))
        self.min_edge = float(config.get("MIN_EDGE", -0.05))
        self.max_edge = float(config.get("MAX_EDGE", 0.50))

    def _predict_flip(self, signal) -> float:
        """Получает P(flip) от модели для данного тика."""
        if not self.model or not self.features:
            return 0.0
            
        X_df = pd.DataFrame(build_feature_vector(signal), columns=FEATURE_COLUMNS)
        
        # Добавляем производные признаки, если они требуются модели
        import numpy as np
        X_df["price_deviation"]     = (X_df["mid_price"] - 0.5).abs()
        X_df["deviation_x_time"]    = X_df["price_deviation"] * X_df["time_left_min"]
        X_df["price_deviation_sq"]  = X_df["price_deviation"] ** 2
        X_df["spread_pct"]          = (X_df["spread"] / (X_df["mid_price"] + 1e-6)).clip(upper=10.0)
        X_df["log_time_left"]       = np.log1p(X_df["time_left_min"])
        
        # Проверяем наличие всех фичей
        missing = [f for f in self.features if f not in X_df.columns]
        if missing:
            return 0.0
            
        X = X_df[self.features]
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.0

    def _calc_bet_size(self, decision) -> float:
        """Скейлинг ставки по edge в диапазоне [base_bet, max_bet]."""
        if self.bet_sizing_mode != "scaled":
            return self.base_bet
        edge = getattr(decision, "edge", None)
        if edge is None or self.max_edge <= self.min_edge:
            return self.base_bet
        # Линейный скейлинг: edge → [0, 1] → [base, max]
        t = (edge - self.min_edge) / (self.max_edge - self.min_edge)
        t = max(0.0, min(1.0, t))  # clip
        return self.base_bet + t * (self.max_bet - self.base_bet)

    def run_market(self, replay: MarketReplay) -> None:
        """
        Прогоняет один рынок через симуляцию.
        Ищет первую возможность для входа.
        """
        if not replay.is_tradeable:
            return

        min_time = float(self.config.get("MIN_TIME_LEFT_MIN", 1.0))
        max_time = float(self.config.get("MAX_TIME_LEFT_MIN", 60.0))
        
        # Берем самый ранний тик в торговом окне
        tick = replay.get_entry_tick(min_time, max_time)
        if not tick:
            return

        signal = tick.to_signal()

        if self.strategy_mode == "PURE_FAVORITE":
            decision = decide_favorite(signal, self.config)
            p_flip = 0.0
        else:
            p_flip = self._predict_flip(signal)
            decision = decide_ml_trend(signal, p_flip, self.config)
            
            if decision.action == "SKIP" and self.trade_on_flip:
                decision = decide_outsider(signal, p_flip, self.config)

        if decision.action != "SKIP":
            bet = self._calc_bet_size(decision)
            # Переопределяем ставку в decision перед передачей трейдеру
            from polyflip.trading.decision_logic import TradeDecision
            decision = TradeDecision(
                action=decision.action,
                buy_price=decision.buy_price,
                bet_size_usdc=bet,
                reason=decision.reason,
                strategy_type=decision.strategy_type,
                p_flip=decision.p_flip,
                edge=decision.edge,
            )
            self.trader.execute_trade(
                market_id=replay.market_id,
                asset=replay.asset,
                decision=decision,
                timestamp=tick.recorded_at,
                p_flip=p_flip
            )

    def run_all(self, replays: dict[str, MarketReplay]) -> list:
        for market_id, replay in replays.items():
            self.run_market(replay)
        return self.trader.trades
