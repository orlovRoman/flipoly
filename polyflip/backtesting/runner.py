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
        
        self.bet_sizing_mode = config.get("BET_SIZING_MODE", "scaled")
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

    def _calc_bet_size(self, decision, signal=None) -> float:
        """Скейлинг ставки по edge с учётом ликвидности."""
        if self.bet_sizing_mode != "scaled":
            bet = self.base_bet
        else:
            edge = getattr(decision, "edge", None)
            if edge is None or self.max_edge <= self.min_edge:
                bet = self.base_bet
            else:
                t = (edge - self.min_edge) / (self.max_edge - self.min_edge)
                t = max(0.0, min(1.0, t))
                bet = self.base_bet + t * (self.max_bet - self.base_bet)
        
        # Применяем liquidity cap если есть signal
        if signal is not None and signal.volume_5min > 0:
            liquidity_fraction = float(self.config.get("LIQUIDITY_FRACTION", 0.05))
            cap = max(signal.volume_5min * liquidity_fraction, self.base_bet)
            bet = min(bet, cap)
        
        return round(bet, 2)

    def _evaluate_tick(self, tick):
        signal = tick.to_signal()
        if self.strategy_mode == "PURE_FAVORITE":
            from polyflip.trading.decision_logic import decide_favorite
            decision = decide_favorite(signal, self.config)
            p_flip = 0.0
        else:
            p_flip = self._predict_flip(signal)
            from polyflip.trading.decision_logic import decide_ml_trend, decide_outsider
            decision = decide_ml_trend(signal, p_flip, self.config)
            if decision.action == "SKIP" and self.trade_on_flip:
                decision = decide_outsider(signal, p_flip, self.config)
        return decision, p_flip, signal

    def run_market(self, replay: MarketReplay) -> None:
        if not replay.is_tradeable:
            return

        min_time = float(self.config.get("MIN_TIME_LEFT_MIN", 1.0))
        max_time = float(self.config.get("MAX_TIME_LEFT_MIN", 60.0))
        
        ticks = replay.get_ticks_in_window(min_time, max_time)
        if not ticks:
            return

        entry_strategy = self.config.get("ENTRY_STRATEGY", "first")
        
        best_decision = None
        best_tick = None
        best_p_flip = 0.0
        best_signal = None
        consecutive_edges = 0
        
        for tick in ticks:
            decision, p_flip, signal = self._evaluate_tick(tick)
            
            if decision.action == "SKIP":
                consecutive_edges = 0
                continue
                
            if entry_strategy == "first":
                best_decision, best_tick, best_p_flip, best_signal = decision, tick, p_flip, signal
                break
            elif entry_strategy == "best_edge":
                if not best_decision or (decision.edge or 0) > (best_decision.edge or 0):
                    best_decision, best_tick, best_p_flip, best_signal = decision, tick, p_flip, signal
            elif entry_strategy == "confirmed":
                consecutive_edges += 1
                if consecutive_edges >= 2:
                    best_decision, best_tick, best_p_flip, best_signal = decision, tick, p_flip, signal
                    break

        if best_decision and best_decision.action != "SKIP":
            from polyflip.trading.decision_logic import TradeDecision
            bet = self._calc_bet_size(best_decision, signal=best_signal)
            decision = TradeDecision(
                action=best_decision.action,
                buy_price=best_decision.buy_price,
                bet_size_usdc=bet,
                reason=best_decision.reason,
                strategy_type=best_decision.strategy_type,
                p_flip=best_p_flip,
                edge=best_decision.edge,
            )
            self.trader.execute_trade(
                market_id=replay.market_id,
                asset=replay.asset,
                decision=decision,
                timestamp=best_tick.recorded_at,
                p_flip=best_p_flip
            )

    def run_all(self, replays: dict[str, MarketReplay]) -> list:
        for market_id, replay in replays.items():
            self.run_market(replay)
        return self.trader.trades
