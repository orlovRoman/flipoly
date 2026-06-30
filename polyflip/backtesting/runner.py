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
from polyflip.trading.feature_builder import build_feature_vector


class BacktestRunner:
    def __init__(self, config: dict, model_blob: bytes, features: str):
        self.config = config
        self.model = pickle.loads(model_blob) if model_blob else None
        self.features = [f.strip() for f in features.split(',')] if features else []
        self.trader = SimulatedTrader(slippage_pct=float(config.get("SLIPPAGE_PCT", 0.005)))
        self.trade_on_flip = config.get("TRADE_ON_FLIP", "true").lower() == "true"
        self.strategy_mode = config.get("STRATEGY_MODE", "ML")  # ML or PURE_FAVORITE

    def _predict_flip(self, signal) -> float:
        """Получает P(flip) от модели для данного тика."""
        if not self.model or not self.features:
            return 0.0
            
        from polyflip.trading.feature_builder import build_feature_vector, FEATURE_COLUMNS
        X_df = pd.DataFrame(build_feature_vector(signal), columns=FEATURE_COLUMNS)
        
        # Проверяем наличие всех фичей
        missing = [f for f in self.features if f not in X_df.columns]
        if missing:
            return 0.0
            
        X = X_df[self.features]
        proba = self.model.predict_proba(X)[0]
        return proba[1] if len(proba) > 1 else 0.0

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
