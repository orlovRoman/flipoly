"""
Ядро бэктеста. Объединяет MarketReplay, ML-модель, decision_logic и SimulatedTrader.
"""
from __future__ import annotations
import pickle

from polyflip.backtesting.market_replay import MarketReplay
from polyflip.backtesting.simulated_trader import SimulatedTrader
from polyflip.trading.decision_logic import decide_outsider, TradeDecision
from polyflip.trading.trading_config import parse_trading_settings
from polyflip.trading.ml_inference import (
    build_inference_dataframe,
    run_model_inference,
)


SUPPORTED_BACKTEST_MODES = {"OUTSIDER"}

class BacktestRunner:
    def __init__(self, config: dict, model_blob: bytes, features: str, closed_candles_by_asset: dict | None = None):
        self.config = config
        self.cfg = parse_trading_settings(config)
        self.model = pickle.loads(model_blob) if model_blob and len(model_blob) > 0 else None
        self.features = [f.strip() for f in features.split(',')] if features else []
        self.trader = SimulatedTrader(slippage_pct=float(config.get("SLIPPAGE_PCT", 0.005)))
        self.closed_candles_by_asset = closed_candles_by_asset or {}
        
        _tof = config.get("TRADE_ON_FLIP", False)
        self.trade_on_flip = _tof if isinstance(_tof, bool) else str(_tof).lower() == "true"
        
        self.strategy_mode = config.get("STRATEGY_MODE", "OUTSIDER").upper()
        if self.strategy_mode in ("PURE_FAVORITE", "FAVORITE"):
            raise ValueError("Pure Favorite mode has been removed. Use 'combined' or 'outsider'.")
        if self.strategy_mode not in SUPPORTED_BACKTEST_MODES:
            raise ValueError(
                f"Backtest strategy_mode='{self.strategy_mode}' is not supported. "
                f"Supported modes: {sorted(SUPPORTED_BACKTEST_MODES)}"
            )
        
        self.bet_sizing_mode = config.get("BET_SIZING_MODE", "scaled")
        self.base_bet = float(config.get("TRADE_BET_SIZE_USDC", 5.0))
        self.max_bet = float(config.get("MAX_BET_SIZE_USDC", 50.0))
        self.min_edge = float(config.get("MIN_EDGE", -0.05))
        self.max_edge = 0.40

    def _predict_flip(self, tick, replay: MarketReplay) -> float:
        override = getattr(self, "p_flips", {}).get((tick.market_id, tick.time_left_min))
        if override is not None:
            return float(override)
        """Run the same feature-building contract used by live LogReg inference."""
        if not self.model or not self.features:
            return 0.0


        history = [
            candidate for candidate in replay.ticks
            if candidate.recorded_at < tick.recorded_at
        ]
        observed_prices = [candidate.mid_price for candidate in history] + [tick.mid_price]
        frame = build_inference_dataframe(
            market=tick,
            history_snaps=history,
            fresh_yes_price=tick.mid_price,
            fresh_spread=tick.spread,
            global_max=max(observed_prices),
            start_time=tick.recorded_at,
            time_left_sec=tick.time_left_min * 60.0,
            closed_candles=self.closed_candles_by_asset.get(tick.asset),
        )
        return run_model_inference(frame, self.model, self.features)
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

    def _evaluate_tick(self, tick, replay=None):
        if replay is None:
            replay = MarketReplay.__new__(MarketReplay)
            replay.ticks = [tick]
        signal = tick.to_signal()
        if self.strategy_mode == "OUTSIDER":
            p_flip = self._predict_flip(tick, replay)
            decision = decide_outsider(signal, p_flip, self.cfg, ece=0.0,
                                       time_left_sec=tick.time_left_min * 60.0)
        else:
            raise RuntimeError(f"Unsupported strategy_mode in _evaluate_tick: {self.strategy_mode}")
        
        return decision, p_flip, signal

    def run_market(self, replay: MarketReplay) -> None:
        if not replay.is_tradeable:
            return

        min_time = min(float(self.config.get("FAVOR_MIN_TIME_LEFT_MIN", 1.0)), float(self.config.get("OUTS_MIN_TIME_LEFT_MIN", 1.0)))
        max_time = max(float(self.config.get("FAVOR_MAX_TIME_LEFT_MIN", 60.0)), float(self.config.get("OUTS_MAX_TIME_LEFT_MIN", 60.0)))
        
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
            decision, p_flip, signal = self._evaluate_tick(tick, replay)
            
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
                if best_decision and decision.action != best_decision.action:
                    consecutive_edges = 0  # смена направления: нужно 2 подтверждения с нуля
                    best_decision = decision
                else:
                    consecutive_edges += 1
                    if not best_decision:
                        best_decision = decision
                best_tick, best_p_flip, best_signal = tick, p_flip, signal
                if consecutive_edges >= 2:
                    break

        if entry_strategy == "confirmed" and consecutive_edges < 2:
            best_decision = None

        if best_decision and best_decision.action != "SKIP":
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
        self.p_flips = {}  # Сбрасываем кэш предсказаний перед новым запуском
        if self.strategy_mode == "ML":
            raise ValueError(
                "ML/LGBM backtesting has been removed. "
                "Use strategy_mode='PURE_FAVORITE' or 'COMBINED'."
            )

        for market_id, replay in replays.items():
            self.run_market(replay)
        return self.trader.trades
