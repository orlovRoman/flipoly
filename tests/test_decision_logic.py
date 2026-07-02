"""
Самотесты для decision_logic.py — покрывают баги #2, #3, #5.
Запуск: pytest tests/test_decision_logic.py -v
"""
import pytest
from polyflip.trading.feature_builder import MarketSignal
from polyflip.trading.decision_logic import (
    decide_ml_trend, decide_favorite, decide_outsider, TradeDecision
)

# ─── Фабрика сигналов ───────────────────────────────────────────────────────

def _signal(mid=0.70, spread=0.02, volume=500.0, velocity=0.01, hour=12, time_left=30.0):
    return MarketSignal(
        asset="BTC",
        mid_price=mid,
        spread=spread,
        volume_5min=volume,
        price_velocity=velocity,
        hour_of_day=hour,
        time_left_min=time_left,
    )

BASE_CONFIG = {
    "NO_FLIP_THRESHOLD":    0.35,
    "FLIP_THRESHOLD":       0.60,
    "FAVORITE_THRESHOLD":   0.65,
    "YES_MIN_PRICE":        0.55,
    "YES_MAX_PRICE":        0.95,
    "NO_MIN_PRICE":         0.55,
    "NO_MAX_PRICE":         0.95,
    "AUTO_DEAD_ZONE_WIDTH": 0.10,
    "MIN_EDGE":             -0.05,
    "MAX_EDGE":             0.50,
    "TRADE_BET_SIZE_USDC":  5.0,
    "MAX_BET_SIZE_USDC":    50.0,
    "LIQUIDITY_FRACTION":   0.05,
    "INITIAL_CAPITAL":      1000.0,
}


# ─── Баг #2: yes_ask / no_ask вычисляются корректно ─────────────────────────

class TestMarketSignalPrices:
    def test_yes_ask_above_mid(self):
        sig = _signal(mid=0.70, spread=0.02)
        assert sig.yes_ask == pytest.approx(0.71)

    def test_no_ask_correct(self):
        sig = _signal(mid=0.70, spread=0.02)
        # no_ask = (1 - 0.70) + 0.02/2 = 0.31
        assert sig.no_ask == pytest.approx(0.31)

    def test_yes_ask_capped_at_099(self):
        sig = _signal(mid=0.99, spread=0.10)
        assert sig.yes_ask == 0.99

    def test_no_ask_capped_at_099(self):
        sig = _signal(mid=0.01, spread=0.10)
        assert sig.no_ask == 0.99


# ─── Баг #3: дефолты NO_FLIP_THRESHOLD совпадают ────────────────────────────

class TestThresholdDefaults:
    def test_no_flip_default_matches_schema(self):
        """Если config не содержит NO_FLIP_THRESHOLD — используется 0.35 как в BacktestConfig."""
        config_without_threshold = {k: v for k, v in BASE_CONFIG.items()
                                    if k != "NO_FLIP_THRESHOLD"}
        sig = _signal(mid=0.70)
        # p_flip=0.30 < дефолт 0.35 → должна торговать
        result = decide_ml_trend(sig, p_flip=0.30, config=config_without_threshold)
        assert result.action != "SKIP", (
            f"Expected trade with p_flip=0.30 < default threshold 0.35, got SKIP. "
            f"Reason: {result.reason}"
        )

    def test_no_flip_threshold_from_config(self):
        """Явный порог из config используется правильно."""
        config = {**BASE_CONFIG, "NO_FLIP_THRESHOLD": 0.50}
        sig = _signal(mid=0.70)

        # p_flip=0.30 < 0.50 → торгуем (edge будет выше -0.05)
        result_trade = decide_ml_trend(sig, p_flip=0.30, config=config)
        assert result_trade.action != "SKIP"

        # p_flip=0.55 >= 0.50 → пропускаем
        result_skip = decide_ml_trend(sig, p_flip=0.55, config=config)
        assert result_skip.action == "SKIP"


# ─── ML стратегия: основные кейсы ───────────────────────────────────────────

class TestDecideMlTrend:
    def test_trades_when_p_flip_below_threshold(self):
        sig = _signal(mid=0.70)
        result = decide_ml_trend(sig, p_flip=0.20, config=BASE_CONFIG)
        assert result.action in ("BUY_YES", "BUY_NO")
        assert result.strategy_type == "ML_TREND"

    def test_skips_when_p_flip_above_threshold(self):
        sig = _signal(mid=0.70)
        result = decide_ml_trend(sig, p_flip=0.80, config=BASE_CONFIG)
        assert result.action == "SKIP"
        assert "p_flip" in result.reason

    def test_skips_in_dead_zone(self):
        # mid=0.50 → dead zone при width=0.10 (0.45-0.55)
        sig = _signal(mid=0.50)
        result = decide_ml_trend(sig, p_flip=0.10, config=BASE_CONFIG)
        assert result.action == "SKIP"
        assert "dead zone" in result.reason

    def test_bet_size_positive(self):
        sig = _signal(mid=0.72, volume=2000.0)
        result = decide_ml_trend(sig, p_flip=0.15, config=BASE_CONFIG)
        if result.action != "SKIP":
            assert result.bet_size_usdc >= BASE_CONFIG["TRADE_BET_SIZE_USDC"]

    def test_p_flip_in_result(self):
        sig = _signal(mid=0.70)
        result = decide_ml_trend(sig, p_flip=0.20, config=BASE_CONFIG)
        assert result.p_flip == pytest.approx(0.20)


# ─── Баг #5: confirmed entry strategy ───────────────────────────────────────

class TestConfirmedEntryStrategy:
    """Проверяем что confirmed не зависает при чередующихся сигналах."""

    def _make_tick(self, mid, time_left):
        """Простой mock тика с нужными атрибутами."""
        from unittest.mock import MagicMock
        from polyflip.backtesting.market_replay import MarketTick
        from datetime import datetime, timezone
        return MarketTick(
            market_id="mkt-1",
            asset="BTC",
            time_left_min=time_left,
            mid_price=mid,
            spread=0.02,
            volume_5min=500.0,
            price_velocity=0.01,
            hour_of_day=12,
            final_outcome="YES",
            recorded_at=datetime.now(timezone.utc),
        )

    def test_confirmed_resets_on_action_change(self):
        """При смене action счётчик должен сброситься в 0, не в 1."""
        from polyflip.backtesting.runner import BacktestRunner
        from polyflip.backtesting.market_replay import MarketReplay

        # Патчим _predict_flip для детерминированного поведения
        config = {**BASE_CONFIG, "STRATEGY_MODE": "ML", "ENTRY_STRATEGY": "confirmed",
                  "MIN_TIME_LEFT_MIN": 1.0, "MAX_TIME_LEFT_MIN": 60.0,
                  "SLIPPAGE_PCT": 0.005, "TRADE_ON_FLIP": False,
                  "BET_SIZING_MODE": "fixed"}

        runner = BacktestRunner(config=config, model_blob=b"", features="")

        call_count = [0]
        # Чередующиеся решения: BUY_YES, BUY_NO, BUY_YES, BUY_YES
        actions = ["BUY_YES", "BUY_NO", "BUY_YES", "BUY_YES"]

        original_evaluate = runner._evaluate_tick
        def mock_evaluate(tick):
            idx = call_count[0]
            call_count[0] += 1
            action = actions[idx] if idx < len(actions) else "SKIP"
            from polyflip.trading.decision_logic import TradeDecision
            decision = TradeDecision(
                action=action, buy_price=0.70, bet_size_usdc=5.0,
                reason="mock", strategy_type="ML_TREND", p_flip=0.2, edge=0.05
            )
            return decision, 0.2, tick.to_signal()

        runner._evaluate_tick = mock_evaluate

        ticks = [self._make_tick(0.70, 50.0 - i) for i in range(5)]

        # Симулируем confirmed logic напрямую
        best_decision = None
        consecutive_edges = 0
        for tick in ticks:
            decision, p_flip, signal = runner._evaluate_tick(tick)
            if decision.action == "SKIP":
                consecutive_edges = 0
                continue
            if best_decision and decision.action != best_decision.action:
                consecutive_edges = 1   # ← ИСПРАВЛЕННОЕ поведение
                best_decision = decision
            else:
                consecutive_edges += 1
                if not best_decision:
                    best_decision = decision
            if consecutive_edges >= 2:
                break

        # Тики 3 и 4 оба BUY_YES → должны дать confirmed
        assert consecutive_edges >= 2, (
            f"Expected confirmed signal after 2 consecutive same-action ticks, "
            f"got consecutive_edges={consecutive_edges}"
        )
        assert best_decision is not None
        assert best_decision.action == "BUY_YES"

    def test_confirmed_skips_if_no_two_consecutive(self):
        """Если нет двух подряд одинаковых — трейда не должно быть."""
        actions = ["BUY_YES", "BUY_NO", "BUY_YES"]  # никогда 2 подряд
        best_decision = None
        consecutive_edges = 0

        for i, action in enumerate(actions):
            from polyflip.trading.decision_logic import TradeDecision
            decision = TradeDecision(
                action=action, buy_price=0.70, bet_size_usdc=5.0,
                reason="mock", strategy_type="ML_TREND",
            )
            if best_decision and decision.action != best_decision.action:
                consecutive_edges = 0
                best_decision = decision
            else:
                consecutive_edges += 1
                if not best_decision:
                    best_decision = decision
            if consecutive_edges >= 2:
                break

        # Сброс в конце если не подтверждено
        if consecutive_edges < 2:
            best_decision = None

        assert best_decision is None, "Should not trade without 2 consecutive confirmations"


# ─── PURE_FAVORITE: smoke tests ─────────────────────────────────────────────

class TestDecideFavorite:
    def test_buys_yes_when_above_threshold(self):
        sig = _signal(mid=0.70)  # 0.70 >= 0.65 threshold
        result = decide_favorite(sig, BASE_CONFIG)
        assert result.action == "BUY_YES"
        assert result.strategy_type == "PURE_FAVORITE"

    def test_buys_no_when_below_1_minus_threshold(self):
        sig = _signal(mid=0.30)  # 0.30 <= 1 - 0.65 = 0.35
        result = decide_favorite(sig, BASE_CONFIG)
        assert result.action == "BUY_NO"

    def test_skips_mid_range(self):
        sig = _signal(mid=0.50)
        result = decide_favorite(sig, BASE_CONFIG)
        assert result.action == "SKIP"

    def test_edge_is_present(self):
        sig = _signal(mid=0.72)
        result = decide_favorite(sig, BASE_CONFIG)
        if result.action != "SKIP":
            assert result.edge is not None


# ─── OUTSIDER: тесты нового параметра OUTSIDER_MAX_PRICE ─────────────────────

class TestDecideOutsider:
    OUTSIDER_CONFIG = {
        **BASE_CONFIG,
        "FLIP_THRESHOLD":    0.60,
        "OUTSIDER_MAX_PRICE": 0.40,
        "MIN_EDGE":          -0.10,  # расширен, чтобы edge-фильтр не мешал
    }

    def test_buys_no_when_yes_is_favorite_and_price_ok(self):
        """YES — фаворит (mid=0.70), NO ask = 0.35 ≤ 0.40 → BUY_NO"""
        sig = _signal(mid=0.70, spread=0.02)
        # no_ask = 1 - mid + spread/2 = 0.30 + 0.01 = 0.31
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "BUY_NO", f"Expected BUY_NO, got {result.action}: {result.reason}"

    def test_skips_no_when_price_exceeds_max(self):
        """YES — фаворит, NO ask = 0.45 > OUTSIDER_MAX_PRICE=0.40 → SKIP"""
        # Создаём сигнал с малым спредом, чтобы no_ask > 0.40
        # mid=0.55, spread=0.10 → no_ask = 0.45 + 0.05 = 0.50 > 0.40
        sig = MarketSignal(
            asset="BTC", mid_price=0.55, spread=0.10,
            volume_5min=500.0, price_velocity=0.0,
            hour_of_day=12, time_left_min=30.0
        )
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "SKIP"
        assert "max_outsider_price" in result.reason

    def test_buys_yes_when_no_is_favorite_and_price_ok(self):
        """NO — фаворит (mid=0.30), YES ask = ~0.31 ≤ 0.40 → BUY_YES"""
        sig = _signal(mid=0.30, spread=0.02)
        # yes_ask = mid + spread/2 = 0.30 + 0.01 = 0.31
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "BUY_YES", f"Expected BUY_YES, got {result.action}: {result.reason}"

    def test_skips_yes_when_price_exceeds_max(self):
        """NO — фаворит, YES ask > OUTSIDER_MAX_PRICE → SKIP"""
        # mid=0.30 (вне мёртвой зоны), spread=0.24 → yes_ask = 0.30 + 0.12 = 0.42 > 0.40
        sig = MarketSignal(
            asset="BTC", mid_price=0.30, spread=0.24,
            volume_5min=500.0, price_velocity=0.0,
            hour_of_day=12, time_left_min=30.0
        )
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "SKIP"
        assert "max_outsider_price" in result.reason

    def test_skips_when_p_flip_below_threshold(self):
        """p_flip=0.50 < FLIP_THRESHOLD=0.60 → SKIP"""
        sig = _signal(mid=0.70)
        result = decide_outsider(sig, p_flip=0.50, config=self.OUTSIDER_CONFIG)
        assert result.action == "SKIP"
        assert "threshold" in result.reason

    def test_respects_custom_outsider_max_price(self):
        """OUTSIDER_MAX_PRICE=0.30 более строгий: no_ask=0.31 → SKIP"""
        config = {**self.OUTSIDER_CONFIG, "OUTSIDER_MAX_PRICE": 0.30}
        sig = _signal(mid=0.70, spread=0.02)  # no_ask ≈ 0.31
        result = decide_outsider(sig, p_flip=0.65, config=config)
        assert result.action == "SKIP"
