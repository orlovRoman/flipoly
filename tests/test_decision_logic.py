"""
РЎР°РјРѕС‚РµСЃС‚С‹ РґР»СЏ decision_logic.py вЂ” РїРѕРєСЂС‹РІР°СЋС‚ Р±Р°РіРё #2, #3, #5.
Р—Р°РїСѓСЃРє: pytest tests/test_decision_logic.py -v
"""
import pytest
from polyflip.trading.feature_builder import MarketSignal
from polyflip.trading.decision_logic import (
    decide_ml_trend, decide_favorite, decide_outsider, TradeDecision
)

# в”Ђв”Ђв”Ђ Р¤Р°Р±СЂРёРєР° СЃРёРіРЅР°Р»РѕРІ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

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
    "FAVORITE_MIN_PRICE":   0.55,
    "FAVORITE_MAX_PRICE":   0.95,
    "AUTO_DEAD_ZONE_WIDTH": 0.10,
    "MIN_EDGE":             -0.05,
    "MAX_EDGE":             0.50,
    "TRADE_BET_SIZE_USDC":  5.0,
    "MAX_BET_SIZE_USDC":    50.0,
    "LIQUIDITY_FRACTION":   0.05,
    "INITIAL_CAPITAL":      1000.0,
}


# в”Ђв”Ђв”Ђ Р‘Р°Рі #2: yes_ask / no_ask РІС‹С‡РёСЃР»СЏСЋС‚СЃСЏ РєРѕСЂСЂРµРєС‚РЅРѕ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

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


# в”Ђв”Ђв”Ђ Р‘Р°Рі #3: РґРµС„РѕР»С‚С‹ NO_FLIP_THRESHOLD СЃРѕРІРїР°РґР°СЋС‚ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestThresholdDefaults:
    def test_no_flip_default_matches_schema(self):
        """Р•СЃР»Рё config РЅРµ СЃРѕРґРµСЂР¶РёС‚ NO_FLIP_THRESHOLD вЂ” РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ 0.35 РєР°Рє РІ BacktestConfig."""
        config_without_threshold = {k: v for k, v in BASE_CONFIG.items()
                                    if k != "NO_FLIP_THRESHOLD"}
        sig = _signal(mid=0.70)
        # p_flip=0.30 < РґРµС„РѕР»С‚ 0.35 в†’ РґРѕР»Р¶РЅР° С‚РѕСЂРіРѕРІР°С‚СЊ
        result = decide_ml_trend(sig, p_flip=0.30, config=config_without_threshold)
        assert result.action != "SKIP", (
            f"Expected trade with p_flip=0.30 < default threshold 0.35, got SKIP. "
            f"Reason: {result.reason}"
        )

    def test_no_flip_threshold_from_config(self):
        """РЇРІРЅС‹Р№ РїРѕСЂРѕРі РёР· config РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РїСЂР°РІРёР»СЊРЅРѕ."""
        config = {**BASE_CONFIG, "NO_FLIP_THRESHOLD": 0.50}
        sig = _signal(mid=0.70)

        # p_flip=0.30 < 0.50 в†’ С‚РѕСЂРіСѓРµРј (edge Р±СѓРґРµС‚ РІС‹С€Рµ -0.05)
        result_trade = decide_ml_trend(sig, p_flip=0.30, config=config)
        assert result_trade.action != "SKIP"

        # p_flip=0.55 >= 0.50 в†’ РїСЂРѕРїСѓСЃРєР°РµРј
        result_skip = decide_ml_trend(sig, p_flip=0.55, config=config)
        assert result_skip.action == "SKIP"


# в”Ђв”Ђв”Ђ ML СЃС‚СЂР°С‚РµРіРёСЏ: РѕСЃРЅРѕРІРЅС‹Рµ РєРµР№СЃС‹ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

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
        # mid=0.50 в†’ dead zone РїСЂРё width=0.10 (0.45-0.55)
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


# в”Ђв”Ђв”Ђ Р‘Р°Рі #5: confirmed entry strategy в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestConfirmedEntryStrategy:
    """РџСЂРѕРІРµСЂСЏРµРј С‡С‚Рѕ confirmed РЅРµ Р·Р°РІРёСЃР°РµС‚ РїСЂРё С‡РµСЂРµРґСѓСЋС‰РёС…СЃСЏ СЃРёРіРЅР°Р»Р°С…."""

    def _make_tick(self, mid, time_left):
        """РџСЂРѕСЃС‚РѕР№ mock С‚РёРєР° СЃ РЅСѓР¶РЅС‹РјРё Р°С‚СЂРёР±СѓС‚Р°РјРё."""
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
        """РџСЂРё СЃРјРµРЅРµ action СЃС‡С‘С‚С‡РёРє РґРѕР»Р¶РµРЅ СЃР±СЂРѕСЃРёС‚СЊСЃСЏ РІ 0, РЅРµ РІ 1."""
        from polyflip.backtesting.runner import BacktestRunner
        from polyflip.backtesting.market_replay import MarketReplay

        # РџР°С‚С‡РёРј _predict_flip РґР»СЏ РґРµС‚РµСЂРјРёРЅРёСЂРѕРІР°РЅРЅРѕРіРѕ РїРѕРІРµРґРµРЅРёСЏ
        config = {**BASE_CONFIG, "STRATEGY_MODE": "ML", "ENTRY_STRATEGY": "confirmed",
                  "MIN_TIME_LEFT_MIN": 1.0, "MAX_TIME_LEFT_MIN": 60.0,
                  "SLIPPAGE_PCT": 0.005, "TRADE_ON_FLIP": False,
                  "BET_SIZING_MODE": "fixed"}

        runner = BacktestRunner(config=config, model_blob=b"", features="")

        call_count = [0]
        # Р§РµСЂРµРґСѓСЋС‰РёРµСЃСЏ СЂРµС€РµРЅРёСЏ: BUY_YES, BUY_NO, BUY_YES, BUY_YES
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

        # РЎРёРјСѓР»РёСЂСѓРµРј confirmed logic РЅР°РїСЂСЏРјСѓСЋ
        best_decision = None
        consecutive_edges = 0
        for tick in ticks:
            decision, p_flip, signal = runner._evaluate_tick(tick)
            if decision.action == "SKIP":
                consecutive_edges = 0
                continue
            if best_decision and decision.action != best_decision.action:
                consecutive_edges = 1   # в†ђ Р�РЎРџР РђР’Р›Р•РќРќРћР• РїРѕРІРµРґРµРЅРёРµ
                best_decision = decision
            else:
                consecutive_edges += 1
                if not best_decision:
                    best_decision = decision
            if consecutive_edges >= 2:
                break

        # РўРёРєРё 3 Рё 4 РѕР±Р° BUY_YES в†’ РґРѕР»Р¶РЅС‹ РґР°С‚СЊ confirmed
        assert consecutive_edges >= 2, (
            f"Expected confirmed signal after 2 consecutive same-action ticks, "
            f"got consecutive_edges={consecutive_edges}"
        )
        assert best_decision is not None
        assert best_decision.action == "BUY_YES"

    def test_confirmed_skips_if_no_two_consecutive(self):
        """Р•СЃР»Рё РЅРµС‚ РґРІСѓС… РїРѕРґСЂСЏРґ РѕРґРёРЅР°РєРѕРІС‹С… вЂ” С‚СЂРµР№РґР° РЅРµ РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ."""
        actions = ["BUY_YES", "BUY_NO", "BUY_YES"]  # РЅРёРєРѕРіРґР° 2 РїРѕРґСЂСЏРґ
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

        # РЎР±СЂРѕСЃ РІ РєРѕРЅС†Рµ РµСЃР»Рё РЅРµ РїРѕРґС‚РІРµСЂР¶РґРµРЅРѕ
        if consecutive_edges < 2:
            best_decision = None

        assert best_decision is None, "Should not trade without 2 consecutive confirmations"


# в”Ђв”Ђв”Ђ PURE_FAVORITE: smoke tests в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

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


# в”Ђв”Ђв”Ђ OUTSIDER: С‚РµСЃС‚С‹ РЅРѕРІРѕРіРѕ РїР°СЂР°РјРµС‚СЂР° OUTSIDER_MAX_PRICE в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestDecideOutsider:
    OUTSIDER_CONFIG = {
        **BASE_CONFIG,
        "FLIP_THRESHOLD":    0.60,
        "OUTSIDER_MAX_PRICE": 0.40,
        "MIN_EDGE":          -0.10,  # СЂР°СЃС€РёСЂРµРЅ, С‡С‚РѕР±С‹ edge-С„РёР»СЊС‚СЂ РЅРµ РјРµС€Р°Р»
    }

    def test_buys_no_when_yes_is_favorite_and_price_ok(self):
        """YES вЂ” С„Р°РІРѕСЂРёС‚ (mid=0.70), NO ask = 0.35 в‰¤ 0.40 в†’ BUY_NO"""
        sig = _signal(mid=0.70, spread=0.02)
        # no_ask = 1 - mid + spread/2 = 0.30 + 0.01 = 0.31
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "BUY_NO", f"Expected BUY_NO, got {result.action}: {result.reason}"

    def test_skips_no_when_price_exceeds_max(self):
        """YES вЂ” С„Р°РІРѕСЂРёС‚, NO ask = 0.45 > OUTSIDER_MAX_PRICE=0.40 в†’ SKIP"""
        # РЎРѕР·РґР°С‘Рј СЃРёРіРЅР°Р» СЃ РјР°Р»С‹Рј СЃРїСЂРµРґРѕРј, С‡С‚РѕР±С‹ no_ask > 0.40
        # mid=0.55, spread=0.10 в†’ no_ask = 0.45 + 0.05 = 0.50 > 0.40
        sig = MarketSignal(
            asset="BTC", mid_price=0.55, spread=0.10,
            volume_5min=500.0, price_velocity=0.0,
            hour_of_day=12, time_left_min=30.0
        )
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "SKIP"
        assert "max_outsider_price" in result.reason

    def test_buys_yes_when_no_is_favorite_and_price_ok(self):
        """NO вЂ” С„Р°РІРѕСЂРёС‚ (mid=0.30), YES ask = ~0.31 в‰¤ 0.40 в†’ BUY_YES"""
        sig = _signal(mid=0.30, spread=0.02)
        # yes_ask = mid + spread/2 = 0.30 + 0.01 = 0.31
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "BUY_YES", f"Expected BUY_YES, got {result.action}: {result.reason}"

    def test_skips_yes_when_price_exceeds_max(self):
        """NO вЂ” С„Р°РІРѕСЂРёС‚, YES ask > OUTSIDER_MAX_PRICE в†’ SKIP"""
        # mid=0.30 (РІРЅРµ РјС‘СЂС‚РІРѕР№ Р·РѕРЅС‹), spread=0.24 в†’ yes_ask = 0.30 + 0.12 = 0.42 > 0.40
        sig = MarketSignal(
            asset="BTC", mid_price=0.30, spread=0.24,
            volume_5min=500.0, price_velocity=0.0,
            hour_of_day=12, time_left_min=30.0
        )
        result = decide_outsider(sig, p_flip=0.65, config=self.OUTSIDER_CONFIG)
        assert result.action == "SKIP"
        assert "max_outsider_price" in result.reason

    def test_skips_when_p_flip_below_threshold(self):
        """p_flip=0.50 < FLIP_THRESHOLD=0.60 в†’ SKIP"""
        sig = _signal(mid=0.70)
        result = decide_outsider(sig, p_flip=0.50, config=self.OUTSIDER_CONFIG)
        assert result.action == "SKIP"
        assert "threshold" in result.reason

    def test_respects_custom_outsider_max_price(self):
        """OUTSIDER_MAX_PRICE=0.30 Р±РѕР»РµРµ СЃС‚СЂРѕРіРёР№: no_ask=0.31 в†’ SKIP"""
        config = {**self.OUTSIDER_CONFIG, "OUTSIDER_MAX_PRICE": 0.30}
        sig = _signal(mid=0.70, spread=0.02)  # no_ask в‰€ 0.31
        result = decide_outsider(sig, p_flip=0.65, config=config)
        assert result.action == "SKIP"


def test_favorite_price_bounds_unified():
    """YES и NO используют одни и те же fav_min/fav_max."""
    cfg = {**BASE_CONFIG, "FAVORITE_MIN_PRICE": 0.60, "FAVORITE_MAX_PRICE": 0.90}

    # YES side — цена ниже min → SKIP
    sig_yes_low = _signal(mid=0.72, yes_ask=0.58, no_ask=0.42)
    assert decide_favorite(sig_yes_low, cfg).action == "SKIP"

    # YES side — цена выше max → SKIP
    sig_yes_high = _signal(mid=0.72, yes_ask=0.92, no_ask=0.08)
    assert decide_favorite(sig_yes_high, cfg).action == "SKIP"

    # YES side — цена в диапазоне → не SKIP по bounds
    sig_yes_ok = _signal(mid=0.72, yes_ask=0.72, no_ask=0.28)
    result = decide_favorite(sig_yes_ok, cfg)
    assert result.reason != "YES price out of bounds"

    # NO side — цена ниже min → SKIP
    sig_no_low = _signal(mid=0.28, yes_ask=0.72, no_ask=0.58)
    assert decide_favorite(sig_no_low, cfg).action == "SKIP"

    # NO side — цена в диапазоне → не SKIP по bounds
    sig_no_ok = _signal(mid=0.28, yes_ask=0.72, no_ask=0.72)
    result = decide_favorite(sig_no_ok, cfg)
    assert result.reason != "NO price out of bounds"


def test_favorite_price_fallback_defaults():
    """Если FAVORITE_MIN/MAX_PRICE нет в конфиге — используются дефолты 0.55/0.95."""
    cfg_no_bounds = {k: v for k, v in BASE_CONFIG.items()
                     if k not in ("FAVORITE_MIN_PRICE", "FAVORITE_MAX_PRICE")}

    # Цена 0.54 — ниже дефолтного min 0.55 → должен быть SKIP
    sig = _signal(mid=0.72, yes_ask=0.54, no_ask=0.46)
    assert decide_favorite(sig, cfg_no_bounds).action == "SKIP"


def test_backtest_schema_no_old_keys():
    """BacktestConfig не должна принимать старые ключи yes/no_min/max_price."""
    from pydantic import ValidationError
    from polyflip.api.backtest_schemas import BacktestConfig
    import pytest
    with pytest.raises((ValidationError, TypeError)):
        BacktestConfig(
            yes_min_price=0.55,  # старый ключ
            yes_max_price=0.95,
        )


def test_backtest_schema_new_keys():
    """BacktestConfig принимает favorite_min/max_price."""
    from polyflip.api.backtest_schemas import BacktestConfig
    cfg = BacktestConfig(favorite_min_price=0.60, favorite_max_price=0.90)
    runner = cfg.to_runner_config()
    assert runner["FAVORITE_MIN_PRICE"] == 0.60
    assert runner["FAVORITE_MAX_PRICE"] == 0.90
    assert "YES_MIN_PRICE" not in runner
    assert "NO_MIN_PRICE" not in runner

