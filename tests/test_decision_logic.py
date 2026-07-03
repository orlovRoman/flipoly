import pytest
from polyflip.trading.decision_logic import (
    decide_favorite, decide_ml_trend, decide_outsider,
)
from polyflip.trading.feature_builder import MarketSignal
from polyflip.constants import (
    FAVORITE_THRESHOLD, FLIP_THRESHOLD, NO_FLIP_THRESHOLD,
    FAVORITE_MIN_PRICE, FAVORITE_MAX_PRICE, FAVORITE_MIN_EDGE,
    MIN_EDGE, MAX_EDGE_FILTER, OUTSIDER_MAX_PRICE, AUTO_DEAD_ZONE_WIDTH,
)


def _signal(mid=0.72, yes_ask=0.74, no_ask=0.28, vol=500.0) -> MarketSignal:
    return MarketSignal(
        market_id="test-market",
        mid_price=mid,
        yes_ask=yes_ask,
        no_ask=no_ask,
        volume_5min=vol,
    )


BASE_CONFIG = {
    "FAVORITE_THRESHOLD": FAVORITE_THRESHOLD,
    "AUTO_DEAD_ZONE_WIDTH": AUTO_DEAD_ZONE_WIDTH,
    "FAVORITE_MIN_PRICE": FAVORITE_MIN_PRICE,
    "FAVORITE_MAX_PRICE": FAVORITE_MAX_PRICE,
    "FAVORITE_MIN_EDGE": FAVORITE_MIN_EDGE,
    "MIN_EDGE": MIN_EDGE,
    "MAX_EDGE": MAX_EDGE_FILTER,
    "FLIP_THRESHOLD": FLIP_THRESHOLD,
    "NO_FLIP_THRESHOLD": NO_FLIP_THRESHOLD,
    "OUTSIDER_MAX_PRICE": OUTSIDER_MAX_PRICE,
    "TRADE_BET_SIZE_USDC": 5.0,
    "MAX_BET_SIZE_USDC": 50.0,
    "LIQUIDITY_FRACTION": 0.05,
}


# ─────────────────────────────────────────
# decide_favorite
# ─────────────────────────────────────────

class TestDecideFavorite:

    def test_buy_yes_when_mid_above_threshold(self):
        sig = _signal(mid=0.72, yes_ask=0.74)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "BUY_YES"
        assert d.buy_price == pytest.approx(0.74)
        assert d.strategy_type == "PURE_FAVORITE"

    def test_buy_no_when_mid_below_threshold(self):
        # mid=0.28 → 1-threshold=0.45 → 0.28 < 0.45 → фаворит NO
        sig = _signal(mid=0.28, yes_ask=0.72, no_ask=0.30)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "BUY_NO"
        assert d.strategy_type == "PURE_FAVORITE"

    def test_skip_dead_zone(self):
        # mid=0.50 попадает в мёртвую зону
        sig = _signal(mid=0.50)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"
        assert "dead zone" in d.reason

    def test_skip_yes_price_out_of_bounds(self):
        # yes_ask=0.97 > FAVORITE_MAX_PRICE=0.95
        sig = _signal(mid=0.72, yes_ask=0.97)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"
        assert "out of bounds" in d.reason

    def test_skip_yes_price_too_low(self):
        # yes_ask=0.50 < FAVORITE_MIN_PRICE=0.55
        sig = _signal(mid=0.72, yes_ask=0.50)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"

    def test_skip_when_edge_below_favorite_min_edge(self):
        # mid=0.56 yes_ask=0.60 → edge = 0.56/0.60 - 1 = -0.067 < FAVORITE_MIN_EDGE=-0.01
        sig = _signal(mid=0.56, yes_ask=0.60)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"
        assert d.edge is not None and d.edge < FAVORITE_MIN_EDGE

    def test_bet_size_positive(self):
        sig = _signal(mid=0.75, yes_ask=0.70, vol=1000.0)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "BUY_YES"
        assert d.bet_size_usdc >= BASE_CONFIG["TRADE_BET_SIZE_USDC"]

    def test_no_clear_favorite_skips(self):
        # mid=0.58, threshold=0.55 → YES side. yes_ask=0.80 > MAX_PRICE → SKIP
        sig = _signal(mid=0.58, yes_ask=0.80)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"


# ─────────────────────────────────────────
# decide_ml_trend
# ─────────────────────────────────────────

class TestDecideMlTrend:

    def test_skip_when_p_flip_above_threshold(self):
        sig = _signal(mid=0.72, yes_ask=0.74)
        d = decide_ml_trend(sig, p_flip=0.50, config=BASE_CONFIG)
        assert d.action == "SKIP"
        assert "p_flip" in d.reason

    def test_buy_yes_when_p_flip_low(self):
        sig = _signal(mid=0.72, yes_ask=0.70, vol=1000.0)
        d = decide_ml_trend(sig, p_flip=0.10, config=BASE_CONFIG)
        assert d.action == "BUY_YES"
        assert d.strategy_type == "ML_TREND"
        assert d.p_flip == pytest.approx(0.10)

    def test_edge_computed_from_p_win(self):
        # p_flip=0.10 → p_win=0.90, yes_ask=0.70 → edge=0.90/0.70-1≈0.286
        sig = _signal(mid=0.72, yes_ask=0.70, vol=1000.0)
        d = decide_ml_trend(sig, p_flip=0.10, config=BASE_CONFIG)
        assert d.edge is not None
        expected_edge = round(0.90 / 0.70 - 1.0, 4)
        assert d.edge == pytest.approx(expected_edge, abs=1e-3)

    def test_skip_when_edge_exceeds_max(self):
        # p_flip=0.01 → p_win=0.99, yes_ask=0.60 → edge≈0.65 > MAX_EDGE_FILTER=0.20
        sig = _signal(mid=0.72, yes_ask=0.60, vol=1000.0)
        d = decide_ml_trend(sig, p_flip=0.01, config=BASE_CONFIG)
        assert d.action == "SKIP"
        assert "out of bounds" in d.reason.lower() or "Edge" in d.reason

    def test_skip_when_favorite_fails_price_bounds(self):
        sig = _signal(mid=0.72, yes_ask=0.97)  # out of bounds
        d = decide_ml_trend(sig, p_flip=0.10, config=BASE_CONFIG)
        assert d.action == "SKIP"


# ─────────────────────────────────────────
# decide_outsider
# ─────────────────────────────────────────

class TestDecideOutsider:

    def test_skip_when_p_flip_below_threshold(self):
        sig = _signal(mid=0.72)
        d = decide_outsider(sig, p_flip=0.40, config=BASE_CONFIG)
        assert d.action == "SKIP"

    def test_buy_no_when_yes_is_favorite(self):
        # mid=0.72 (YES фаворит) → покупаем NO (аутсайдера)
        # no_ask=0.30 < OUTSIDER_MAX_PRICE=0.45 → OK
        sig = _signal(mid=0.72, no_ask=0.30, vol=1000.0)
        d = decide_outsider(sig, p_flip=0.70, config=BASE_CONFIG)
        assert d.action == "BUY_NO"
        assert d.strategy_type == "OUTSIDER"

    def test_skip_outsider_no_ask_too_high(self):
        # no_ask=0.50 > OUTSIDER_MAX_PRICE=0.45
        sig = _signal(mid=0.72, no_ask=0.50)
        d = decide_outsider(sig, p_flip=0.70, config=BASE_CONFIG)
        assert d.action == "SKIP"
        assert "OUTSIDER_MAX_PRICE" in d.reason or "max_outsider" in d.reason.lower()

    def test_skip_dead_zone(self):
        sig = _signal(mid=0.50)
        d = decide_outsider(sig, p_flip=0.70, config=BASE_CONFIG)
        assert d.action == "SKIP"
        assert "dead zone" in d.reason


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
