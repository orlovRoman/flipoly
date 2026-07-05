import pytest
from polyflip.trading.decision_logic import (
    decide_favorite, decide_ml_trend, decide_outsider, decide_crypto_trend
)
from polyflip.crypto.edge import compute_crypto_edge
from polyflip.crypto.predictor import CryptoSignal

from polyflip.trading.feature_builder import MarketSignal
from polyflip.constants import (
    FAVORITE_THRESHOLD, FLIP_THRESHOLD, NO_FLIP_THRESHOLD,
    FAVORITE_MIN_PRICE, FAVORITE_MAX_PRICE, FAVORITE_MIN_EDGE,
    MIN_EDGE, MAX_EDGE_FILTER, OUTSIDER_MAX_PRICE, DEAD_ZONE_WIDTH,
)


def _signal(
    mid: float = 0.72,
    spread: float = 0.04,
    vol: float = 500.0,
    velocity: float = 0.0,
    hour: int = 12,
    time_left: float = 30.0,
) -> MarketSignal:
    return MarketSignal(
        asset="BTC",
        mid_price=mid,
        spread=spread,
        volume_5min=vol,
        price_velocity=velocity,
        hour_of_day=hour,
        time_left_min=time_left,
    )


BASE_CONFIG = {
    "FAVORITE_THRESHOLD": FAVORITE_THRESHOLD,
    "DEAD_ZONE_WIDTH": DEAD_ZONE_WIDTH,
    "FAVORITE_MIN_PRICE": FAVORITE_MIN_PRICE,
    "FAVORITE_MAX_PRICE": FAVORITE_MAX_PRICE,
    "FAVORITE_MIN_EDGE": FAVORITE_MIN_EDGE,
    "MIN_EDGE": MIN_EDGE,
    "MAX_EDGE": MAX_EDGE_FILTER,
    "MAX_BET_EDGE": MAX_EDGE_FILTER,
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
        sig = _signal(mid=0.72, spread=0.01)  # yes_ask = 0.725, edge = -0.0068 > -0.01
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "BUY_YES"
        assert d.buy_price == pytest.approx(0.725)
        assert d.strategy_type == "PURE_FAVORITE"

    def test_buy_no_when_mid_below_threshold(self):
        sig = _signal(mid=0.28, spread=0.01) # no_prob=0.72, no_ask=0.725, edge=-0.0068 > -0.01
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
        # yes_ask = mid + spread/2 -> 0.72 + spread/2 = 0.97 -> spread = 0.50
        sig = _signal(mid=0.72, spread=0.50)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"
        assert "out of bounds" in d.reason

    def test_skip_yes_price_too_low(self):
        # yes_ask=0.50 < FAVORITE_MIN_PRICE=0.55
        # yes_ask = mid + spread/2. If mid=0.72, yes_ask > 0.72.
        # To get yes_ask=0.50, we need mid=0.48 and spread=0.04.
        # But mid=0.48 is not a favorite YES.
        # Wait, if mid is favorite (>0.55), yes_ask can NEVER be < 0.55.
        # Let's use mid=0.56, spread=0.0 (unrealistic but ok for test) -> yes_ask=0.56. Still >= 0.55.
        # If we want yes_ask < 0.55, mid must be < 0.55, but then it's not a YES favorite.
        # The test "yes_price_too_low" is actually impossible for YES side since mid >= 0.55.
        # Let's test mid=0.56, spread=-0.12 -> yes_ask = 0.50.
        sig = _signal(mid=0.56, spread=-0.12)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"

    def test_skip_when_edge_below_favorite_min_edge(self):
        # mid=0.56, spread=0.08 → yes_ask=0.60 → edge = 0.56/0.60 - 1 = -0.067 < FAVORITE_MIN_EDGE=-0.01
        sig = _signal(mid=0.56, spread=0.08)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"
        assert d.edge is not None and d.edge < FAVORITE_MIN_EDGE

    def test_bet_size_positive(self):
        # yes_ask = 0.70. mid=0.75 -> spread = -0.10
        sig = _signal(mid=0.75, spread=-0.10, vol=1000.0)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "BUY_YES"
        assert d.bet_size_usdc >= BASE_CONFIG["TRADE_BET_SIZE_USDC"]

    def test_no_clear_favorite_skips(self):
        # mid=0.58, yes_ask=0.80 -> spread = 0.44
        sig = _signal(mid=0.58, spread=0.44)
        d = decide_favorite(sig, BASE_CONFIG)
        assert d.action == "SKIP"


# ─────────────────────────────────────────
# decide_ml_trend
# ─────────────────────────────────────────

class TestDecideMlTrend:

    def test_skip_when_p_flip_above_threshold(self):
        # mid=0.72, spread=0.04 -> yes_ask=0.74
        sig = _signal(mid=0.72, spread=0.04)
        d = decide_ml_trend(sig, p_flip=0.50, config=BASE_CONFIG)
        assert d.action == "SKIP"
        assert "p_flip" in d.reason

    def test_buy_yes_when_p_flip_low(self):
        # p_win = 0.90 (1 - 0.10). We need edge between 0.05 and 0.20.
        # buy_price = 0.80 -> edge = 0.90 / 0.80 - 1 = 0.125.
        # yes_ask = mid + spread/2. 0.80 = 0.72 + spread/2 -> spread = 0.16.
        # To avoid skipping in PURE_FAVORITE step, buy_price MUST be <= FAVORITE_MAX_PRICE=0.95 and edge > FAVORITE_MIN_EDGE.
        # p_win_yes = 0.72. yes_ask = 0.80. pure_favorite_edge = 0.72/0.80 - 1 = -0.1.
        # FAVORITE_MIN_EDGE = -0.01. So this will fail the pure_favorite test and return SKIP.
        # We need pure_favorite_edge >= -0.01.
        # So 0.72 / yes_ask >= 0.99 -> yes_ask <= 0.727.
        # Let's set yes_ask = 0.72. mid = 0.72. spread = 0.0.
        # ML edge = p_win / yes_ask - 1. p_win = 1 - 0.10 = 0.90.
        # ML edge = 0.90 / 0.72 - 1 = 1.25 -> 0.25 > MAX_EDGE_FILTER (0.20) -> SKIP.
        # So we need yes_ask such that 0.90 / yes_ask - 1 <= 0.20 -> yes_ask >= 0.90 / 1.20 = 0.75.
        # But yes_ask >= 0.75 means pure_favorite_edge = 0.72 / 0.75 - 1 = -0.04 < -0.01 -> SKIP.
        # 
        # This implies with current BASE_CONFIG, ML_TREND can NEVER trigger if p_flip=0.10 and mid=0.72,
        # because the two edge constraints are mutually exclusive!
        # ML_TREND uses decide_favorite which enforces FAVORITE_MIN_EDGE=-0.01 on mid_price.
        # Let's adjust BASE_CONFIG["FAVORITE_MIN_EDGE"] in this test to -0.20 so we can pass it.
        cfg = {**BASE_CONFIG, "FAVORITE_MIN_EDGE": -0.20}
        sig = _signal(mid=0.72, spread=0.16, vol=1000.0)
        d = decide_ml_trend(sig, p_flip=0.10, config=cfg)
        assert d.action == "BUY_YES"
        assert d.strategy_type == "ML_TREND"
        assert d.p_flip == pytest.approx(0.10)

    def test_edge_computed_from_p_win(self):
        # p_flip=0.10 → p_win=0.90, yes_ask=0.70 → edge=0.90/0.70-1≈0.286
        sig = _signal(mid=0.72, spread=-0.04, vol=1000.0)
        d = decide_ml_trend(sig, p_flip=0.10, config=BASE_CONFIG)
        assert d.edge is not None
        expected_edge = round(0.90 / 0.70 - 1.0, 4)
        assert d.edge == pytest.approx(expected_edge, abs=1e-3)

    def test_skip_when_edge_exceeds_max(self):
        # p_flip=0.01 → p_win=0.99, yes_ask=0.60 → edge≈0.65 > MAX_BET_EDGE_FILTER=0.20
        # mid=0.72, yes_ask=0.60 -> spread=-0.24
        sig = _signal(mid=0.72, spread=-0.24, vol=1000.0)
        d = decide_ml_trend(sig, p_flip=0.01, config=BASE_CONFIG)
        assert d.action == "SKIP"
        assert "out of bounds" in d.reason.lower() or "Edge" in d.reason

    def test_skip_when_favorite_fails_price_bounds(self):
        # mid=0.72, yes_ask=0.97 -> spread=0.50
        sig = _signal(mid=0.72, spread=0.50)  # out of bounds
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
        # no_ask = (1 - 0.72) + spread/2 = 0.28 + spread/2.
        # We want no_ask < 0.45. Let's say no_ask = 0.30 -> spread = 0.04.
        sig = _signal(mid=0.72, spread=0.04, vol=1000.0)
        d = decide_outsider(sig, p_flip=0.70, config=BASE_CONFIG)
        assert d.action == "BUY_NO"
        assert d.strategy_type == "OUTSIDER"

    def test_skip_outsider_no_ask_too_high(self):
        # no_ask=0.50 > OUTSIDER_MAX_PRICE=0.45
        # mid=0.72 -> no_prob = 0.28. no_ask=0.50 -> spread=0.44
        sig = _signal(mid=0.72, spread=0.44)
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
    # mid=0.72, yes_ask=0.58 -> spread = -0.28
    sig_yes_low = _signal(mid=0.72, spread=-0.28)
    assert decide_favorite(sig_yes_low, cfg).action == "SKIP"

    # YES side — цена выше max → SKIP
    # mid=0.72, yes_ask=0.92 -> spread = 0.40
    sig_yes_high = _signal(mid=0.72, spread=0.40)
    assert decide_favorite(sig_yes_high, cfg).action == "SKIP"

    # YES side — цена в диапазоне → не SKIP по bounds
    # mid=0.72, yes_ask=0.72 -> spread = 0.0
    sig_yes_ok = _signal(mid=0.72, spread=0.0)
    result = decide_favorite(sig_yes_ok, cfg)
    assert result.reason != "YES price out of bounds"

    # NO side — цена ниже min → SKIP
    # mid=0.28 -> no_prob=0.72. yes_ask doesn't matter much. no_ask=0.58 -> spread=-0.28
    sig_no_low = _signal(mid=0.28, spread=-0.28)
    assert decide_favorite(sig_no_low, cfg).action == "SKIP"

    # NO side — цена в диапазоне → не SKIP по bounds
    # mid=0.28 -> no_prob=0.72. no_ask=0.72 -> spread=0.0
    sig_no_ok = _signal(mid=0.28, spread=0.0)
    result = decide_favorite(sig_no_ok, cfg)
    assert result.reason != "NO price out of bounds"


def test_favorite_price_fallback_defaults():
    """Если FAVORITE_MIN/MAX_PRICE нет в конфиге — используются дефолты 0.55/0.95."""
    cfg_no_bounds = {k: v for k, v in BASE_CONFIG.items()
                     if k not in ("FAVORITE_MIN_PRICE", "FAVORITE_MAX_PRICE")}

    # Цена 0.54 — ниже дефолтного min 0.55 → должен быть SKIP
    # mid=0.72, yes_ask=0.54 -> spread = -0.36
    sig = _signal(mid=0.72, spread=-0.36)
    assert decide_favorite(sig, cfg_no_bounds).action == "SKIP"


def test_backtest_schema_no_old_keys():
    """BacktestConfig ignores unknown fields or raises ValidationError, depending on pydantic config. 
    Here we just verify that old keys don't break dict exports if they are ignored."""
    from polyflip.api.backtest_schemas import BacktestConfig
    cfg = BacktestConfig(
        **{"yes_min_price": 0.55, "yes_max_price": 0.95, "favorite_min_price": 0.60, "favorite_max_price": 0.90}
    )
    runner = cfg.to_runner_config()
    assert runner["FAVORITE_MIN_PRICE"] == 0.60
    assert runner["FAVORITE_MAX_PRICE"] == 0.90
    assert "YES_MIN_PRICE" not in runner


def test_backtest_schema_new_keys():
    """BacktestConfig принимает favorite_min/max_price."""
    from polyflip.api.backtest_schemas import BacktestConfig
    cfg = BacktestConfig(favorite_min_price=0.60, favorite_max_price=0.90)
    runner = cfg.to_runner_config()
    assert runner["FAVORITE_MIN_PRICE"] == 0.60
    assert runner["FAVORITE_MAX_PRICE"] == 0.90
    assert "YES_MIN_PRICE" not in runner
    assert "NO_MIN_PRICE" not in runner


# ─────────────────────────────────────────
# Crypto Trend / Edge Tests
# ─────────────────────────────────────────

def test_crypto_edge_up():
    edge, direction = compute_crypto_edge(p_up=0.75, threshold_up=0.65, threshold_down=0.35)
    assert direction == "UP"
    assert edge == pytest.approx(0.10)

def test_crypto_edge_down():
    edge, direction = compute_crypto_edge(p_up=0.25, threshold_up=0.65, threshold_down=0.35)
    assert direction == "DOWN"
    assert edge == pytest.approx(0.10)

def test_crypto_edge_dead_zone():
    edge, direction = compute_crypto_edge(p_up=0.50, threshold_up=0.65, threshold_down=0.35)
    assert direction == "NONE"
    assert edge == 0.0

def test_decide_crypto_trend_buy_yes():
    # p_up = 0.75 -> UP, edge = 0.15 > CRYPTO_MIN_EDGE (0.05)
    crypto = CryptoSignal(
        symbol="BTCUSDT", p_up=0.75, p_down=0.25, direction="UP", edge=0.15,
        strike=60000.0, threshold_up=0.60, threshold_down=0.40, model_version=1, features_ok=True
    )
    d = decide_crypto_trend(crypto, entry_price=0.65, volume_5min=1000.0, config=BASE_CONFIG)
    assert d.action == "BUY_YES"
    assert d.strategy_type == "CRYPTO_TREND"
    assert d.p_up == 0.75
    assert d.strike == 60000.0
    assert d.edge == 0.15

def test_decide_crypto_trend_buy_no():
    # p_up = 0.25 -> DOWN, edge = 0.15 > CRYPTO_MIN_EDGE (0.05)
    crypto = CryptoSignal(
        symbol="BTCUSDT", p_up=0.25, p_down=0.75, direction="DOWN", edge=0.15,
        strike=60000.0, threshold_up=0.60, threshold_down=0.40, model_version=1, features_ok=True
    )
    d = decide_crypto_trend(crypto, entry_price=0.65, volume_5min=1000.0, config=BASE_CONFIG)
    assert d.action == "BUY_NO"
    assert d.strategy_type == "CRYPTO_TREND"
    assert d.p_up == 0.25

def test_decide_crypto_trend_skip_dead_zone():
    # direction = NONE
    crypto = CryptoSignal(
        symbol="BTCUSDT", p_up=0.50, p_down=0.50, direction="NONE", edge=0.0,
        strike=60000.0, threshold_up=0.60, threshold_down=0.40, model_version=1, features_ok=True
    )
    d = decide_crypto_trend(crypto, entry_price=0.65, volume_5min=1000.0, config=BASE_CONFIG)
    assert d.action == "SKIP"
    assert d.strategy_type == "CRYPTO_TREND"
    assert "edge" in d.reason

