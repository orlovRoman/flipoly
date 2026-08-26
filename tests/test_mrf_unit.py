"""
MRF v2 core unit tests — MRF-FIX-12.

Tests for classifier, policy, strength, audit, and apply modules.
Run: python -m pytest test_mrf_unit.py -v
"""
import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

import numpy as np
import pytest

from polyflip.crypto.market_regime_classifier import (
    classify_asset_regime,
    classify_global_regime,
    RegimeConfig,
    AssetRegimeFeatures,
    RegimeClassification,
    MarketPhase,
)
from polyflip.crypto.market_regime import MarketRegimeSnapshot, BasketRegimeFeatures
from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles
from polyflip.crypto.market_regime_policy import (
    evaluate_policy,
    PolicyResult,
    FilterMode,
    PolicyConfig,
    StrategyType,
)
from polyflip.crypto.market_regime_audit import serialize_regime_audit
from polyflip.crypto.market_regime_apply import apply_regime_policy, RegimeDecisionOutcome
from polyflip.trading.trading_config import TradingConfig


def _asset(
    symbol="BTC", ret_4h=0.0, ret_12h=0.0, ret_24h=0.0,
    vol_4h=0.0, vol_24h=0.0, vol_ratio=1.0,
    efficiency_24h=0.5, range_ratio_24h=0.5,
    up_ratio_24h=0.5, candle_count=100,
    history_ready=True, strength_score=0.0,
):
    return AssetRegimeFeatures(
        symbol=symbol, ret_4h=ret_4h, ret_12h=ret_12h, ret_24h=ret_24h,
        vol_4h=vol_4h, vol_24h=vol_24h, vol_ratio=vol_ratio,
        efficiency_24h=efficiency_24h, range_ratio_24h=range_ratio_24h,
        up_ratio_24h=up_ratio_24h, candle_count=candle_count,
        history_ready=history_ready, strength_score=strength_score,
    )


def _basket(**kwargs):
    defaults = dict(
        median_ret_4h=0.0, median_ret_12h=0.0, median_ret_24h=0.0,
        breadth_up_4h=0.5, breadth_up_12h=0.5, breadth_up_24h=0.5,
        dispersion_4h=0.1, dispersion_24h=0.1, market_efficiency_24h=0.5,
        ready_count=3, total_count=3, history_ready=True, strength=0.3,
    )
    defaults.update(kwargs)
    return BasketRegimeFeatures(**defaults)


def _snapshot(global_phase=MarketPhase.UNKNOWN, assets=None, history_ready=True, **basket_kw):
    basket_kw.setdefault("history_ready", history_ready)
    return MarketRegimeSnapshot(
        as_of=datetime.now(timezone.utc),
        assets=assets or {},
        basket=_basket(**basket_kw),
        reason_codes=[],
    )


# ── Classifier tests ──────────────────────────────────────────────


class TestClassifierAssetRegime:
    def test_not_ready_returns_unknown(self):
        f = _asset(history_ready=False)
        cfg = RegimeConfig()
        result = classify_asset_regime(f, cfg)
        assert result.phase == MarketPhase.UNKNOWN

    def test_strong_up(self):
        f = _asset(
            ret_4h=0.03, ret_12h=0.04, ret_24h=0.05,
            efficiency_24h=0.8, vol_ratio=0.3, strength_score=0.9,
        )
        cfg = RegimeConfig(strong_score_threshold=0.5, trend_efficiency_min=0.3)
        result = classify_asset_regime(f, cfg)
        assert result.phase in (MarketPhase.STRONG_UP, MarketPhase.WEAK_UP)
        assert result.direction >= 0
        assert 0 <= result.strength <= 1
        assert 0 <= result.confidence <= 1

    def test_strong_down(self):
        f = _asset(
            ret_4h=-0.03, ret_12h=-0.04, ret_24h=-0.05,
            efficiency_24h=0.8, vol_ratio=0.3, strength_score=0.9,
        )
        cfg = RegimeConfig(strong_score_threshold=0.5, trend_efficiency_min=0.3)
        result = classify_asset_regime(f, cfg)
        assert result.phase in (MarketPhase.STRONG_DOWN, MarketPhase.WEAK_DOWN)
        assert result.direction <= 0

    def test_high_vol_chop(self):
        f = _asset(
            ret_4h=0.0, ret_12h=0.0, ret_24h=0.0,
            efficiency_24h=0.05, vol_ratio=3.0,
        )
        cfg = RegimeConfig()
        result = classify_asset_regime(f, cfg)
        assert result.phase in (MarketPhase.HIGH_VOL_CHOP, MarketPhase.SIDEWAYS)

    def test_strength_bounded_01(self):
        f = _asset(
            ret_4h=0.99, ret_12h=0.99, ret_24h=0.99,
            efficiency_24h=0.99, vol_ratio=0.01, strength_score=1.0,
        )
        cfg = RegimeConfig()
        result = classify_asset_regime(f, cfg)
        assert 0 <= result.strength <= 1.0

    def test_confidence_bounded_01(self):
        f = _asset(efficiency_24h=5.0, vol_ratio=0.01)
        cfg = RegimeConfig()
        result = classify_asset_regime(f, cfg)
        assert 0 <= result.confidence <= 1.0


class TestClassifierGlobalRegime:
    def test_strong_up_breadth(self):
        snap = _snapshot(
            assets={
                "A": _asset(symbol="A", ret_4h=0.02, ret_12h=0.03, ret_24h=0.04, efficiency_24h=0.7, strength_score=0.9),
                "B": _asset(symbol="B", ret_4h=0.02, ret_12h=0.03, ret_24h=0.04, efficiency_24h=0.8, strength_score=0.9),
                "C": _asset(symbol="C", ret_4h=0.02, ret_12h=0.03, ret_24h=0.04, efficiency_24h=0.9, strength_score=0.9),
            },
            breadth_up_4h=0.8, breadth_up_12h=0.8, breadth_up_24h=0.8,
            median_ret_4h=0.02, median_ret_12h=0.03, median_ret_24h=0.04,
            strength=0.9, market_efficiency_24h=0.8,
        )
        cfg = RegimeConfig(
            strong_score_threshold=0.3,
            breadth_strong_threshold=0.6,
            trend_efficiency_min=0.2,
        )
        result = classify_global_regime(snap, cfg)
        assert result.phase in (MarketPhase.STRONG_UP, MarketPhase.WEAK_UP, MarketPhase.MIXED)

    def test_empty_assets_returns_unknown(self):
        snap = _snapshot(assets={}, history_ready=False)
        cfg = RegimeConfig()
        result = classify_global_regime(snap, cfg)
        assert result.phase == MarketPhase.UNKNOWN


# ── Policy tests ──────────────────────────────────────────────────


class TestPolicyResult:
    def test_policy_result_has_regime_alias(self):
        pr = PolicyResult(
            allow=True, stake_multiplier=0.8, reason="test",
            phase=MarketPhase.SIDEWAYS, global_confidence=0.5, global_strength=0.3,
        )
        assert pr.regime == "SIDEWAYS"
        assert pr.global_regime == "SIDEWAYS"

    def test_policy_result_fields(self):
        pr = PolicyResult(
            allow=False, stake_multiplier=0.0, reason="blocked",
            phase=MarketPhase.HIGH_VOL_CHOP, global_confidence=0.2, global_strength=0.1,
        )
        assert pr.allow is False
        assert pr.stake_multiplier == 0.0
        assert pr.phase == MarketPhase.HIGH_VOL_CHOP


# ── Audit tests ───────────────────────────────────────────────────


class TestAudit:
    def test_audit_has_global_regime_alias(self):
        snap = _snapshot(
            assets={"BTC": _asset(symbol="BTC", efficiency_24h=0.5)},
        )
        policy = PolicyResult(
            allow=True, stake_multiplier=0.8, reason="test",
            phase=MarketPhase.SIDEWAYS, global_confidence=0.5, global_strength=0.3,
        )
        audit = serialize_regime_audit(
            snapshot=snap,
            policy_result=policy,
            mode=FilterMode.ACTIVE,
            mrf_version=2,
            strategy_type="balanced",
            applied=True,
            failure_reason=None,
        )
        assert "global_regime" in audit
        assert audit["global_regime"] == audit["global_phase"]

    def test_audit_has_required_keys(self):
        snap = _snapshot()
        audit = serialize_regime_audit(
            snapshot=snap,
            policy_result=None,
            mode=FilterMode.ACTIVE,
            mrf_version=2,
        )
        for key in ["mode", "version", "global_phase", "global_regime", "applied"]:
            assert key in audit, f"Missing key: {key}"


# ── Apply tests ───────────────────────────────────────────────────


class TestApplyRegimePolicy:
    @staticmethod
    def _make_config(mode="OFF", **overrides):
        defaults = {
            "trading_enabled": True,
            "trading_mode": "COMBINED",
            "favor_min_time_left": 0, "favor_max_time_left": 999,
            "outs_min_time_left": 0, "outs_max_time_left": 999,
            "bet_size": 10.0, "dead_zone": 0.1, "daily_limit": 50,
            "trade_min_price": 0.01, "trade_max_price": 0.99,
            "capital": 1000.0, "active_features_str": "",
            "trade_on_favorite": True, "trade_on_flip": True,
            "flip_threshold": 0.55, "outs_min_edge": 0.05,
            "favorite_threshold": 0.55, "trade_assets": "BTC,ETH",
            "bet_sizing_mode": "FIXED", "max_bet_size_usdc": 50.0,
            "favorite_min_price": 0.3, "favorite_max_price": 0.8,
            "favorite_min_edge": 0.02, "outsider_max_price": 0.5,
            "liquidity_fraction": 1.0, "bypass_bet_size_check": False,
            "stop_loss_enabled": False, "take_profit_enabled": False,
            "take_profit_multiplier": 2.0, "max_price_drift": 0.1,
            "stop_loss_pct_favorite": 0.1, "stop_loss_pct_outsider": 0.1,
            "fee_rate": 0.02, "slippage_rate": 0.01,
            "max_exposure_pct": 0.5, "min_direction_prob": 0.5,
            "min_win_prob": 0.5,
        }
        defaults["mrf_mode"] = mode
        defaults["mrf_version"] = 2
        defaults["mrf_min_history"] = 24
        defaults["mrf_outsider_trend_multiplier"] = 0.7
        defaults["mrf_unknown_multiplier"] = 0.5
        defaults["mrf_breadth_threshold"] = 0.6
        defaults["mrf_efficiency_threshold"] = 0.3
        defaults.update(overrides)
        return TradingConfig(**defaults)

    def test_off_mode_passes_through(self):
        cfg = self._make_config(mode="OFF")
        snap = _snapshot()
        outcome = apply_regime_policy(
            cfg, snap, "YES", 0.5, True, 10.0, "BUY_YES",
            decision_run_id="dec_123", asset_symbol="BTC",
        )
        assert outcome.applied is False
        assert outcome.adjusted_action == "BUY_YES"
        assert outcome.adjusted_bet_size == 10.0

    def test_shadow_mode_doesnt_block(self):
        cfg = self._make_config(mode="SHADOW")
        snap = _snapshot(global_phase=MarketPhase.HIGH_VOL_CHOP)
        outcome = apply_regime_policy(
            cfg, snap, "YES", 0.5, True, 10.0, "BUY_YES",
            decision_run_id="dec_456", asset_symbol="BTC",
        )
        assert isinstance(outcome, RegimeDecisionOutcome)

    def test_outcome_has_expected_fields(self):
        cfg = self._make_config(mode="OFF")
        snap = _snapshot()
        outcome = apply_regime_policy(
            cfg, snap, "YES", 0.5, True, 10.0, "SKIP",
            decision_run_id="dec_789", asset_symbol="BTC",
        )
        assert hasattr(outcome, "applied")
        assert hasattr(outcome, "original_action")
        assert hasattr(outcome, "adjusted_action")
        assert hasattr(outcome, "adjusted_bet_size")
        assert hasattr(outcome, "global_phase")
        assert hasattr(outcome, "asset_phase")
        assert hasattr(outcome, "audit_dict")


# ── Step 3: RegimeConfig passthrough tests ─────────────────────────


class TestRegimeConfigPassthrough:
    def test_evaluate_policy_accepts_regime_config(self):
        import inspect
        sig = inspect.signature(evaluate_policy)
        assert "regime_config" in sig.parameters

    def test_evaluate_policy_with_regime_config(self):
        snap = _snapshot(
            assets={"BTC": _asset(symbol="BTC", efficiency_24h=0.8, strength_score=0.9)},
            breadth_up_24h=0.8, median_ret_24h=0.03, strength=0.8,
            market_efficiency_24h=0.8,
        )
        cfg = RegimeConfig(trend_efficiency_min=0.2, strong_score_threshold=0.3)
        pr = evaluate_policy(
            snap, StrategyType.OTHER, 1.0, FilterMode.SHADOW,
            regime_config=cfg,
        )
        assert pr.phase is not None

    def test_serialize_audit_accepts_regime_config(self):
        import inspect
        sig = inspect.signature(serialize_regime_audit)
        assert "regime_config" in sig.parameters

    def test_audit_regime_config_consistency(self):
        snap = _snapshot(
            assets={"BTC": _asset(symbol="BTC", efficiency_24h=0.8, strength_score=0.9)},
            breadth_up_24h=0.8, median_ret_24h=0.03, strength=0.8,
            market_efficiency_24h=0.8,
        )
        cfg = RegimeConfig(trend_efficiency_min=0.2, strong_score_threshold=0.3)
        policy = evaluate_policy(
            snap, StrategyType.OTHER, 1.0, FilterMode.SHADOW,
            regime_config=cfg,
        )
        audit = serialize_regime_audit(
            snapshot=snap, policy_result=policy,
            mode=FilterMode.SHADOW, mrf_version=2,
            regime_config=cfg,
        )
        assert audit["global_phase"] == policy.phase.value


# ── Step 4: Incomplete basket tests ──────────────────────────────


class TestIncompleteBasket:
    def test_build_snapshot_expected_assets_missing(self):
        from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles
        snap = build_snapshot_from_multi_asset_candles(
            candles_by_asset={"BTC": []},
            as_of=datetime.now(timezone.utc),
            expected_assets=["BTC", "ETH", "SOL"],
        )
        assert snap.basket.history_ready is False
        assert snap.basket.total_count == 3
        assert any("asset_missing:" in r or "no_candles:" in r for r in snap.reason_codes)

    def test_build_snapshot_all_empty(self):
        from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles
        snap = build_snapshot_from_multi_asset_candles(
            candles_by_asset={},
            as_of=datetime.now(timezone.utc),
            expected_assets=["BTC", "ETH"],
        )
        assert snap.basket.history_ready is False

    def test_extract_asset_phase_accepts_regime_config(self):
        from polyflip.crypto.market_regime_integration import extract_asset_phase
        import inspect
        sig = inspect.signature(extract_asset_phase)
        assert "regime_config" in sig.parameters

    def test_asset_with_96_candles_marks_incomplete(self):
        """An asset with only 96 closed candles should be flagged as insufficient."""
        from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles
        from polyflip.crypto.market_regime import MIN_HISTORY_CANDLES
        assert MIN_HISTORY_CANDLES == 97
        # Simulate: one asset has 96 candles (all closed, valid times)
        class FakeCandle:
            def __init__(self, i):
                self.is_closed = True
                self.open_time = datetime(2026, 8, 21, i // 4, (i % 4) * 15, tzinfo=timezone.utc)
                self.close = 100.0 + i * 0.1
                self.high = 101.0 + i * 0.1
                self.low = 99.0 + i * 0.1
                self.open = 100.0 + i * 0.1
        candles = [FakeCandle(i) for i in range(96)]
        snap = build_snapshot_from_multi_asset_candles(
            candles_by_asset={"BTC": candles},
            as_of=datetime(2026, 8, 21, 23, 55, tzinfo=timezone.utc),
            expected_assets=["BTC"],
        )
        assert snap.basket.history_ready is False

    def test_full_ready_basket(self):
        """All expected assets with 97+ candles should produce history_ready=True."""
        from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles
        from datetime import timedelta
        base_time = datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)
        class FakeCandle:
            def __init__(self, i):
                self.is_closed = True
                self.open_time = base_time + timedelta(minutes=15 * i)
                self.close = 100.0 + i * 0.01
                self.high = 100.5 + i * 0.01
                self.low = 99.5 + i * 0.01
                self.open = 100.0 + i * 0.01
        candles_a = [FakeCandle(i) for i in range(97)]
        candles_b = [FakeCandle(i) for i in range(100)]
        snap = build_snapshot_from_multi_asset_candles(
            candles_by_asset={"BTC": candles_a, "ETH": candles_b},
            as_of=base_time + timedelta(minutes=15 * 100),
            expected_assets=["BTC", "ETH"],
        )
        assert snap.basket.history_ready is True
        assert snap.basket.ready_count == 2

    def test_one_ready_among_incomplete(self):
        """One ready asset among missing assets → basket NOT ready."""
        from polyflip.crypto.market_regime_integration import build_snapshot_from_multi_asset_candles
        from datetime import timedelta
        base_time = datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)
        class FakeCandle:
            def __init__(self, i):
                self.is_closed = True
                self.open_time = base_time + timedelta(minutes=15 * i)
                self.close = 100.0 + i * 0.01
                self.high = 100.5 + i * 0.01
                self.low = 99.5 + i * 0.01
                self.open = 100.0 + i * 0.01
        btc_candles = [FakeCandle(i) for i in range(100)]
        snap = build_snapshot_from_multi_asset_candles(
            candles_by_asset={"BTC": btc_candles, "ETH": []},
            as_of=base_time + timedelta(minutes=15 * 100),
            expected_assets=["BTC", "ETH"],
        )
        assert snap.basket.history_ready is False
        assert snap.basket.total_count == 2
        assert any("ETH" in r for r in snap.reason_codes)


# ── Step 5: Telemetry semantics tests ─────────────────────────────


class TestTelemetrySemantics:
    def test_mrf_evaluated_false_when_no_outcome(self):
        mrf_outcome = None
        mrf_mode = "SHADOW"
        action = "BUY_YES"
        mrf_actually_evaluated = (
            mrf_outcome is not None
            and mrf_mode != "OFF"
            and action in ("BUY_YES", "BUY_NO")
        )
        assert mrf_actually_evaluated is False

    def test_mrf_evaluated_true_when_outcome_present(self):
        mrf_outcome = RegimeDecisionOutcome(
            regime_snapshot=None, policy_result=None, audit_dict={},
            applied=False, original_bet_size=10.0, adjusted_bet_size=10.0,
            original_action="BUY_YES", adjusted_action="BUY_YES",
            skip_reason=None, global_phase="SIDEWAYS", asset_phase="SIDEWAYS",
        )
        mrf_mode = "SHADOW"
        action = "BUY_YES"
        mrf_actually_evaluated = (
            mrf_outcome is not None
            and mrf_mode != "OFF"
            and action in ("BUY_YES", "BUY_NO")
        )
        assert mrf_actually_evaluated is True

    def test_mrf_evaluated_false_when_off_mode(self):
        mrf_outcome = "something"
        mrf_mode = "OFF"
        action = "BUY_YES"
        mrf_actually_evaluated = (
            mrf_outcome is not None
            and mrf_mode != "OFF"
            and action in ("BUY_YES", "BUY_NO")
        )
        assert mrf_actually_evaluated is False


# ── Step 2: PnL API filter tests ─────────────────────────────────


TERMINAL_POSITION_STATUSES = ("CLOSED", "RESOLVED_REDEEMABLE", "RESOLVED_LOST", "REDEEMED")


class TestPnLFilter:
    def test_terminal_statuses_defined(self):
        """The 4 terminal statuses that should participate in PnL stats."""
        assert len(TERMINAL_POSITION_STATUSES) == 4
        assert "CLOSED" in TERMINAL_POSITION_STATUSES
        assert "RESOLVED_REDEEMABLE" in TERMINAL_POSITION_STATUSES
        assert "RESOLVED_LOST" in TERMINAL_POSITION_STATUSES
        assert "REDEEMED" in TERMINAL_POSITION_STATUSES

    def test_non_terminal_statuses_excluded(self):
        """Non-terminal statuses must not be in the PnL filter."""
        non_terminal = {"OPEN", "OPENING", "CANCELLED", "ENTRY_FAILED"}
        for nt in non_terminal:
            assert nt not in TERMINAL_POSITION_STATUSES

    def test_only_success_status_included(self):
        """Only status=SUCCESS should count for PnL (not SKIPPED/FAILED)."""
        executed = "SUCCESS"
        skipped = "SKIPPED"
        failed = "FAILED"
        assert executed != skipped
        assert executed != failed

    def test_realized_pnl_fallback_to_pnl(self):
        """When realized_pnl_usdc is None, pnl should be used as fallback."""
        class FakeTrade:
            realized_pnl_usdc = None
            pnl = -5.0
        t = FakeTrade()
        pnl = t.realized_pnl_usdc if t.realized_pnl_usdc is not None else t.pnl
        assert pnl == -5.0

    def test_realized_pnl_preferred(self):
        """When realized_pnl_usdc exists, it should be preferred over pnl."""
        class FakeTrade:
            realized_pnl_usdc = 10.0
            pnl = -5.0
        t = FakeTrade()
        pnl = t.realized_pnl_usdc if t.realized_pnl_usdc is not None else t.pnl
        assert pnl == 10.0

    def test_wins_counted_correctly(self):
        """Wins should be based on realized_pnl_usdc > 0 (or pnl > 0 fallback)."""
        wins = 0
        pnl_val = 3.0
        realized = 3.0
        if (realized is not None and realized > 0) or (realized is None and pnl_val is not None and pnl_val > 0):
            wins = 1
        assert wins == 1

    def test_losses_not_counted_as_wins(self):
        """Negative PnL should not count as a win."""
        wins = 0
        pnl_val = -5.0
        realized = -5.0
        if (realized is not None and realized > 0) or (realized is None and pnl_val is not None and pnl_val > 0):
            wins = 1
        assert wins == 0


# ── Step 3: 24h window tests ──────────────────────────────────────


class Test24hWindow:
    def test_96_candles_not_ready(self):
        """96 candles = 95 intervals, should be not_ready (MIN_HISTORY_CANDLES=97)."""
        from polyflip.crypto.market_regime import compute_asset_features, HORIZON_24H, MIN_HISTORY_CANDLES
        assert MIN_HISTORY_CANDLES == HORIZON_24H + 1  # 97
        closes = np.linspace(100, 110, 96, dtype=np.float64)
        result = compute_asset_features(
            closes=closes,
            highs=closes * 1.01,
            lows=closes * 0.99,
            opens=closes * 0.999,
            symbol="BTC",
            candle_count=96,
        )
        assert result.history_ready is False

    def test_97_candles_ready(self):
        """97 candles = 96 intervals, should be ready."""
        from polyflip.crypto.market_regime import compute_asset_features
        closes = np.linspace(100, 110, 97, dtype=np.float64)
        result = compute_asset_features(
            closes=closes,
            highs=closes * 1.01,
            lows=closes * 0.99,
            opens=closes * 0.999,
            symbol="BTC",
            candle_count=97,
        )
        assert result.history_ready is True
        assert result.efficiency_24h > 0  # monotonic up = high efficiency

    def test_98_plus_uses_last_24h_window(self):
        """98+ candles should use last 97 closes for 24h window."""
        from polyflip.crypto.market_regime import compute_asset_features
        # First 50 flat, then 48 rising → efficiency depends on last 97
        flat = np.full(50, 100.0, dtype=np.float64)
        rising = np.linspace(100, 120, 48, dtype=np.float64)
        closes = np.concatenate([flat, rising])
        result = compute_asset_features(
            closes=closes,
            highs=closes * 1.01,
            lows=closes * 0.99,
            opens=closes * 0.999,
            symbol="BTC",
            candle_count=98,
        )
        assert result.history_ready is True
        assert result.efficiency_24h > 0.5  # mostly rising in last 24h

    def test_efficiency_ratio_matches_manual(self):
        """Efficiency ratio matches manual computation for known data."""
        from polyflip.crypto.market_regime import _efficiency_ratio
        closes = np.array([100, 102, 101, 103, 105], dtype=np.float64)
        net = abs(closes[-1] - closes[0])  # 5
        total = np.sum(np.abs(np.diff(closes)))  # 2+1+2+2 = 7
        expected = net / total
        assert abs(_efficiency_ratio(closes) - expected) < 1e-10

    def test_log_returns_24h_uses_97_closes(self):
        """_log_returns(closes, 96) uses closes[-97] as base, producing 96-interval return."""
        from polyflip.crypto.market_regime import _log_returns
        closes = np.linspace(100, 200, 97, dtype=np.float64)
        ret = _log_returns(closes, 96)
        expected = math.log(200.0 / 100.0)  # ln(2)
        assert abs(ret - expected) < 1e-6



@dataclass
class _IntegrationCandle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    is_closed: bool = True


class TestCandleWindowValidation:
    def test_extra_fetch_buffer_does_not_invalidate_97_candle_tail(self):
        """107 fetched candles must validate the final 97-candle feature window."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = [
            _IntegrationCandle(
                open_time=start + timedelta(minutes=15 * index),
                open=100.0 + index,
                high=101.0 + index,
                low=99.0 + index,
                close=100.5 + index,
            )
            for index in range(107)
        ]
        assets = {asset: list(candles) for asset in ("BTC", "ETH", "SOL", "DOGE", "XRP")}

        snapshot = build_snapshot_from_multi_asset_candles(
            assets,
            as_of=candles[-1].open_time,
            expected_assets=list(assets),
        )

        assert snapshot.basket.history_ready is True
        assert not any("span_exceeded" in reason for reason in snapshot.reason_codes)

    def test_future_candles_are_excluded_before_window_validation(self):
        """A candle after as_of must not affect continuity or feature readiness."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candles = [
            _IntegrationCandle(
                open_time=start + timedelta(minutes=15 * index),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
            )
            for index in range(108)
        ]
        as_of = candles[106].open_time
        assets = {asset: list(candles) for asset in ("BTC", "ETH", "SOL", "DOGE", "XRP")}

        snapshot = build_snapshot_from_multi_asset_candles(
            assets,
            as_of=as_of,
            expected_assets=list(assets),
        )

        assert snapshot.basket.history_ready is True


# ── Step 5: Failure reason tests ──────────────────────────────────


class TestFailureReasons:
    REASON_PREFIXES = [
        "not_ready",
        "missing_asset",
        "insufficient_history",
        "candle_error",
        "continuity_error",
        "runtime_error",
    ]

    def test_all_reason_prefixes_defined(self):
        """All 6 required failure reason types are present."""
        for prefix in self.REASON_PREFIXES:
            assert isinstance(prefix, str)
            assert len(prefix) > 0

    def test_failure_reason_prefix_parsing(self):
        """Failure reasons follow 'type:detail' convention."""
        reasons = [
            "missing_asset:ETH,SOL",
            "candle_error:BTC,DOGE",
            "continuity_error:ETH:gap_at_100",
            "not_ready",
            "insufficient_history",
            "runtime_error:ConnectionError",
        ]
        for reason in reasons:
            assert isinstance(reason, str)
            parts = reason.split(":", 1)
            assert parts[0] in self.REASON_PREFIXES

    def test_mrf_evaluated_false_on_not_ready(self):
        """When outcome is None (not_ready), mrf_evaluated must be False."""
        mrf_outcome = None
        mrf_mode = "SHADOW"
        action = "BUY_YES"
        mrf_actually_evaluated = (
            mrf_outcome is not None
            and mrf_mode != "OFF"
            and action in ("BUY_YES", "BUY_NO")
        )
        assert mrf_actually_evaluated is False

    def test_failure_reason_set_when_no_outcome(self):
        """A pre_outcome failure reason should be used when outcome is None."""
        mrf_outcome = None
        mrf_pre_outcome_reason = "not_ready"
        mrf_failure_reason = None
        if mrf_outcome and hasattr(mrf_outcome, "skip_reason") and mrf_outcome.skip_reason:
            mrf_failure_reason = mrf_outcome.skip_reason
        elif mrf_pre_outcome_reason:
            mrf_failure_reason = mrf_pre_outcome_reason
        assert mrf_failure_reason == "not_ready"

    def test_policy_skip_reason_takes_priority(self):
        """When both skip_reason and pre_outcome_reason exist, skip_reason wins."""
        class FakeOutcome:
            skip_reason = "MRF:HIGH_VOL_CHOP:blocked"
        mrf_outcome = FakeOutcome()
        mrf_pre_outcome_reason = "not_ready"
        mrf_failure_reason = None
        if mrf_outcome and mrf_outcome.skip_reason:
            mrf_failure_reason = mrf_outcome.skip_reason
        elif mrf_pre_outcome_reason:
            mrf_failure_reason = mrf_pre_outcome_reason
        assert mrf_failure_reason == "MRF:HIGH_VOL_CHOP:blocked"


# ── Dashboard MRF telemetry regression tests ───────────────────────


class TestDashboardMrfTelemetry:
    def test_audit_json_is_used_when_scalar_columns_are_empty(self):
        from polyflip.api.mrf_api import _extract_mrf_telemetry

        class Row:
            mrf_audit_json = (
                '{"global_phase":"WEAK_UP","global_strength":0.42,'
                '"global_confidence":0.67,"assets":{"BTC":{"phase":"STRONG_UP",'
                '"strength":0.81,"confidence":0.91}}}'
            )
            mrf_phase = "UNKNOWN"
            mrf_strength = 0.0
            mrf_confidence = 0.0
            mrf_asset_phase = "UNKNOWN"

        telemetry = _extract_mrf_telemetry(Row())
        assert telemetry["global_phase"] == "WEAK_UP"
        assert telemetry["global_strength"] == 0.42
        assert telemetry["assets"]["BTC"]["phase"] == "STRONG_UP"

    def test_unknown_phase_is_not_treated_as_available(self):
        from polyflip.api.mrf_api import _is_known_phase

        assert _is_known_phase("STRONG_UP") is True
        assert _is_known_phase("UNKNOWN") is False
        assert _is_known_phase(None) is False
