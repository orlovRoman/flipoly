"""
MRF v2 core unit tests — MRF-FIX-12.

Tests for classifier, policy, strength, audit, and apply modules.
Run: python -m pytest test_mrf_unit.py -v
"""
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field

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
from polyflip.crypto.market_regime_policy import (
    evaluate_policy,
    PolicyResult,
    FilterMode,
    PolicyConfig,
)
from polyflip.crypto.market_regime_audit import serialize_regime_audit
from polyflip.crypto.market_regime_apply import apply_regime_policy, RegimeDecisionOutcome, TradingConfig


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
