from dataclasses import replace

import pytest

from polyflip.crypto.market_regime import (
    AssetRegimeFeatures,
    BasketRegimeFeatures,
    MarketRegimeSnapshot,
)
from polyflip.crypto.market_regime_apply import (
    apply_market_regime_filter,
    apply_regime_veto_gate,
)
from polyflip.trading.trading_config import parse_trading_settings


def _directional_snapshot() -> MarketRegimeSnapshot:
    """A deterministic, strongly upward snapshot used to veto BUY_NO."""
    asset = AssetRegimeFeatures(
        symbol="BTC",
        ret_4h=0.09,
        ret_12h=0.09,
        ret_24h=0.09,
        efficiency_24h=0.9,
        vol_ratio=1.0,
        candle_count=97,
        history_ready=True,
        strength_score=0.8,
    )
    basket = BasketRegimeFeatures(
        median_ret_4h=0.09,
        median_ret_12h=0.09,
        median_ret_24h=0.09,
        breadth_up_4h=1.0,
        breadth_up_12h=1.0,
        breadth_up_24h=1.0,
        dispersion_4h=0.0,
        dispersion_24h=0.0,
        market_efficiency_24h=0.9,
        ready_count=1,
        total_count=1,
        history_ready=True,
        strength=0.8,
    )
    from datetime import datetime, timezone

    return MarketRegimeSnapshot(
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        assets={"BTC": asset},
        basket=basket,
    )


def _cfg(mode: str, version: int = 3):
    return parse_trading_settings(
        {
            "MARKET_REGIME_FILTER_MODE": mode,
            "MARKET_REGIME_FILTER_VERSION": str(version),
            "MARKET_REGIME_VETO_THRESHOLD": "0.15",
            "MARKET_REGIME_EDGE_OVERRIDE_MARGIN": "0.05",
            "MARKET_REGIME_ASSET_WEIGHT": "0.70",
            "MARKET_REGIME_GLOBAL_WEIGHT": "0.30",
        }
    )


def _veto_kwargs(cfg, *, candidate_side="BUY_NO", action=None):
    return {
        "cfg": cfg,
        "snapshot": _directional_snapshot(),
        "asset_symbol": "BTC",
        "candidate_side": candidate_side,
        "candidate_ask": 0.30 if candidate_side == "BUY_NO" else 0.70,
        "net_edge": 0.05,
        "min_edge_used": 0.04,
        "bet_size_usdc": 10.0,
        "action": action or candidate_side,
    }


def test_shadow_records_veto_without_changing_buy():
    outcome = apply_regime_veto_gate(**_veto_kwargs(_cfg("SHADOW")))

    assert outcome.gate_result is not None
    assert outcome.gate_result.would_block is True
    assert outcome.applied is False
    assert outcome.adjusted_action == "BUY_NO"
    assert outcome.adjusted_bet_size == pytest.approx(10.0)
    assert outcome.audit_dict["applied"] is False
    assert outcome.audit_dict["gate"]["would_block"] is True
    assert outcome.audit_dict["policy"] == {}


def test_active_veto_changes_action_only():
    outcome = apply_regime_veto_gate(**_veto_kwargs(_cfg("ACTIVE")))

    assert outcome.applied is True
    assert outcome.adjusted_action == "SKIP"
    assert outcome.adjusted_bet_size == pytest.approx(0.0)
    assert outcome.original_bet_size == pytest.approx(10.0)
    assert outcome.skip_reason.startswith("MRF:V3:")
    assert outcome.audit_dict["applied"] is True
    assert outcome.audit_dict["gate"]["effective_block"] is True


def test_active_pass_preserves_full_bet():
    outcome = apply_regime_veto_gate(
        **_veto_kwargs(_cfg("ACTIVE"), candidate_side="BUY_YES")
    )

    assert outcome.gate_result.would_block is False
    assert outcome.applied is False
    assert outcome.adjusted_action == "BUY_YES"
    assert outcome.adjusted_bet_size == outcome.original_bet_size


def test_dispatcher_keeps_off_and_legacy_paths_separate():
    snapshot = _directional_snapshot()
    off = apply_market_regime_filter(
        cfg=_cfg("OFF"),
        snapshot=snapshot,
        candidate_side="BUY_NO",
        candidate_ask=0.30,
        fresh_yes_price=0.70,
        lgbm_applied=False,
        bet_size_usdc=10.0,
        net_edge=0.05,
        min_edge_used=0.04,
        action="BUY_NO",
        asset_symbol="BTC",
    )
    assert off.gate_result is None
    assert off.adjusted_action == "BUY_NO"

    legacy = apply_market_regime_filter(
        cfg=_cfg("ACTIVE", version=2),
        snapshot=snapshot,
        candidate_side="BUY_NO",
        candidate_ask=0.30,
        fresh_yes_price=0.70,
        lgbm_applied=False,
        bet_size_usdc=10.0,
        net_edge=0.05,
        min_edge_used=0.04,
        action="BUY_NO",
        asset_symbol="BTC",
    )
    assert legacy.gate_result is None
    assert legacy.policy_result is not None
    assert legacy.policy_version == 2


def test_dispatcher_rejects_unknown_version():
    cfg = replace(_cfg("ACTIVE"), mrf_version=4)
    with pytest.raises(ValueError, match="Unsupported MRF policy version"):
        apply_market_regime_filter(
            cfg=cfg,
            snapshot=_directional_snapshot(),
            candidate_side="BUY_YES",
            candidate_ask=0.70,
            fresh_yes_price=0.70,
            lgbm_applied=False,
            bet_size_usdc=10.0,
            net_edge=0.05,
            min_edge_used=0.04,
            action="BUY_YES",
            asset_symbol="BTC",
        )
