import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polyflip.crypto.market_regime_classifier import MarketPhase
from polyflip.crypto.market_regime_policy import PolicyResult, RegimeGateResult
from polyflip.crypto.predictor import CryptoSignal
from polyflip.crypto.market_regime_apply import RegimeDecisionOutcome
from polyflip.trading.decision_runners import _apply_mrf_filter, decide_combined_mode
from polyflip.trading.trading_config import parse_trading_settings


def _make_runner_context(mode: str):
    db_session = AsyncMock()
    api_client = AsyncMock()
    api_client.get_market_prices.return_value = {
        "current_yes_price": "0.60",
        "best_ask": "0.62",
        "current_spread": "0.02",
    }
    market = SimpleNamespace(
        market_id=501,
        asset="BTC",
        yes_token_id="yes-token",
        no_token_id="no-token",
        volume_5min=500.0,
        underlying_price=65000.0,
        current_spread=0.02,
    )
    cfg = parse_trading_settings(
        {
            "LIGHTGBM_DECISION_MODE": "ACTIVE",
            "COMBINED_COST_BUFFER": "0.03",
            "MARKET_REGIME_FILTER_MODE": mode,
            "MARKET_REGIME_FILTER_VERSION": "3",
        }
    )
    models_cache = SimpleNamespace(
        models={"BTC_leaning": MagicMock()},
        versions={"BTC_leaning": 4},
        features={"BTC_leaning": ["f1", "f2"]},
        eces={"BTC_leaning": 0.0},
    )
    signal = CryptoSignal(
        symbol="BTCUSDT",
        model_key="BTCUSDT_mid_vol",
        p_up=0.85,
        p_down=0.15,
        direction="UP",
        signal_strength=0.7,
        strike=65000.0,
        threshold_up=0.55,
        threshold_down=0.45,
        model_version=10,
        features_ok=True,
        risk_vetoed=False,
        regime="MID_VOL",
        status="OK",
    )
    gate = RegimeGateResult(
        would_block=True,
        reason="regime_veto",
        candidate_direction=1.0,
        asset_phase=MarketPhase.STRONG_DOWN,
        global_phase=MarketPhase.STRONG_DOWN,
        asset_strength=0.8,
        asset_confidence=0.8,
        global_strength=0.8,
        global_confidence=0.8,
        asset_evidence=-0.64,
        global_evidence=-0.64,
        regime_evidence=-0.64,
        net_edge=0.08,
        min_edge_used=0.05,
        edge_margin=0.03,
        veto_threshold=0.15,
        edge_override_margin=0.05,
    )
    active = mode == "ACTIVE"
    outcome = RegimeDecisionOutcome(
        regime_snapshot=None,
        policy_result=None,
        audit_dict={
            "mode": mode,
            "version": 3,
            "global_phase": "STRONG_DOWN",
            "policy": {},
            "gate": {"would_block": True, "reason": "regime_veto"},
        },
        applied=active,
        original_bet_size=10.0,
        adjusted_bet_size=0.0 if active else 10.0,
        original_action="BUY_YES",
        adjusted_action="SKIP" if active else "BUY_YES",
        skip_reason="MRF:V3:STRONG_DOWN:STRONG_DOWN:regime_veto" if active else None,
        global_phase="STRONG_DOWN",
        asset_phase="STRONG_DOWN",
        gate_result=gate,
        policy_version=3,
    )
    return db_session, api_client, market, cfg, models_cache, signal, outcome


def test_mrf_v3_active_invalid_candidate_side_fails_closed():
    cfg = parse_trading_settings(
        {
            "MARKET_REGIME_FILTER_MODE": "ACTIVE",
            "MARKET_REGIME_FILTER_VERSION": "3",
        }
    )

    action, bet, audit, outcome, failure = asyncio.run(
        _apply_mrf_filter(
            db_session=AsyncMock(),
            cfg=cfg,
            asset_upper="BTC",
            binance_symbol="BTCUSDT",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            candidate_side=None,
            fresh_yes_price=0.7,
            candidate_ask=0.7,
            bet_size_usdc=10.0,
            net_edge=0.1,
            min_edge_used=0.05,
            action="BUY_YES",
        )
    )

    assert (action, bet) == ("SKIP", 0.0)
    assert outcome is None
    assert failure == "invalid_candidate_side:NONE"
    assert audit["failure_reason"] == failure


@pytest.mark.parametrize("mode, expected_action, expected_applied", [
    ("SHADOW", "BUY_YES", False),
    ("ACTIVE", "SKIP", True),
])
def test_decision_runner_records_v3_gate_without_resizing_shadow(
    mode, expected_action, expected_applied,
):
    (
        db_session,
        api_client,
        market,
        cfg,
        models_cache,
        signal,
        outcome,
    ) = _make_runner_context(mode)
    with patch(
        "polyflip.trading.decision_runners._fetch_lgbm_signal",
        AsyncMock(return_value=signal),
    ), patch(
        "polyflip.trading.decision_runners.infer_flip_for_market",
        AsyncMock(return_value=0.15),
    ), patch(
        "polyflip.trading.decision_runners._apply_mrf_filter",
        AsyncMock(return_value=(
            outcome.adjusted_action,
            outcome.adjusted_bet_size,
            outcome.audit_dict,
            outcome,
            None,
        )),
    ), patch(
        "polyflip.trading.decision_runners.log_funnel",
        AsyncMock(),
    ) as log_funnel:
        result = asyncio.run(
            decide_combined_mode(
                db_session=db_session,
                api_client=api_client,
                market=market,
                cfg=cfg,
                raw_settings={},
                models_cache=models_cache,
                crypto_predictor=MagicMock(),
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                time_left_sec=300.0,
                execution_mode="PAPER",
            )
        )

    assert result.decision_obj.action == expected_action
    assert log_funnel.await_count == 1
    fields = log_funnel.await_args.kwargs
    assert fields["mrf_policy_version"] == 3
    assert fields["mrf_gate_would_block"] is True
    assert fields["mrf_gate_reason"] == "regime_veto"
    assert fields["mrf_multiplier"] is None
    assert fields["mrf_applied"] is expected_applied
    assert fields["mrf_original_action"] == "BUY_YES"
    assert fields["mrf_final_action"] == expected_action


def test_decision_runner_off_does_not_call_mrf_filter():
    db_session, api_client, market, cfg, models_cache, signal, _ = _make_runner_context("OFF")
    with patch(
        "polyflip.trading.decision_runners._fetch_lgbm_signal",
        AsyncMock(return_value=signal),
    ), patch(
        "polyflip.trading.decision_runners.infer_flip_for_market",
        AsyncMock(return_value=0.15),
    ), patch(
        "polyflip.trading.decision_runners._apply_mrf_filter",
        AsyncMock(side_effect=AssertionError("MRF must be short-circuited")),
    ), patch(
        "polyflip.trading.decision_runners.log_funnel",
        AsyncMock(),
    ):
        result = asyncio.run(
            decide_combined_mode(
                db_session=db_session,
                api_client=api_client,
                market=market,
                cfg=cfg,
                raw_settings={},
                models_cache=models_cache,
                crypto_predictor=MagicMock(),
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                time_left_sec=300.0,
                execution_mode="PAPER",
            )
        )

    assert result.decision_obj.action == "BUY_YES"


def test_decision_runner_v2_keeps_legacy_multiplier_separate_from_gate_fields():
    (
        db_session,
        api_client,
        market,
        cfg,
        models_cache,
        signal,
        _,
    ) = _make_runner_context("ACTIVE")
    cfg = replace(cfg, mrf_version=2)
    policy = PolicyResult(
        allow=True,
        stake_multiplier=0.5,
        reason="legacy_multiplier",
        phase=MarketPhase.MIXED,
        global_confidence=0.4,
        global_strength=0.2,
    )
    outcome = RegimeDecisionOutcome(
        regime_snapshot=None,
        policy_result=policy,
        audit_dict={
            "mode": "ACTIVE",
            "version": 2,
            "global_phase": "MIXED",
            "policy": {"multiplier": 0.5},
        },
        applied=True,
        original_bet_size=10.0,
        adjusted_bet_size=5.0,
        original_action="BUY_YES",
        adjusted_action="BUY_YES",
        global_phase="MIXED",
        asset_phase="MIXED",
        policy_version=2,
    )
    with patch(
        "polyflip.trading.decision_runners._fetch_lgbm_signal",
        AsyncMock(return_value=signal),
    ), patch(
        "polyflip.trading.decision_runners.infer_flip_for_market",
        AsyncMock(return_value=0.15),
    ), patch(
        "polyflip.trading.decision_runners._apply_mrf_filter",
        AsyncMock(return_value=(
            outcome.adjusted_action,
            outcome.adjusted_bet_size,
            outcome.audit_dict,
            outcome,
            None,
        )),
    ), patch(
        "polyflip.trading.decision_runners.log_funnel",
        AsyncMock(),
    ) as log_funnel:
        result = asyncio.run(
            decide_combined_mode(
                db_session=db_session,
                api_client=api_client,
                market=market,
                cfg=cfg,
                raw_settings={},
                models_cache=models_cache,
                crypto_predictor=MagicMock(),
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                time_left_sec=300.0,
                execution_mode="PAPER",
            )
        )

    assert result.decision_obj.action == "BUY_YES"
    assert result.decision_obj.bet_size_usdc == pytest.approx(5.0)
    fields = log_funnel.await_args.kwargs
    assert fields["mrf_policy_version"] == 2
    assert fields["mrf_multiplier"] == pytest.approx(0.5)
    assert fields["mrf_regime_evidence"] is None
