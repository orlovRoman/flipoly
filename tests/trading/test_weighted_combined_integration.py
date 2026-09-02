from dataclasses import replace

from polyflip.crypto.predictor import CryptoSignal
from polyflip.trading.combined_voting import evaluate_combined_entry
from polyflip.trading.policy_artifact import create_policy_artifact, save_policy_artifact
from polyflip.trading.trading_config import parse_trading_settings
from polyflip.trading.weighted_policy import WeightedPolicyConfig


def _signal() -> CryptoSignal:
    return CryptoSignal(
        symbol="BTCUSDT",
        p_up=0.80,
        p_down=0.20,
        direction="UP",
        signal_strength=0.30,
        strike=65000.0,
        threshold_up=0.55,
        threshold_down=0.45,
        model_version=7,
        features_ok=True,
        risk_vetoed=False,
        regime="MID_VOL",
        status="READY",
    )


def _weighted_cfg(mode: str):
    return parse_trading_settings(
        {
            "TRADING_POLICY_MODE": mode,
            "LIGHTGBM_DECISION_MODE": "ACTIVE",
            "TRADE_ON_FAVORITE": "true",
            "TRADE_ON_FLIP": "true",
            "MIN_WIN_PROB": "0.50",
            "FAVORITE_MIN_EDGE": "0.00",
            "FAVORITE_MIN_PRICE": "0.05",
            "OUTS_MIN_EDGE": "0.00",
            "COMBINED_COST_BUFFER": "0.00",
            "WEIGHTED_FEE_RATE": "0.00",
            "WEIGHTED_SLIPPAGE_RATE": "0.00",
        }
    )


def _evaluate(cfg):
    return evaluate_combined_entry(
        crypto_sig=_signal(),
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="BTC_mid_vol",
        entry_model_version=4,
        entry_model_source="PHASE",
        p_flip=0.20,
        fresh_yes_price=0.55,
        yes_ask=0.54,
        no_ask=0.46,
        cost_buffer=0.0,
        time_left_sec=300.0,
        cfg=cfg,
    )


def test_weighted_active_replaces_hard_direction_consensus():
    result = _evaluate(_weighted_cfg("WEIGHTED_ACTIVE"))

    assert result.action == "BUY_YES"
    assert result.consensus_type == "WEIGHTED_SCORE"
    assert result.weighted_policy_mode == "WEIGHTED_ACTIVE"
    assert result.weighted_selected_side == "BUY_YES"
    assert result.direction_status == "WEIGHTED_LGBM_USED"
    assert result.weighted_benchmark_json is not None
    assert set(result.weighted_benchmark_json) == {
        "MARKET_ONLY",
        "MARKET_LOGREG",
        "MARKET_LGBM",
        "FULL_WEIGHTED",
        "FULL_WEIGHTED_MRF",
        "OUTSIDER_AGREE_ONLY",
        "LEGACY",
    }
    # Market prior is normalized from asks: 0.54 / (0.54 + 0.46).
    # LogReg and LGBM then add 0.05 log-odds residuals from that prior.
    assert result.weighted_p_final_yes == 0.57026632
    assert result.weighted_cost_per_share == 0.0


def test_weighted_active_ignores_legacy_probability_edge_and_sizing_gates():
    cfg = replace(
        _weighted_cfg("WEIGHTED_ACTIVE"),
        min_win_prob=0.99,
        favorite_min_edge=0.99,
        outs_min_edge=0.99,
        bet_sizing_mode="edge_scaled",
        weighted_min_net_ev_favorite=0.02,
        weighted_fixed_bet_usdc=1.0,
    )
    result = _evaluate(cfg)

    assert result.action == "BUY_YES"
    assert result.bet_size_usdc == 1.0


def test_weighted_shadow_records_score_without_changing_legacy_action():
    result = _evaluate(_weighted_cfg("WEIGHTED_SHADOW"))

    assert result.action == "BUY_YES"
    assert result.consensus_type != "WEIGHTED_SCORE"
    assert result.weighted_policy_mode == "WEIGHTED_SHADOW"
    assert result.weighted_selected_side == "BUY_YES"
    assert result.weighted_p_final_yes is not None


def test_weighted_runtime_uses_half_spread_and_fee_schedule_role():
    cfg = replace(
        _weighted_cfg("WEIGHTED_ACTIVE"),
        weighted_execution_role="MAKER",
        weighted_maker_fee_rate=0.01,
    )
    result = evaluate_combined_entry(
        crypto_sig=_signal(),
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="BTC_mid_vol",
        entry_model_version=4,
        entry_model_source="PHASE",
        p_flip=0.20,
        fresh_yes_price=0.55,
        yes_ask=0.54,
        no_ask=0.46,
        cost_buffer=0.0,
        time_left_sec=300.0,
        cfg=cfg,
        spread=0.04,
        spread_cost=0.02,
        weighted_maker_fee_rate=0.01,
        weighted_taker_only=True,
        weighted_fee_rate=0.07,
    )

    assert result.weighted_policy_mode == "WEIGHTED_ACTIVE"
    assert result.weighted_execution_role == "TAKER"
    assert result.weighted_spread_per_share == 0.02
    assert result.weighted_maker_fee_rate == 0.01
    assert result.weighted_taker_fee_per_share is not None


def test_weighted_active_loads_policy_artifact_and_lower_bound_sizing(tmp_path):
    artifact = create_policy_artifact(
        version="runtime-v1",
        created_at="2026-01-01T00:00:00+00:00",
        training_window={"rows": 100, "fingerprint": "fixture"},
        stacker=None,
        policy_config=WeightedPolicyConfig(
            market_weight=0.75,
            logreg_weight=0.15,
            lgbm_weight=0.10,
            fee_rate=0.0,
            slippage_rate=0.0,
        ),
        thresholds={"min_net_ev_favorite": 0.0},
    )
    artifact_path = tmp_path / "weighted-policy.json"
    save_policy_artifact(artifact_path, artifact)
    cfg = replace(
        _weighted_cfg("WEIGHTED_ACTIVE"),
        weighted_policy_artifact_path=str(artifact_path),
        weighted_sizing_mode="LOWER_BOUND_KELLY",
        weighted_standard_error=0.0,
        weighted_kelly_fraction=1.0,
        weighted_min_net_ev_favorite=0.0,
        weighted_fixed_bet_usdc=1.0,
        weighted_size_cap_usdc=3.0,
    )

    result = _evaluate(cfg)

    assert result.action == "BUY_YES"
    assert result.weighted_policy_id == artifact.artifact_id
    assert result.weighted_market_weight == 0.75
    assert result.weighted_edge_lower_bound is not None
    assert 0.0 < result.weighted_size_multiplier <= 1.0
    assert result.bet_size_usdc == result.weighted_size_multiplier


def test_weighted_active_stepped_edge_sizing_uses_lower_bound_and_cap():
    cfg = replace(
        _weighted_cfg("WEIGHTED_ACTIVE"),
        weighted_sizing_mode="STEPPED_EDGE",
        weighted_standard_error=0.0,
        weighted_fixed_bet_usdc=1.0,
        weighted_size_cap_usdc=3.0,
    )
    result = _evaluate(cfg)

    assert result.action == "BUY_YES"
    assert result.weighted_edge_lower_bound is not None
    assert result.weighted_size_multiplier == 1.5
    assert result.bet_size_usdc == 1.5


def test_weighted_sizing_parser_accepts_stepped_edge_mode():
    cfg = parse_trading_settings({"WEIGHTED_SIZING_MODE": "STEPPED_EDGE"})
    assert cfg.weighted_sizing_mode == "STEPPED_EDGE"




def test_weighted_active_rejects_missing_policy_artifact(tmp_path):
    cfg = replace(
        _weighted_cfg("WEIGHTED_ACTIVE"),
        weighted_policy_artifact_path=str(tmp_path / "missing-policy.json"),
    )

    result = _evaluate(cfg)

    assert result.action == "SKIP"
    assert "POLICY_ARTIFACT_INVALID" in result.reason


def test_weighted_mrf_stake_mode_does_not_double_adjust_probability():
    cfg = replace(
        _weighted_cfg("WEIGHTED_ACTIVE"),
        weighted_mrf_application="STAKE",
        weighted_mrf_sizing_gamma=0.5,
        weighted_fixed_bet_usdc=1.0,
    )
    result = evaluate_combined_entry(
        crypto_sig=_signal(),
        market_phase="mid_vol",
        entry_requested_key="BTC_mid_vol",
        entry_model_key="BTC_mid_vol",
        entry_model_version=4,
        entry_model_source="PHASE",
        p_flip=0.20,
        fresh_yes_price=0.55,
        yes_ask=0.54,
        no_ask=0.46,
        cost_buffer=0.0,
        time_left_sec=300.0,
        cfg=cfg,
        mrf_evidence=-1.0,
    )
    assert result.action == "BUY_YES"
    assert result.weighted_mrf_contribution_logodds == 0.0
    assert result.weighted_size_multiplier == 0.5
    assert result.bet_size_usdc == 0.5
