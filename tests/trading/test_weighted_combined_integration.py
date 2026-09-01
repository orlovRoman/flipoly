from dataclasses import replace

from polyflip.crypto.predictor import CryptoSignal
from polyflip.trading.combined_voting import evaluate_combined_entry
from polyflip.trading.trading_config import parse_trading_settings


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
