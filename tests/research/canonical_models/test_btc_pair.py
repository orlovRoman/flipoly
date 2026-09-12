import pandas as pd

from polyflip.research.canonical_models.dev_compare import summarize


def _scored():
    return pd.DataFrame([
        {"market_id": "m1", "asset": "BTC", "in_sample": True,
         "decision_at": pd.Timestamp("2026-09-09T10:00:00Z"),
         "side": "UP", "ct_regime": "REVERSION", "ct_sufficient": True,
         "control_trade": True, "ct_trade": True,
         "control_pnl_gross": 4.0, "ct_pnl_gross": 4.0,
         "control_pnl_fee_0.0": 4.0, "control_pnl_fee_0.001": 4.0,
         "control_pnl_fee_0.002": 4.0, "control_pnl_formula": 4.0,
         "ct_pnl_fee_0.0": 4.0, "ct_pnl_fee_0.001": 4.0,
         "ct_pnl_fee_0.002": 4.0, "ct_pnl_formula": 4.0,
         "win": True},
        {"market_id": "m2", "asset": "ETH", "in_sample": True,
         "decision_at": pd.Timestamp("2026-09-09T11:00:00Z"),
         "side": "DOWN", "ct_regime": "UNCERTAIN", "ct_sufficient": False,
         "control_trade": True, "ct_trade": False,
         "control_pnl_gross": 10.0, "ct_pnl_gross": 0.0,
         "control_pnl_fee_0.0": 10.0, "control_pnl_fee_0.001": 10.0,
         "control_pnl_fee_0.002": 10.0, "control_pnl_formula": 10.0,
         "ct_pnl_fee_0.0": 0.0, "ct_pnl_fee_0.001": 0.0,
         "ct_pnl_fee_0.002": 0.0, "ct_pnl_formula": 0.0,
         "win": True},
    ])


def test_btc_pair_is_main_and_descriptive_separated():
    out = summarize(_scored())
    assert out["btc_pair_main"]["n"] == 1
    assert out["btc_pair_main"]["n_l0_eligible"] == 1
    assert out["btc_pair_main"]["n_insufficient_excluded"] == 0
    assert out["btc_pair_main"]["control_gross"] == 4.0
    assert out["btc_pair_main"]["diff_gross"] == 0.0
    # ETH outlier must NOT leak into the main pair
    assert out["control"]["gross"] == 14.0
    assert "descriptive only" in out["control_all_assets_descriptive"]
    assert out["common_sample_check"]["identical_opportunity_ids_pre_ct"] is True
    assert out["common_sample_check"]["scope"].startswith("BTC")


def test_insufficient_data_excluded_not_zeroed():
    df = _scored()
    ins = df.iloc[[0]].copy()
    ins["ct_sufficient"] = False
    ins["ct_trade"] = False
    ins["ct_pnl_gross"] = 0.0
    out = summarize(pd.concat([df, ins], ignore_index=True))
    # m1 dup insufficient: L0 counts 2 BTC opps, L1 scores 1 (no zero leak)
    assert out["btc_pair_main"]["n_l0_eligible"] == 2
    assert out["btc_pair_main"]["n"] == 1
    assert out["btc_pair_main"]["n_insufficient_excluded"] == 1
    assert out["btc_pair_main"]["control_gross"] == 4.0
