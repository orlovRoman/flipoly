import numpy as np
import pandas as pd

from polyflip.crypto.feature_builder import build_crypto_features, CRYPTO_FEATURE_COLUMNS
from polyflip.crypto.feature_sets import get_feature_set, validate_feature_schema
from polyflip.crypto.trainer import _controlled_lgbm_candidates, _make_lgbm


def test_lgbm_def_feature_sets_have_expected_contracts():
    stable = get_feature_set("D")
    strike = get_feature_set("E")
    context = get_feature_set("F")

    assert stable.features
    assert "strike_gap_pct" not in stable.features
    assert {"strike_gap_pct", "log_moneyness"} <= set(strike.features)
    assert {"pm_momentum_5m", "pm_volume_5m", "pm_spread_pct", "pm_quote_pressure"} <= set(context.features)
    assert validate_feature_schema(context.features) == context.features


def test_market_context_features_are_decision_time_values():
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(110):
        ts = start + pd.Timedelta(minutes=15 * i)
        price = 100.0 + i * 0.01
        rows.append({
            "open_time": ts,
            "open": price,
            "high": price + 0.1,
            "low": price - 0.1,
            "close": price + 0.02,
            "volume": 1000.0,
            "taker_buy_volume": 520.0,
        })

    vector = build_crypto_features(
        pd.DataFrame(rows),
        underlying_price=100.0,
        market_context={
            "price_velocity": 0.012,
            "volume_5min": 42.0,
            "current_yes_price": 0.62,
            "current_spread": 0.04,
            "best_bid": 0.60,
            "best_ask": 0.64,
        },
    )
    values = dict(zip(CRYPTO_FEATURE_COLUMNS, vector.features[0]))
    assert values["pm_momentum_5m"] == 0.012
    assert values["pm_volume_5m"] == 42.0
    assert values["pm_spread_pct"] > 0.0
    assert values["pm_quote_pressure"] == 0.12
    assert values["pm_best_bid"] == 0.60
    assert values["pm_best_ask"] == 0.64


def test_lgbm_subsample_frequency_enables_row_bagging():
    model = _make_lgbm(subsample=0.8)
    assert model.get_params()["subsample"] == 0.8
    assert model.get_params()["subsample_freq"] == 1


def test_controlled_candidates_are_bounded_and_deterministic():
    candidates = _controlled_lgbm_candidates({"learning_rate": 0.05, "num_leaves": 15}, 20)
    assert len(candidates) == 4
    assert candidates[0]["learning_rate"] == 0.05
    assert candidates[0]["num_leaves"] == 15
