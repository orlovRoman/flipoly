import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.threshold_optimizer import (
    ThresholdPair,
    evaluate_thresholded_polymarket,
    optimize_joint_thresholds,
)

def test_evaluate_thresholded_fast_vs_slow():
    np.random.seed(42)
    N = 1000

    frame = pd.DataFrame({
        "market_id": np.arange(N).astype(str),
        "asset": ["BTC"] * N,
        "target": np.random.choice([0, 1], size=N),
        "recorded_at": pd.date_range("2023-01-01", periods=N, freq="h").astype(str),
        "time_left_min": np.random.uniform(10, 1000, size=N),
        "vol_regime": ["NORMAL"] * N,
    })

    raw_scores = np.random.uniform(0, 1, size=N)
    raw_scores[0] = np.nan # some missing oof
    calibrated_scores = raw_scores.copy()
    calibrated_scores[1] = 1.5 # invalid calibrated oof

    quotes = pd.DataFrame({
        "market_id": np.arange(800).astype(str),
        "yes_ask": np.random.uniform(0.1, 0.9, size=800),
        "no_ask": np.random.uniform(0.1, 0.9, size=800),
    })

    # Intentionally make some quotes invalid
    quotes.loc[10, "yes_ask"] = np.nan
    quotes.loc[20, "no_ask"] = np.nan

    lower = 0.3
    upper = 0.7

    res_slow = evaluate_thresholded_polymarket(
        frame, raw_scores, calibrated_scores, quotes, lower=lower, upper=upper, strategy_branch="COMBINED"
    )

    from polyflip.crypto.threshold_optimizer import ThresholdEvaluatorFast
    evaluator = ThresholdEvaluatorFast(
        frame, raw_scores, calibrated_scores, quotes, strategy_branch="COMBINED"
    )
    res_fast = evaluator.evaluate(lower=lower, upper=upper)

    assert res_slow["n_markets"] == N

    # Compare identical keys except trades
    keys = set(res_slow.keys()) - {"trades"}
    for key in keys:
        assert res_fast[key] == res_slow[key], f"Mismatch for key {key}: {res_fast[key]} != {res_slow[key]}"

    # Compare trades individually (time order should match)
    assert len(res_fast["trades"]) == len(res_slow["trades"])
    for tf, ts in zip(res_fast["trades"], res_slow["trades"]):
        assert tf == ts

def test_evaluate_summary_only():
    np.random.seed(42)
    N = 100
    frame = pd.DataFrame({
        "market_id": np.arange(N).astype(str),
        "asset": ["BTC"] * N,
        "target": np.random.choice([0, 1], size=N),
        "recorded_at": pd.date_range("2023-01-01", periods=N, freq="h").astype(str),
        "time_left_min": np.random.uniform(10, 1000, size=N),
        "vol_regime": ["NORMAL"] * N,
    })

    raw_scores = np.random.uniform(0, 1, size=N)
    calibrated_scores = raw_scores.copy()

    quotes = pd.DataFrame({
        "market_id": np.arange(80).astype(str),
        "yes_ask": np.random.uniform(0.1, 0.9, size=80),
        "no_ask": np.random.uniform(0.1, 0.9, size=80),
    })

    from polyflip.crypto.threshold_optimizer import ThresholdEvaluatorFast
    evaluator = ThresholdEvaluatorFast(
        frame, raw_scores, calibrated_scores, quotes, strategy_branch="COMBINED"
    )

    res_full = evaluator.evaluate(0.3, 0.7, summary_only=False)
    res_summary = evaluator.evaluate(0.3, 0.7, summary_only=True)

    # Assert missing keys in summary
    assert "trades" not in res_summary
    assert "slices" not in res_summary

    assert "oot_windows" in res_summary
    assert "oot_windows" in res_full
    assert len(res_summary["oot_windows"]) == len(res_full["oot_windows"])
    for sum_win, full_win in zip(res_summary["oot_windows"], res_full["oot_windows"]):
        assert sum_win["window"] == full_win["window"]
        assert sum_win["n_trades"] == full_win["n_trades"]
        assert np.isclose(sum_win["net_profit"], full_win["net_profit"])
        assert np.isclose(sum_win["max_drawdown_usdc"], full_win["max_drawdown_usdc"])
        assert np.isclose(sum_win["max_drawdown_pct"], full_win["max_drawdown_pct"])

    # Assert all summary keys match the full result
    for key in res_summary:
        if key == "oot_windows":
            continue
        assert res_summary[key] == res_full[key], f"Mismatch for key {key}"


def test_evaluate_duplicate_market_ids_preserves_row_alignment():
    frame = pd.DataFrame({
        "market_id": ["duplicate-market", "duplicate-market"],
        "asset": ["BTC", "BTC"],
        "target": [1, 0],
        "recorded_at": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z"],
        "time_left_min": [30.0, 20.0],
        "vol_regime": ["NORMAL", "NORMAL"],
    })
    raw_scores = np.array([0.95, 0.50])
    calibrated_scores = raw_scores.copy()
    quotes = pd.DataFrame({
        "market_id": ["duplicate-market"],
        "best_ask": [0.40],
        "best_bid": [0.35],
    })

    res_slow = evaluate_thresholded_polymarket(
        frame,
        raw_scores,
        calibrated_scores,
        quotes,
        lower=0.30,
        upper=0.80,
        strategy_branch="COMBINED",
    )
    from polyflip.crypto.threshold_optimizer import ThresholdEvaluatorFast
    res_fast = ThresholdEvaluatorFast(
        frame,
        raw_scores,
        calibrated_scores,
        quotes,
        strategy_branch="COMBINED",
    ).evaluate(0.30, 0.80)

    assert res_slow["n_signals"] == 1
    assert res_slow["n_trades"] == 1
    assert res_fast["n_trades"] == 1
    for key in ("n_signals", "n_trades", "net_profit", "max_drawdown_usdc", "max_drawdown_pct"):
        assert np.isclose(res_slow[key], res_fast[key])
    assert res_slow["trades"] == res_fast["trades"]

def test_evaluate_duplicate_market_id():
    frame = pd.DataFrame({
        "market_id": ["DUPLICATE", "DUPLICATE"],
        "asset": ["BTC", "BTC"],
        "target": [1, 0],
        "recorded_at": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        "time_left_min": [100.0, 40.0],
        "vol_regime": ["NORMAL", "NORMAL"],
    })

    # First row signals UP, second row is missing OOF (NONE)
    raw_scores = np.array([0.9, np.nan])
    calibrated_scores = raw_scores.copy()

    quotes = pd.DataFrame({
        "market_id": ["DUPLICATE"],
        "best_ask": [0.4],
        "best_bid": [0.4],
    })

    lower, upper = 0.3, 0.7

    res_slow = evaluate_thresholded_polymarket(
        frame, raw_scores, calibrated_scores, quotes, lower=lower, upper=upper, strategy_branch="COMBINED"
    )

    from polyflip.crypto.threshold_optimizer import ThresholdEvaluatorFast
    evaluator = ThresholdEvaluatorFast(
        frame, raw_scores, calibrated_scores, quotes, strategy_branch="COMBINED"
    )
    res_fast = evaluator.evaluate(lower=lower, upper=upper)

    assert res_slow["n_markets"] == 2
    assert res_slow["n_signals"] == 1
    assert res_slow["up_signals"] == 1
    assert res_slow["n_trades"] == 1

    # Compare identical keys except trades
    keys = set(res_slow.keys()) - {"trades", "slices", "oot_windows"}
    for key in keys:
        assert res_fast[key] == res_slow[key], f"Mismatch for key {key}: {res_fast[key]} != {res_slow[key]}"

    assert len(res_fast["trades"]) == len(res_slow["trades"])
    assert len(res_fast["trades"]) == 1
    assert res_fast["trades"][0]["action"] == "UP"
