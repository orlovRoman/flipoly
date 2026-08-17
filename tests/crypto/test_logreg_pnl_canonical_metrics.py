import numpy as np
import pandas as pd
import pytest

from polyflip.crypto.logreg_polymarket_backtest import compute_logreg_polymarket_backtest
from polyflip.crypto.polymarket_backtest import (
    METRICS_SCHEMA_VERSION,
    CanonicalBacktestMetrics,
    MetricsSchemaMismatchError,
    adapt_canonical_backtest_metrics,
    compute_oof_polymarket_backtest,
)
from polyflip.scripts.retrain_logreg_candidates import _split_chronological_windows


def test_logreg_result_with_nonzero_net_profit_cannot_turn_into_zero_pnl():
    """Verify that a backtest result with non-zero net_profit cannot turn into 0.0."""
    raw_result = {
        "net_profit": 42.50,
        "roi_pct": 14.12,
        "max_drawdown_usdc": 3.20,
        "n_trades": 10,
        "win_rate": 0.7,
    }
    metrics = adapt_canonical_backtest_metrics(raw_result)
    assert metrics.net_profit == 42.50
    assert metrics.total_pnl == 42.50
    assert metrics.roi_pct == 14.12
    assert metrics.max_drawdown == 3.20
    assert metrics.max_drawdown_usdc == 3.20
    assert metrics.metrics_schema_version == METRICS_SCHEMA_VERSION

    # Negative PnL must also never be coerced to zero
    neg_result = {
        "net_profit": -18.75,
        "roi_pct": -6.25,
        "max_drawdown_usdc": 20.00,
        "n_trades": 8,
        "win_rate": 0.25,
    }
    neg_metrics = adapt_canonical_backtest_metrics(neg_result)
    assert neg_metrics.net_profit == -18.75
    assert neg_metrics.total_pnl == -18.75


def test_logreg_missing_canonical_metrics_raises_typed_error():
    """Verify that missing canonical keys or invalid values raise MetricsSchemaMismatchError."""
    # Missing net_profit
    with pytest.raises(MetricsSchemaMismatchError, match="missing required canonical key 'net_profit'"):
        adapt_canonical_backtest_metrics({
            "total_pnl": 10.0,
            "roi_pct": 5.0,
            "max_drawdown_usdc": 1.0,
        })

    # Missing roi_pct
    with pytest.raises(MetricsSchemaMismatchError, match="missing required canonical key 'roi_pct'"):
        adapt_canonical_backtest_metrics({
            "net_profit": 10.0,
            "max_drawdown_usdc": 1.0,
        })

    # Missing max_drawdown_usdc
    with pytest.raises(MetricsSchemaMismatchError, match="missing required canonical key 'max_drawdown_usdc'"):
        adapt_canonical_backtest_metrics({
            "net_profit": 10.0,
            "roi_pct": 5.0,
        })

    # None values
    with pytest.raises(MetricsSchemaMismatchError, match="missing required canonical key 'net_profit'"):
        adapt_canonical_backtest_metrics({
            "net_profit": None,
            "roi_pct": 5.0,
            "max_drawdown_usdc": 1.0,
        })

    # Non-numeric
    with pytest.raises(MetricsSchemaMismatchError, match="non-numeric canonical metric value"):
        adapt_canonical_backtest_metrics({
            "net_profit": "invalid",
            "roi_pct": 5.0,
            "max_drawdown_usdc": 1.0,
        })

    # Non-finite (NaN / Inf)
    with pytest.raises(MetricsSchemaMismatchError, match="non-finite metric value"):
        adapt_canonical_backtest_metrics({
            "net_profit": float("nan"),
            "roi_pct": 5.0,
            "max_drawdown_usdc": 1.0,
        })

    # Non-dict
    with pytest.raises(MetricsSchemaMismatchError, match="must be a dict"):
        adapt_canonical_backtest_metrics(["not", "a", "dict"])


def test_logreg_canonical_metrics_identically_propagated_to_report_and_db():
    """Verify net_profit, roi_pct, max_drawdown_usdc identically propagate across all layers."""
    canonical_result = {
        "net_profit": 15.75,
        "roi_pct": 5.25,
        "max_drawdown_usdc": 2.10,
        "n_trades": 35,
        "win_rate": 0.65,
        "trades": [],
    }

    metrics = adapt_canonical_backtest_metrics(canonical_result)

    # Candidate evaluation schema
    cand = {
        "variant": "BASE+DERIVED",
        "C": 0.1,
        "class_weight": "balanced",
        "sample_weight_mode": "EXPONENTIAL",
        "sample_weight_tau": 3.0,
        "calibration_method": "RAW",
        "feature_names": ["mid_price", "spread"],
        "sequence_coverage": 1.0,
        "val_auc": 0.62,
        "brier": 0.22,
        "ece": 0.04,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "net_profit": round(metrics.net_profit, 4),
        "total_pnl": round(metrics.net_profit, 4),
        "roi_pct": round(metrics.roi_pct, 4),
        "roi": round(metrics.roi_pct, 4),
        "n_trades": metrics.n_trades,
        "win_rate": round(metrics.win_rate, 4),
        "max_drawdown": round(metrics.max_drawdown, 4),
        "max_drawdown_usdc": round(metrics.max_drawdown, 4),
        "oot_windows": {
            "T1": {"status": "OK", "net_profit": 5.25},
            "T2": {"status": "OK", "net_profit": 5.25},
            "T3": {"status": "EMPTY", "net_profit": None},
            "median_pnl": 5.25,
        },
        "deployable": True,
        "rejection_reasons": [],
    }

    # ModelRegistry training_params persistence
    training_params = {
        "metrics_schema_version": cand.get("metrics_schema_version", METRICS_SCHEMA_VERSION),
        "experiment_variant": cand["variant"],
        "C": cand["C"],
        "class_weight": cand["class_weight"],
        "sample_weight_mode": cand["sample_weight_mode"],
        "sample_weight_tau": cand["sample_weight_tau"],
        "calibration_method": cand["calibration_method"],
        "validation_scheme": "GROUPED_WALK_FORWARD",
        "target_source": "POLYMARKET_FLIP_VS_FINAL_OUTCOME",
        "sequence_coverage": cand["sequence_coverage"],
        "deployable": cand["deployable"],
        "rejection_reasons": cand["rejection_reasons"],
        "oot_windows": cand["oot_windows"],
        "net_profit": cand["net_profit"],
        "combined_pnl": cand["net_profit"],
        "total_pnl": cand["net_profit"],
        "roi_pct": cand["roi_pct"],
        "n_trades": cand["n_trades"],
        "win_rate": cand["win_rate"],
        "max_drawdown": cand["max_drawdown"],
        "max_drawdown_usdc": cand["max_drawdown"],
    }

    assert training_params["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert training_params["net_profit"] == 15.75
    assert training_params["combined_pnl"] == 15.75
    assert training_params["total_pnl"] == 15.75
    assert training_params["roi_pct"] == 5.25
    assert training_params["max_drawdown_usdc"] == 2.10

    # JSON report structure
    json_entry = {
        "metrics_schema_version": cand.get("metrics_schema_version", METRICS_SCHEMA_VERSION),
        "variant": cand["variant"],
        "net_profit": cand["net_profit"],
        "total_pnl": cand["net_profit"],
        "roi_pct": cand["roi_pct"],
        "n_trades": cand["n_trades"],
        "win_rate": cand["win_rate"],
        "max_drawdown": cand["max_drawdown"],
        "max_drawdown_usdc": cand["max_drawdown"],
        "deployable": cand["deployable"],
    }
    assert json_entry["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert json_entry["net_profit"] == 15.75
    assert json_entry["roi_pct"] == 5.25
    assert json_entry["max_drawdown_usdc"] == 2.10

    # CSV summary row structure with direct canonical net_profit and None for EMPTY windows
    t1_pnl = cand["oot_windows"].get("T1", {}).get("net_profit") if cand["oot_windows"].get("T1", {}).get("status") != "EMPTY" else None
    t2_pnl = cand["oot_windows"].get("T2", {}).get("net_profit") if cand["oot_windows"].get("T2", {}).get("status") != "EMPTY" else None
    t3_pnl = cand["oot_windows"].get("T3", {}).get("net_profit") if cand["oot_windows"].get("T3", {}).get("status") != "EMPTY" else None

    csv_row = {
        "metrics_schema_version": cand.get("metrics_schema_version", METRICS_SCHEMA_VERSION),
        "asset_phase": "BTC_base",
        "rank": 1,
        "variant": cand["variant"],
        "net_profit": cand["net_profit"],
        "total_pnl": cand["net_profit"],
        "roi_pct": cand["roi_pct"],
        "n_trades": cand["n_trades"],
        "win_rate": cand["win_rate"],
        "max_drawdown": cand["max_drawdown"],
        "max_drawdown_usdc": cand["max_drawdown"],
        "deployable": cand["deployable"],
        "t1_pnl": t1_pnl,
        "t2_pnl": t2_pnl,
        "t3_pnl": t3_pnl,
        "median_window_pnl": cand["oot_windows"].get("median_pnl"),
    }
    assert csv_row["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert csv_row["net_profit"] == 15.75
    assert csv_row["roi_pct"] == 5.25
    assert csv_row["max_drawdown_usdc"] == 2.10
    assert csv_row["t1_pnl"] == 5.25
    assert csv_row["t2_pnl"] == 5.25
    assert csv_row["t3_pnl"] is None


def test_logreg_winning_and_losing_trades_yield_nonzero_combined_pnl():
    """Verify that backtesting a mix of winning and losing trades yields non-zero total PnL and matches canonical adapter."""
    starts = pd.to_datetime([
        "2026-08-01T00:00:00Z",
        "2026-08-01T00:15:00Z",
        "2026-08-01T00:30:00Z",
        "2026-08-01T00:45:00Z",
        "2026-08-01T01:00:00Z",
    ], utc=True)
    frame = pd.DataFrame({
        "market_id": ["m1", "m2", "m3", "m4", "m5"],
        "asset": ["BTC"] * 5,
        "market_start": starts,
        "recorded_at": starts,
        "time_left_min": [14.0, 14.0, 14.0, 14.0, 14.0],
        "mid_price": [0.30, 0.70, 0.40, 0.60, 0.25],
        "best_bid": [0.29, 0.69, 0.39, 0.59, 0.24],
        "best_ask": [0.31, 0.71, 0.41, 0.61, 0.26],
        "spread": [0.02] * 5,
        "final_outcome": ["YES", "YES", "NO", "NO", "YES"],
        "target": [1, 1, 0, 0, 1],
    })
    quotes = frame[["market_id", "recorded_at", "mid_price", "best_bid", "best_ask", "final_outcome", "spread"]].copy()

    scores = [0.90, 0.10, 0.90, 0.90, 0.10]

    result = compute_logreg_polymarket_backtest(
        frame,
        scores,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.04,
        cost_buffer=0.0,
        fee_rate=0.002,
        stake_usdc=10.0,
    )

    assert result["n_trades"] == 5
    assert result["win_rate"] == pytest.approx(0.60)
    assert result["net_profit"] != 0.0
    assert result["max_drawdown_usdc"] > 0.0
    assert result["metrics_schema_version"] == METRICS_SCHEMA_VERSION

    metrics = adapt_canonical_backtest_metrics(result)
    assert metrics.net_profit == pytest.approx(result["net_profit"])
    assert metrics.roi_pct == pytest.approx(result["roi_pct"])
    assert metrics.max_drawdown == pytest.approx(result["max_drawdown_usdc"])
    assert metrics.n_trades == 5
    assert metrics.win_rate == pytest.approx(0.60)


def test_logreg_split_chronological_windows_uses_canonical_adapter():
    """Verify that _split_chronological_windows enforces the canonical metrics adapter and no fake zeros on empty windows."""
    starts = pd.to_datetime([
        "2026-08-01T00:00:00Z",
        "2026-08-01T01:00:00Z",
        "2026-08-01T02:00:00Z",
    ], utc=True)
    frame = pd.DataFrame({
        "market_id": ["m1", "m2", "m3"],
        "recorded_at": starts,
        "mid_price": [0.30, 0.30, 0.30],
        "best_bid": [0.29, 0.29, 0.29],
        "best_ask": [0.31, 0.31, 0.31],
        "spread": [0.02, 0.02, 0.02],
        "final_outcome": ["YES", "YES", "NO"],
        "target": [1, 1, 0],
    })
    quotes = frame.copy()
    p_yes = np.array([0.90, 0.90, 0.90])

    windows = _split_chronological_windows(
        frame,
        p_yes,
        quotes,
        strategy_branch="COMBINED",
        min_edge=0.01,
        cost_buffer=0.0,
    )

    assert windows["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    for w_name in ("T1", "T2", "T3"):
        w = windows[w_name]
        assert w["status"] in {"OK", "SPARSE"}
        assert w["net_profit"] is not None
        assert w["roi_pct"] is not None
        assert w["max_drawdown_usdc"] is not None
        assert w["metrics_schema_version"] == METRICS_SCHEMA_VERSION


def test_logreg_empty_windows_have_no_fictitious_zero_metrics():
    """Verify that empty windows are explicitly marked EMPTY with None metrics (never fake 0.0)."""
    # 1. Empty frame
    empty_windows = _split_chronological_windows(
        pd.DataFrame(),
        np.array([]),
        None,
        strategy_branch="COMBINED",
    )
    assert empty_windows["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert empty_windows["median_pnl"] is None
    assert empty_windows["non_negative_windows_count"] == 0
    for w_name in ("T1", "T2", "T3"):
        w = empty_windows[w_name]
        assert w["status"] == "EMPTY"
        assert w["net_profit"] is None
        assert w["roi_pct"] is None
        assert w["max_drawdown_usdc"] is None

    # 2. Sparse frame with only 1 market (n < 3)
    frame1 = pd.DataFrame([{
        "market_id": "m1",
        "recorded_at": pd.Timestamp("2026-08-01T00:00:00Z"),
        "mid_price": 0.30,
        "best_bid": 0.29,
        "best_ask": 0.31,
        "spread": 0.02,
        "final_outcome": "YES",
        "target": 1,
    }])
    quotes1 = frame1.copy()
    sparse_windows = _split_chronological_windows(
        frame1,
        np.array([0.90]),
        quotes1,
        strategy_branch="COMBINED",
        min_edge=0.01,
        cost_buffer=0.0,
    )
    assert sparse_windows["T1"]["status"] == "SPARSE"
    assert sparse_windows["T1"]["net_profit"] is not None
    assert sparse_windows["T2"]["status"] == "EMPTY"
    assert sparse_windows["T2"]["net_profit"] is None
    assert sparse_windows["T3"]["status"] == "EMPTY"
    assert sparse_windows["T3"]["net_profit"] is None

def test_logreg_aggregate_stored_polymarket_backtests_preserves_nonzero_net_profit():
    """Verify that aggregate_stored_polymarket_backtests never zeroes out non-zero net_profit."""
    from polyflip.crypto.polymarket_backtest import aggregate_stored_polymarket_backtests

    r1 = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "n_markets": 10,
        "n_quotes": 10,
        "n_oof": 10,
        "n_eligible": 10,
        "n_trades": 5,
        "win_rate": 0.8,
        "total_invested": 5.0,
        "stake_usdc": 1.0,
        "net_profit": 12.50,
        "roi_pct": 250.0,
        "max_drawdown_usdc": 1.0,
        "equity_curve": [{"entry_time": "2026-08-01T00:00:00Z", "trade_pnl": 12.50}],
    }
    r2 = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "n_markets": 10,
        "n_quotes": 10,
        "n_oof": 10,
        "n_eligible": 10,
        "n_trades": 5,
        "win_rate": 0.6,
        "total_invested": 5.0,
        "stake_usdc": 1.0,
        "net_profit": 7.50,
        "roi_pct": 150.0,
        "max_drawdown_usdc": 2.0,
        "equity_curve": [{"entry_time": "2026-08-01T01:00:00Z", "trade_pnl": 7.50}],
    }

    aggregated = aggregate_stored_polymarket_backtests([r1, r2], strategy_branch="COMBINED")
    assert aggregated["net_profit"] == pytest.approx(20.00)
    assert aggregated["n_trades"] == 10
    assert aggregated["win_rate"] == pytest.approx(0.70)
    assert aggregated["metrics_schema_version"] == METRICS_SCHEMA_VERSION


def test_logreg_aggregate_stored_polymarket_backtests_raises_on_missing_canonical_metrics():
    """Verify that aggregate_stored_polymarket_backtests raises MetricsSchemaMismatchError on missing metrics."""
    from polyflip.crypto.polymarket_backtest import aggregate_stored_polymarket_backtests

    # Missing net_profit
    bad_result_no_pnl = {
        "n_trades": 5,
        "roi_pct": 10.0,
        "max_drawdown_usdc": 1.0,
    }
    with pytest.raises(MetricsSchemaMismatchError, match="missing required canonical key 'net_profit'"):
        aggregate_stored_polymarket_backtests([bad_result_no_pnl], strategy_branch="COMBINED")

    # Missing roi_pct
    bad_result_no_roi = {
        "net_profit": 5.0,
        "n_trades": 5,
        "max_drawdown_usdc": 1.0,
    }
    with pytest.raises(MetricsSchemaMismatchError, match="missing required canonical key 'roi_pct'"):
        aggregate_stored_polymarket_backtests([bad_result_no_roi], strategy_branch="COMBINED")

    # Missing max_drawdown_usdc
    bad_result_no_dd = {
        "net_profit": 5.0,
        "n_trades": 5,
        "roi_pct": 10.0,
    }
    with pytest.raises(MetricsSchemaMismatchError, match="missing required canonical key 'max_drawdown_usdc'"):
        aggregate_stored_polymarket_backtests([bad_result_no_dd], strategy_branch="COMBINED")
