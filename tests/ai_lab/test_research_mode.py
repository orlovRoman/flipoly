"""Regression tests for the AI Lab research contour."""

import pytest

from polyflip.ai_lab.logreg_adapters import _dt as logreg_dt
from polyflip.ai_lab.lgbm_adapters import _dt as lgbm_dt, _iso_dt
from polyflip.ai_lab.orchestrator import _oot_window_key, _result_oot_windows
from polyflip.ai_lab.policy import evaluate_candidate_policy
from polyflip.ai_lab.service import _live_trading_enabled
from polyflip.crypto.polymarket_backtest import aggregate_stored_polymarket_backtests


def test_research_mode_keeps_low_evidence_candidate_provisional():
    metrics = {
        "median_pnl": -0.25,
        "max_drawdown_usdc": 1.5,
        "total_trades": 4,
        "window_count": 1,
    }
    result = evaluate_candidate_policy(metrics, mode="RESEARCH")
    assert result.gate_passed is True
    assert result.diagnostics["provisional"] is True
    assert any(reason.startswith("INSUFFICIENT_TRADES") for reason in result.rejection_reasons)


def test_standard_mode_still_rejects_low_evidence_candidate():
    result = evaluate_candidate_policy(
        {
            "median_pnl": -0.25,
            "max_drawdown_usdc": 1.5,
            "total_trades": 4,
            "window_count": 1,
        },
        mode="STANDARD",
    )
    assert result.gate_passed is False


def test_research_mode_keeps_safety_drawdown_gate_hard():
    result = evaluate_candidate_policy(
        {
            "median_pnl": 1.0,
            "max_drawdown_usdc": 30.0,
            "total_trades": 100,
            "window_count": 3,
        },
        mode="RESEARCH",
    )
    assert result.gate_passed is False
    assert any(reason.startswith("EXCESSIVE_DRAWDOWN") for reason in result.rejection_reasons)


def test_aggregated_polymarket_backtest_preserves_window_metrics():
    summary = aggregate_stored_polymarket_backtests(
        [
            {
                "n_markets": 10,
                "n_quotes": 10,
                "n_oof": 10,
                "n_eligible": 4,
                "n_trades": 4,
                "net_profit": 1.0,
                "total_invested": 4.0,
                "equity_curve": [
                    {"entry_time": "2026-01-01T00:00:00Z", "trade_pnl": 1.0},
                ],
                "oot_windows": [
                    {"window": 1, "n_trades": 2, "net_profit": 0.5, "max_drawdown_usdc": 0.1},
                    {"window": 2, "n_trades": 2, "net_profit": 0.5, "max_drawdown_usdc": 0.2},
                ],
            }
        ],
        strategy_branch="COMBINED",
    )
    assert summary["window_count"] == 2
    assert summary["median_oot_pnl"] == 0.5
    assert summary["median_oot_drawdown"] == pytest.approx(0.15)

def test_research_mode_rejects_missing_polymarket_sample():
    result = evaluate_candidate_policy(
        {"median_pnl": 0.0, "max_drawdown_usdc": 0.0, "total_trades": 0, "window_count": 0},
        mode="RESEARCH",
    )
    assert result.gate_passed is False
    assert "NO_PNL_SAMPLE" in result.rejection_reasons


def test_manifest_datetime_parser_preserves_iso8601_values():
    value = "2026-07-01T00:00:00Z"
    assert lgbm_dt(value) is not None
    assert lgbm_dt(value).isoformat() == "2026-07-01T00:00:00+00:00"
    assert logreg_dt(value).isoformat() == "2026-07-01T00:00:00+00:00"
    assert lgbm_dt("not-a-timestamp") is None
    assert _iso_dt(None) is None


def test_oot_window_extraction_reads_persisted_slices():
    result = {
        "slices": {
            "oot_windows": [
                {"window": 1, "net_profit": 0.4, "n_trades": 2},
                {"window": 2, "net_profit": 0.6, "n_trades": 3},
            ]
        }
    }
    windows = _result_oot_windows(result)
    assert len(windows) == 2
    assert _oot_window_key(windows[0]) == ("ordinal", "1")
    assert _oot_window_key(windows[1]) == ("ordinal", "2")


def test_research_gate_allows_paper_but_blocks_live_trading():
    assert not _live_trading_enabled([("TRADING_ENABLED", "true")])
    assert not _live_trading_enabled([("LIVE_TRADING_ENABLED", "false")])
    assert _live_trading_enabled([("LIVE_TRADING_ENABLED", "true")])
