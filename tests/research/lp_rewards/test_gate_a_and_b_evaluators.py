from decimal import Decimal
import pytest

from polyflip.research.lp_rewards.evaluation import (
    compute_block_bootstrap_ci,
    determine_gate_a_verdict,
    determine_gate_b_verdict,
    evaluate_gate_a_full,
)
from polyflip.research.lp_rewards.models import DailyEvaluationRecord
from polyflip.research.lp_rewards.reward_calibration import RewardCalibrator


def test_evaluate_gate_a_full_happy_path():
    # 7 days, 12 markets, 120 quote-hours, 99.5% coverage, balanced PnL, valid hash, 0 uncertain
    daily_records = []
    for day in range(7):
        market_breakdown = {}
        for m in range(12):
            market_breakdown[f"cond_{m}"] = {
                "net_pnl": "0.35",  # 12 * 0.35 = 4.20 per day
                "coverage_ratio": "0.995",
            }
        daily_records.append({
            "day": day,
            "date": f"2026-09-0{day+1}",
            "net_pnl": "4.20",
            "quote_hours": "20.0",  # 7 * 20 = 140 quote-hours
            "protocol_hash": "hash_v0.1_verified",
            "book_uncertain_count": 0,
            "market_breakdown": market_breakdown,
        })

    report = evaluate_gate_a_full(
        daily_records=daily_records,
        expected_protocol_hash="hash_v0.1_verified",
        target_rate=Decimal("3.50"),
        min_calendar_days=7,
        min_quote_hours=100,
        min_active_markets=10,
        min_market_coverage_ratio=Decimal("0.99"),
        max_single_market_pnl_share=Decimal("0.50"),
    )

    assert report.verdict == "PROCEED_LIVE"
    assert report.protocol_hash_valid is True
    assert report.total_quote_hours == Decimal("140.0")
    assert report.active_markets_count == 12
    assert report.book_uncertain_count == 0
    assert report.stress_test_passed is True
    assert len(report.rejection_reasons) == 0


def test_evaluate_gate_a_full_failure_modes():
    # Case 1: Insufficient quote-hours & insufficient markets
    records_short = [
        {
            "net_pnl": "4.00",
            "quote_hours": "10.0",
            "protocol_hash": "hash_good",
            "book_uncertain_count": 0,
            "market_breakdown": {
                "cond_1": {"net_pnl": "4.00", "coverage_ratio": "0.995"}
            },
        }
        for _ in range(7)
    ]
    rep1 = evaluate_gate_a_full(
        records_short,
        expected_protocol_hash="hash_good",
        min_quote_hours=100,
        min_active_markets=10,
    )
    assert rep1.verdict != "PROCEED_LIVE"
    assert any("quote-hours" in r for r in rep1.rejection_reasons)
    assert any("active markets" in r for r in rep1.rejection_reasons)

    # Case 2: Market coverage below 99%
    records_low_cov = [
        {
            "net_pnl": "4.00",
            "quote_hours": "20.0",
            "protocol_hash": "hash_good",
            "book_uncertain_count": 0,
            "market_breakdown": {
                f"cond_{m}": {"net_pnl": "0.40", "coverage_ratio": "0.95" if m == 0 else "0.995"}
                for m in range(10)
            },
        }
        for _ in range(7)
    ]
    rep2 = evaluate_gate_a_full(records_low_cov, min_market_coverage_ratio=Decimal("0.99"))
    assert any("coverage" in r for r in rep2.rejection_reasons)

    # Case 3: Single market PnL share > 50%
    records_dominant = [
        {
            "net_pnl": "10.00",
            "quote_hours": "20.0",
            "protocol_hash": "hash_good",
            "book_uncertain_count": 0,
            "market_breakdown": {
                "cond_whale": {"net_pnl": "8.00", "coverage_ratio": "0.995"},
                **{f"cond_{m}": {"net_pnl": "0.20", "coverage_ratio": "0.995"} for m in range(1, 11)},
            },
        }
        for _ in range(7)
    ]
    rep3 = evaluate_gate_a_full(records_dominant, max_single_market_pnl_share=Decimal("0.50"))
    assert any("PnL share" in r for r in rep3.rejection_reasons)

    # Case 4: Protocol hash mismatch
    rep4 = evaluate_gate_a_full(records_low_cov, expected_protocol_hash="hash_expected_different")
    assert rep4.protocol_hash_valid is False
    assert any("Protocol hash mismatch" in r for r in rep4.rejection_reasons)

    # Case 5: BOOK_UNCERTAIN detected
    records_uncertain = [
        {
            "net_pnl": "4.00",
            "quote_hours": "20.0",
            "protocol_hash": "hash_good",
            "book_uncertain_count": 5,
            "market_breakdown": {f"cond_{m}": {"net_pnl": "0.40", "coverage_ratio": "0.995"} for m in range(10)},
        }
        for _ in range(7)
    ]
    rep5 = evaluate_gate_a_full(records_uncertain)
    assert rep5.book_uncertain_count == 35
    assert any("BOOK_UNCERTAIN" in r for r in rep5.rejection_reasons)


def test_determine_gate_b_verdict():
    # 1. Insufficient days (< 14)
    v1 = determine_gate_b_verdict(
        point_est=Decimal("4.00"),
        lower_95=Decimal("3.60"),
        upper_95=Decimal("4.40"),
        max_drawdown=Decimal("0.05"),
        mean_prediction_error=Decimal("0.15"),
        total_live_days=10,
        min_live_days=14,
    )
    assert v1 == "INSUFFICIENT_DATA"

    # 2. Drawdown violation (> 0.10)
    v2 = determine_gate_b_verdict(
        point_est=Decimal("4.00"),
        lower_95=Decimal("3.60"),
        upper_95=Decimal("4.40"),
        max_drawdown=Decimal("0.15"),  # > 10%
        mean_prediction_error=Decimal("0.15"),
        total_live_days=14,
        min_live_days=14,
    )
    assert v2 == "TARGET_NOT_CONFIRMED"

    # 3. Target not reached (upper_95 < 3.50)
    v3 = determine_gate_b_verdict(
        point_est=Decimal("2.50"),
        lower_95=Decimal("2.00"),
        upper_95=Decimal("3.00"),
        max_drawdown=Decimal("0.04"),
        mean_prediction_error=Decimal("0.10"),
        total_live_days=14,
    )
    assert v3 == "TARGET_NOT_CONFIRMED"

    # 4. Target confirmed
    v4 = determine_gate_b_verdict(
        point_est=Decimal("4.00"),
        lower_95=Decimal("3.60"),
        upper_95=Decimal("4.40"),
        max_drawdown=Decimal("0.05"),
        mean_prediction_error=Decimal("0.15"),
        total_live_days=14,
    )
    assert v4 == "TARGET_CONFIRMED"

    # 5. Inconclusive (CI spans 3.50)
    v5 = determine_gate_b_verdict(
        point_est=Decimal("3.50"),
        lower_95=Decimal("2.80"),
        upper_95=Decimal("4.20"),
        max_drawdown=Decimal("0.05"),
        mean_prediction_error=Decimal("0.15"),
        total_live_days=14,
    )
    assert v5 == "INCONCLUSIVE"


def test_reward_calibrator_relative_error_relative_to_actual():
    calibrator = RewardCalibrator(max_allowed_error_ratio=Decimal("0.30"))

    # If actual = 10.0, projected = 12.0 -> abs(12-10)/10 = 0.20 (20% error)
    err1 = calibrator.record_observation("c1", Decimal("12.0"), Decimal("10.0"))
    assert err1 == Decimal("0.20")

    # If actual = 20.0, projected = 16.0 -> abs(16-20)/20 = 0.20 (20% error)
    err2 = calibrator.record_observation("c2", Decimal("16.0"), Decimal("20.0"))
    assert err2 == Decimal("0.20")

    passes, mean_err = calibrator.evaluate_gate_b_accuracy()
    assert passes is True
    assert mean_err == Decimal("0.20")

    # Add big error observation: actual = 10.0, projected = 18.0 -> error = 0.80
    calibrator.record_observation("c3", Decimal("18.0"), Decimal("10.0"))
    # Mean error: (0.20 + 0.20 + 0.80) / 3 = 0.40 -> exceeds 0.30
    passes_updated, mean_err_updated = calibrator.evaluate_gate_b_accuracy()
    assert passes_updated is False
    assert mean_err_updated == Decimal("0.40")


def test_daily_evaluation_record_quote_hours_bidirectional_sync():
    """Verify DailyEvaluationRecord bidirectionally synchronizes simulated_quote_hours and quote_hours."""
    # From simulated_quote_hours
    r1 = DailyEvaluationRecord(
        date="2026-09-14",
        protocol_id="proto1",
        protocol_hash="hash1",
        simulated_quote_hours=Decimal("15.5"),
    )
    assert r1.simulated_quote_hours == Decimal("15.5")
    assert r1.quote_hours == Decimal("15.5")
    assert r1.actual_quote_hours == Decimal("0.0")

    # From legacy quote_hours
    r2 = DailyEvaluationRecord(
        date="2026-09-15",
        protocol_id="proto1",
        protocol_hash="hash1",
        quote_hours=Decimal("18.25"),
    )
    assert r2.quote_hours == Decimal("18.25")
    assert r2.simulated_quote_hours == Decimal("18.25")
    assert r2.actual_quote_hours == Decimal("0.0")


def test_evaluate_gate_a_full_simulated_and_actual_quote_hours():
    """Verify GateAEvaluationReport tracks simulated vs actual quote-hours separately."""
    daily_records = []
    for day in range(7):
        market_breakdown = {}
        for m in range(12):
            market_breakdown[f"cond_{m}"] = {
                "net_pnl": "0.35",
                "coverage_ratio": "0.995",
            }
        daily_records.append({
            "day": day,
            "date": f"2026-09-0{day+1}",
            "net_pnl": "4.20",
            "simulated_quote_hours": "18.0",
            "actual_quote_hours": "0.0",
            "protocol_hash": "hash_v0.1_verified",
            "book_uncertain_count": 0,
            "market_breakdown": market_breakdown,
        })

    report = evaluate_gate_a_full(
        daily_records=daily_records,
        expected_protocol_hash="hash_v0.1_verified",
        target_rate=Decimal("3.50"),
        min_calendar_days=7,
        min_quote_hours=100,
        min_active_markets=10,
        min_market_coverage_ratio=Decimal("0.99"),
        max_single_market_pnl_share=Decimal("0.50"),
    )

    assert report.total_simulated_quote_hours == Decimal("126.0")
    assert report.total_actual_quote_hours == Decimal("0.0")
    assert report.total_quote_hours == Decimal("126.0")
    assert report.verdict == "PROCEED_LIVE"
