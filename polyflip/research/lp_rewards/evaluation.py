from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel

getcontext().prec = 28


class GateAEvaluationReport(BaseModel):
    verdict: str
    point_estimate: Decimal
    lower_95: Decimal
    upper_95: Decimal
    total_days: int
    total_quote_hours: Decimal
    total_simulated_quote_hours: Decimal = Decimal("0.0")
    total_actual_quote_hours: Decimal = Decimal("0.0")
    active_markets_count: int
    min_market_coverage: Decimal
    max_market_pnl_share: Decimal
    protocol_hash_valid: bool
    book_uncertain_count: int
    stress_test_passed: bool
    rejection_reasons: List[str]
    details: Dict[str, Any]


def compute_block_bootstrap_ci(
    daily_net_pnls: List[Decimal],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[Decimal, Decimal, Decimal]:
    """Perform block bootstrap strictly by whole UTC days.

    Each sample in daily_net_pnls represents the aggregated Net PnL of all markets for one complete UTC day.
    Returns (point_estimate, lower_ci, upper_ci) for daily return on $100 allocated capital.
    """
    if not daily_net_pnls:
        return Decimal("0.0"), Decimal("0.0"), Decimal("0.0")

    pnl_array = np.array([float(p) for p in daily_net_pnls])
    point_est = float(np.mean(pnl_array))

    rng = np.random.default_rng(seed)
    n_days = len(pnl_array)
    resamples = rng.choice(pnl_array, size=(n_bootstrap, n_days), replace=True)
    resample_means = np.mean(resamples, axis=1)

    lower_bound = float(np.percentile(resample_means, 100 * (alpha / 2.0)))
    upper_bound = float(np.percentile(resample_means, 100 * (1.0 - alpha / 2.0)))

    return (
        Decimal(str(round(point_est, 4))),
        Decimal(str(round(lower_bound, 4))),
        Decimal(str(round(upper_bound, 4))),
    )


def determine_gate_a_verdict(
    point_est: Decimal,
    lower_95: Decimal,
    upper_95: Decimal,
    target_rate: Decimal = Decimal("3.50"),
    total_days: int = 7,
    min_days: int = 7,
) -> str:
    """Classify results into the 5 mutually exclusive Gate A verdicts:
    - EDGE_REJECTED: upper_95 <= 0
    - PROFITABLE_BELOW_TARGET: lower_95 > 0 and upper_95 < 3.50
    - TARGET_REJECTED: upper_95 < 3.50
    - TARGET_PLAUSIBLE: lower_95 <= 3.50 <= upper_95
    - PROCEED_LIVE: lower_95 > 0 and point_est >= 3.50 and total_days >= min_days
    """
    if upper_95 <= Decimal("0.0"):
        return "EDGE_REJECTED"

    if lower_95 > Decimal("0.0") and upper_95 < target_rate:
        return "PROFITABLE_BELOW_TARGET"

    if upper_95 < target_rate:
        return "TARGET_REJECTED"

    if lower_95 > Decimal("0.0") and point_est >= target_rate and total_days >= min_days:
        return "PROCEED_LIVE"

    if lower_95 <= target_rate <= upper_95:
        return "TARGET_PLAUSIBLE"

    return "TARGET_REJECTED"


def evaluate_gate_a_full(
    daily_records: List[Dict[str, Any]],
    expected_protocol_hash: Optional[str] = None,
    target_rate: Decimal = Decimal("3.50"),
    min_calendar_days: int = 7,
    min_quote_hours: int = 100,
    min_active_markets: int = 10,
    min_market_coverage_ratio: Decimal = Decimal("0.99"),
    max_single_market_pnl_share: Decimal = Decimal("0.50"),
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> GateAEvaluationReport:
    """Strictly evaluate all Gate A criteria per protocol."""
    rejection_reasons = []

    total_days = len(daily_records)
    if total_days < min_calendar_days:
        rejection_reasons.append(f"Insufficient days: {total_days} < {min_calendar_days}")

    daily_pnls: List[Decimal] = []
    total_quote_hours = Decimal("0.0")
    total_simulated_quote_hours = Decimal("0.0")
    total_actual_quote_hours = Decimal("0.0")
    markets_seen = set()
    market_pnls: Dict[str, Decimal] = {}
    coverage_by_market: Dict[str, List[Decimal]] = {}
    total_book_uncertain = 0
    hashes_seen = set()

    for r in daily_records:
        pnl_val = r.get("net_pnl")
        pnl = Decimal(str(pnl_val if pnl_val is not None else "0.0"))
        daily_pnls.append(pnl)

        sim_raw = r.get("simulated_quote_hours")
        if sim_raw is None:
            sim_raw = r.get("quote_hours")
        if sim_raw is None:
            sim_raw = "0.0"
        sim_qh = Decimal(str(sim_raw))

        act_raw = r.get("actual_quote_hours")
        if act_raw is None:
            act_raw = "0.0"
        act_qh = Decimal(str(act_raw))

        if sim_qh < Decimal("0.0"):
            rejection_reasons.append(f"Negative simulated quote-hours {sim_qh} in record {r.get('date')}")
        elif sim_qh > Decimal("24.01"):
            rejection_reasons.append(f"Simulated quote-hours {sim_qh} exceeds 24h in record {r.get('date')}")

        if act_qh < Decimal("0.0"):
            rejection_reasons.append(f"Negative actual quote-hours {act_qh} in record {r.get('date')}")
        elif act_qh > Decimal("24.01"):
            rejection_reasons.append(f"Actual quote-hours {act_qh} exceeds 24h in record {r.get('date')}")

        total_simulated_quote_hours += sim_qh
        total_actual_quote_hours += act_qh
        total_quote_hours += sim_qh

        ph = r.get("protocol_hash")
        if ph:
            hashes_seen.add(ph)

        total_book_uncertain += int(r.get("book_uncertain_count", 0))

        per_m = r.get("market_breakdown", {})
        for cid, mdata in per_m.items():
            markets_seen.add(cid)
            m_pnl = Decimal(str(mdata.get("net_pnl", "0.0")))
            market_pnls[cid] = market_pnls.get(cid, Decimal("0.0")) + m_pnl
            cov = Decimal(str(mdata.get("coverage_ratio", "1.0")))
            coverage_by_market.setdefault(cid, []).append(cov)

    if total_simulated_quote_hours < Decimal(str(min_quote_hours)):
        rejection_reasons.append(
            f"Insufficient quote-hours: {total_simulated_quote_hours:.1f} simulated < {min_quote_hours} "
            f"(actual quote-hours in shadow mode: {total_actual_quote_hours:.1f})"
        )

    active_markets_count = len(markets_seen)
    if active_markets_count < min_active_markets:
        rejection_reasons.append(f"Insufficient active markets: {active_markets_count} < {min_active_markets}")

    min_market_coverage = Decimal("1.0")
    if coverage_by_market:
        for cid, cov_list in coverage_by_market.items():
            avg_cov = sum(cov_list, Decimal("0.0")) / Decimal(str(len(cov_list)))
            if avg_cov < min_market_coverage:
                min_market_coverage = avg_cov
            if avg_cov < min_market_coverage_ratio:
                rejection_reasons.append(f"Market {cid} coverage {avg_cov*100:.1f}% < {min_market_coverage_ratio*100:.1f}%")

    total_net_pnl = sum(daily_pnls, Decimal("0.0"))
    max_market_pnl_share = Decimal("0.0")
    if total_net_pnl > Decimal("0.0") and market_pnls:
        for cid, mpnl in market_pnls.items():
            share = mpnl / total_net_pnl
            if share > max_market_pnl_share:
                max_market_pnl_share = share
            if share > max_single_market_pnl_share:
                rejection_reasons.append(f"Market {cid} PnL share {share*100:.1f}% > {max_single_market_pnl_share*100:.1f}%")

    protocol_hash_valid = True
    missing_hashes = [i for i, r in enumerate(daily_records) if not r.get("protocol_hash")]
    if expected_protocol_hash:
        if missing_hashes:
            protocol_hash_valid = False
            rejection_reasons.append(f"Protocol hash missing from evaluation records at indices {missing_hashes}; expected {expected_protocol_hash}")
        mismatches = [r.get("protocol_hash") for r in daily_records if r.get("protocol_hash") and r.get("protocol_hash") != expected_protocol_hash]
        if mismatches:
            protocol_hash_valid = False
            rejection_reasons.append(f"Protocol hash mismatch. Seen: {set(mismatches)}, Expected: {expected_protocol_hash}")

    missing_dates = [i for i, r in enumerate(daily_records) if not r.get("date")]
    if missing_dates:
        rejection_reasons.append(f"Required 'date' field missing from evaluation records at indices: {missing_dates}")

    dates_seen = [str(r.get("date")) for r in daily_records if r.get("date")]
    if len(dates_seen) != len(set(dates_seen)):
        rejection_reasons.append(f"Duplicate evaluation dates detected: {len(dates_seen)} records, {len(set(dates_seen))} unique")

    if total_book_uncertain > 0:
        rejection_reasons.append(f"BOOK_UNCERTAIN detected: {total_book_uncertain} occurrences")

    # Stress test: apply 20% adverse haircut to daily pnls
    stress_pnls = [p * Decimal("0.80") for p in daily_pnls] if daily_pnls else [Decimal("0.0")]
    point_stress, lower_stress, upper_stress = compute_block_bootstrap_ci(
        stress_pnls, n_bootstrap=min(1000, n_bootstrap), seed=seed
    )
    stress_test_passed = (lower_stress > Decimal("0.0")) if daily_pnls else False
    if not stress_test_passed and daily_pnls:
        rejection_reasons.append(f"Stress test failed: 95% lower bound under shock is {lower_stress} <= 0")

    point, lower, upper = compute_block_bootstrap_ci(
        daily_pnls, n_bootstrap=n_bootstrap, seed=seed
    )

    base_verdict = determine_gate_a_verdict(
        point, lower, upper, target_rate=target_rate, total_days=total_days, min_days=min_calendar_days
    )

    if rejection_reasons:
        final_verdict = "EDGE_REJECTED" if base_verdict == "EDGE_REJECTED" else "TARGET_REJECTED"
    else:
        final_verdict = base_verdict

    return GateAEvaluationReport(
        verdict=final_verdict,
        point_estimate=point,
        lower_95=lower,
        upper_95=upper,
        total_days=total_days,
        total_quote_hours=total_quote_hours,
        total_simulated_quote_hours=total_simulated_quote_hours,
        total_actual_quote_hours=total_actual_quote_hours,
        active_markets_count=active_markets_count,
        min_market_coverage=min_market_coverage,
        max_market_pnl_share=max_market_pnl_share,
        protocol_hash_valid=protocol_hash_valid,
        book_uncertain_count=total_book_uncertain,
        stress_test_passed=stress_test_passed,
        rejection_reasons=rejection_reasons,
        details={
            "base_verdict": base_verdict,
            "daily_pnls": [str(p) for p in daily_pnls],
            "stress_point": str(point_stress),
            "stress_lower": str(lower_stress),
            "stress_upper": str(upper_stress),
        },
    )


def determine_gate_b_verdict(
    point_est: Decimal,
    lower_95: Decimal,
    upper_95: Decimal,
    max_drawdown: Decimal,
    mean_prediction_error: Decimal,
    target_rate: Decimal = Decimal("3.50"),
    max_drawdown_limit: Decimal = Decimal("0.10"),
    max_error_limit: Decimal = Decimal("0.30"),
    total_live_days: int = 14,
    min_live_days: int = 14,
) -> str:
    """Classify live results for Gate B:
    - INSUFFICIENT_DATA: total_live_days < min_live_days
    - TARGET_CONFIRMED: lower_95 >= 3.50, max_drawdown <= 0.10, prediction_error <= 0.30, days >= 14
    - TARGET_NOT_CONFIRMED: upper_95 < 3.50 or max_drawdown > 0.10
    - INCONCLUSIVE: CI spans across 3.50 threshold
    """
    if total_live_days < min_live_days:
        return "INSUFFICIENT_DATA"

    if max_drawdown > max_drawdown_limit:
        return "TARGET_NOT_CONFIRMED"

    if upper_95 < target_rate:
        return "TARGET_NOT_CONFIRMED"

    if lower_95 >= target_rate and mean_prediction_error <= max_error_limit:
        return "TARGET_CONFIRMED"

    return "INCONCLUSIVE"
