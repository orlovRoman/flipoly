import argparse
import datetime
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import List, Optional

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.evaluation import (
    compute_block_bootstrap_ci,
    determine_gate_a_verdict,
    evaluate_gate_a_full,
)
from polyflip.research.lp_rewards.protocol import load_protocol


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Formally evaluate Gate A criteria from daily evaluation records.")
    parser.add_argument(
        "--storage-root",
        type=str,
        default=None,
        help="Path to root storage directory (overrides protocol default and LP_STORAGE_ROOT).",
    )
    cli_args = parser.parse_args(argv)

    protocol = load_protocol(storage_root=cli_args.storage_root)
    storage_path = Path(protocol.data_storage.root_path)
    daily_eval_dir = storage_path / "daily_evaluations"

    if not daily_eval_dir.exists() or not list(daily_eval_dir.glob("*.json")):
        print(f"Notice: No completed daily evaluations found in {daily_eval_dir}.")
        print("Gate A cannot be evaluated without completed daily evaluations.")
        return

    daily_records = []
    for f in sorted(daily_eval_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as jf:
            daily_records.append(json.load(jf))

    report = evaluate_gate_a_full(
        daily_records=daily_records,
        expected_protocol_hash=protocol.sha256_hash,
        target_rate=protocol.hypothesis.target_r100_calendar,
        min_calendar_days=protocol.gates.gate_a.min_calendar_days,
        min_quote_hours=protocol.gates.gate_a.min_quote_hours,
        min_active_markets=protocol.gates.gate_a.min_active_markets,
        min_market_coverage_ratio=protocol.gates.gate_a.min_market_coverage_ratio,
        max_single_market_pnl_share=protocol.gates.gate_a.max_single_market_pnl_share,
        n_bootstrap=protocol.gates.gate_a.bootstrap_samples,
    )

    print("=" * 60)
    print(f"Gate A Formal Evaluation Report — Protocol {protocol.protocol_id}")
    print(f"Protocol SHA-256: {protocol.sha256_hash}")
    print("=" * 60)
    print(f"Criteria Verification Checklist:")
    print(f"  [{'PASS' if report.total_days >= protocol.gates.gate_a.min_calendar_days else 'FAIL'}] Calendar Days: {report.total_days} (min {protocol.gates.gate_a.min_calendar_days})")
    sim_pass = report.total_simulated_quote_hours >= Decimal(str(protocol.gates.gate_a.min_quote_hours))
    print(f"  [{'PASS' if sim_pass else 'FAIL'}] Simulated Quote-Hours: {report.total_simulated_quote_hours:.1f} (min {protocol.gates.gate_a.min_quote_hours})")
    print(f"  [INFO] Actual Quote-Hours (Shadow mode): {report.total_actual_quote_hours:.1f} (real CLOB orders = 0.0)")
    print(f"  [{'PASS' if sim_pass else 'FAIL'}] Quote-Hours (Legacy): {report.total_quote_hours:.1f}")
    print(f"  [{'PASS' if report.active_markets_count >= protocol.gates.gate_a.min_active_markets else 'FAIL'}] Active Markets: {report.active_markets_count} (min {protocol.gates.gate_a.min_active_markets})")
    print(f"  [{'PASS' if report.min_market_coverage >= protocol.gates.gate_a.min_market_coverage_ratio else 'FAIL'}] Min Coverage Ratio: {float(report.min_market_coverage)*100:.2f}% (min {float(protocol.gates.gate_a.min_market_coverage_ratio)*100:.1f}%)")
    print(f"  [{'PASS' if report.max_market_pnl_share <= protocol.gates.gate_a.max_single_market_pnl_share else 'FAIL'}] Max Single Market PnL Share: {float(report.max_market_pnl_share)*100:.2f}% (max {float(protocol.gates.gate_a.max_single_market_pnl_share)*100:.1f}%)")
    print(f"  [{'PASS' if report.protocol_hash_valid else 'FAIL'}] Protocol SHA-256 Integrity: {'VALID' if report.protocol_hash_valid else 'MISMATCH'}")
    print(f"  [{'PASS' if report.book_uncertain_count == 0 else 'FAIL'}] BOOK_UNCERTAIN Count: {report.book_uncertain_count} (required 0)")
    print(f"  [{'PASS' if report.stress_test_passed else 'FAIL'}] Stress Test (20% Adverse Shock): {'PASSED' if report.stress_test_passed else 'FAILED'}")

    print("\nStatistical Return Distribution ($100 Base Capital):")
    print(f"  Point Estimate R_100_calendar: ${report.point_estimate:.2f}/day")
    print(f"  95% CI: [${report.lower_95:.2f}, ${report.upper_95:.2f}] / day")
    print(f"  Target Threshold: ${protocol.hypothesis.target_r100_calendar:.2f}/day")
    print(f"\nFINAL VERDICT: {report.verdict}")
    if report.rejection_reasons:
        print("Rejection Reasons:")
        for reason in report.rejection_reasons:
            print(f"  - {reason}")
    print("=" * 60)

    # Save Gate A verdict artifact
    verdict_artifact = {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.sha256_hash,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": report.verdict,
        "point_estimate": str(report.point_estimate),
        "lower_95": str(report.lower_95),
        "upper_95": str(report.upper_95),
        "total_days": report.total_days,
        "total_quote_hours": str(report.total_quote_hours),
        "total_simulated_quote_hours": str(report.total_simulated_quote_hours),
        "total_actual_quote_hours": str(report.total_actual_quote_hours),
        "active_markets_count": report.active_markets_count,
        "min_market_coverage": str(report.min_market_coverage),
        "max_market_pnl_share": str(report.max_market_pnl_share),
        "protocol_hash_valid": report.protocol_hash_valid,
        "book_uncertain_count": report.book_uncertain_count,
        "stress_test_passed": report.stress_test_passed,
        "rejection_reasons": report.rejection_reasons,
    }
    verdict_file = storage_path / "gate_a_verdict.json"
    with open(verdict_file, "w", encoding="utf-8") as vf:
        json.dump(verdict_artifact, vf, indent=2)
    print(f"Saved Gate A verdict artifact to {verdict_file}")


if __name__ == "__main__":
    main()
