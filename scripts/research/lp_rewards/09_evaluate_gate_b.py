import datetime
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.evaluation import (
    compute_block_bootstrap_ci,
    determine_gate_b_verdict,
)
from polyflip.research.lp_rewards.protocol import load_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_gate_b")


def calculate_max_drawdown(daily_pnls: List[Decimal], initial_capital: Decimal = Decimal("100.00")) -> Decimal:
    """Calculate maximum peak-to-trough drawdown from equity curve."""
    if not daily_pnls:
        return Decimal("0.0")

    equity = initial_capital
    peak = equity
    max_dd = Decimal("0.0")

    for pnl in daily_pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        if peak > Decimal("0.0"):
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    return max_dd


def main():
    isolated_wallet = os.getenv("LP_ISOLATED_WALLET_ADDRESS")
    zero_address = "0x0000000000000000000000000000000000000000"
    if (
        not isolated_wallet
        or not isolated_wallet.strip()
        or isolated_wallet.strip().lower() == zero_address
    ):
        logger.error("LP_ISOLATED_WALLET_ADDRESS is missing or invalid. Rejecting Gate B.")
        print("TARGET_REJECTED: LP_ISOLATED_WALLET_ADDRESS is missing or invalid. Rejecting Gate B.")
        sys.exit(1)
    isolated_wallet = isolated_wallet.strip()

    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)

    logger.info(f"Gate B Evaluator for protocol: {protocol.protocol_id} (SHA-256: {protocol.sha256_hash})")
    logger.info(f"Required Live Days: {protocol.gates.gate_b.min_live_days} (Days 8-21)")
    logger.info(f"Target Confirmation Threshold: ${protocol.hypothesis.target_r100_calendar:.2f}/day (lower 95% bound)")
    logger.info(f"Max Drawdown Limit: {float(protocol.gates.gate_b.max_drawdown_limit)*100:.1f}%")
    logger.info(f"Max Prediction Error: {float(protocol.gates.gate_b.max_reward_prediction_error)*100:.1f}%")

    # 1. Load live daily evaluations
    live_eval_dir = storage_path / "daily_evaluations_live"
    eval_files = sorted(live_eval_dir.glob("*.json")) if live_eval_dir.exists() else []

    if not eval_files:
        print("DATA_INSUFFICIENT: No live daily evaluations found.")
        sys.exit(1)

    daily_records: List[Dict[str, Any]] = []
    daily_pnls: List[Decimal] = []
    daily_dates = set()
    invalid_eval_reason = None

    for f in eval_files:
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            daily_records.append(data)
            daily_pnls.append(Decimal(str(data.get("net_pnl", "0.0"))))
            rec_date = data.get("date")
            if not rec_date:
                invalid_eval_reason = f"Missing date in evaluation file {f.name}"
            elif rec_date in daily_dates:
                invalid_eval_reason = f"Duplicate evaluation date {rec_date} in {f.name}"
            else:
                daily_dates.add(rec_date)

            rec_hash = data.get("protocol_hash")
            if not rec_hash or rec_hash != protocol.sha256_hash:
                invalid_eval_reason = f"Protocol hash mismatch in {f.name}: {rec_hash} != {protocol.sha256_hash}"

    if invalid_eval_reason:
        logger.error(f"[GATE B BLOCKED] {invalid_eval_reason}")
        print(f"TARGET_REJECTED: {invalid_eval_reason}")
        sys.exit(1)

    total_live_days = len(daily_records)
    min_days = protocol.gates.gate_b.min_live_days

    # 2. Check latest reward prediction error from reconciliation reports
    recon_dir = storage_path / "reconciliation"
    reward_reports = sorted(recon_dir.glob("rewards_*.json")) if recon_dir.exists() else []
    if not reward_reports:
        logger.error("No reconciliation reports found. Rejecting Gate B.")
        print("DATA_INSUFFICIENT: No reconciliation reports found. Rejecting Gate B.")
        sys.exit(1)

    with open(reward_reports[-1], "r", encoding="utf-8") as rf:
        rdata = json.load(rf)
        rep_hash = rdata.get("protocol_hash")
        rep_wallet = rdata.get("wallet_address")
        if rep_hash != protocol.sha256_hash:
            logger.error(f"Reconciliation report hash mismatch: {rep_hash} != {protocol.sha256_hash}. Rejecting Gate B.")
            sys.exit(1)
        elif not rep_wallet or not rep_wallet.strip() or rep_wallet.strip().lower() == zero_address:
            logger.error("Reconciliation report wallet_address is missing or zero address. Rejecting Gate B.")
            sys.exit(1)
        elif rep_wallet.strip().lower() != isolated_wallet.lower():
            logger.error(f"Reconciliation report wallet mismatch: {rep_wallet} != {isolated_wallet}. Rejecting Gate B.")
            sys.exit(1)
        else:
            mean_prediction_error = Decimal(str(rdata.get("mean_error_ratio", "1.0")))

    # 3. Calculate bootstrap CI and drawdown
    point, lower, upper = compute_block_bootstrap_ci(
        daily_pnls, n_bootstrap=10000, seed=42
    )
    max_drawdown = calculate_max_drawdown(
        daily_pnls, initial_capital=protocol.capital_allocation.allocated_working_capital
    )

    # 4. Determine Gate B verdict
    verdict = determine_gate_b_verdict(
        point_est=point,
        lower_95=lower,
        upper_95=upper,
        max_drawdown=max_drawdown,
        mean_prediction_error=mean_prediction_error,
        target_rate=protocol.hypothesis.target_r100_calendar,
        max_drawdown_limit=protocol.gates.gate_b.max_drawdown_limit,
        max_error_limit=protocol.gates.gate_b.max_reward_prediction_error,
        total_live_days=total_live_days,
        min_live_days=min_days,
    )

    print("=" * 60)
    print(f"Gate B Live Calibration Evaluation Report — Protocol {protocol.protocol_id}")
    print("=" * 60)
    print(f"Live Observation Window: {total_live_days} days (required min {min_days} live days)")
    print(f"Equity Curve Max Drawdown: {float(max_drawdown)*100:.2f}% (Limit: {float(protocol.gates.gate_b.max_drawdown_limit)*100:.1f}%)")
    print(f"Mean Reward Prediction Error: {float(mean_prediction_error)*100:.2f}% (Limit: {float(protocol.gates.gate_b.max_reward_prediction_error)*100:.1f}%)")
    print("\nLive Return Statistics ($100 Base Capital):")
    print(f"  Point Estimate R_100_calendar: ${point:.2f}/day")
    print(f"  95% CI: [${lower:.2f}, ${upper:.2f}] / day")
    print(f"  Target Confirmation Threshold: ${protocol.hypothesis.target_r100_calendar:.2f}/day (lower bound)")
    print(f"\nFINAL GATE B VERDICT: {verdict}")
    print("=" * 60)

    # Save Gate B verdict artifact
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    artifact = {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.sha256_hash,
        "evaluated_at": now_utc.isoformat(),
        "verdict": verdict,
        "total_live_days": total_live_days,
        "min_live_days": min_days,
        "point_estimate": str(point),
        "lower_95": str(lower),
        "upper_95": str(upper),
        "max_drawdown": str(max_drawdown),
        "mean_prediction_error": str(mean_prediction_error),
    }

    verdict_file = storage_path / "gate_b_verdict.json"
    with open(verdict_file, "w", encoding="utf-8") as vf:
        json.dump(artifact, vf, indent=2)
    logger.info(f"Saved Gate B verdict artifact to {verdict_file}")


if __name__ == "__main__":
    main()
