import asyncio
import datetime
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import httpx

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from polyflip.research.lp_rewards.protocol import load_protocol
from polyflip.research.lp_rewards.reward_calibration import RewardCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconcile_rewards")


async def main():
    protocol = load_protocol()
    storage_path = Path(protocol.data_storage.root_path)
    wallet_address = os.getenv("LP_ISOLATED_WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")

    calibrator = RewardCalibrator(
        max_allowed_error_ratio=protocol.gates.gate_b.max_reward_prediction_error
    )
    logger.info(f"Reconciling rewards for protocol: {protocol.protocol_id} (Wallet: {wallet_address})")
    logger.info(f"Max allowed prediction error threshold: {float(protocol.gates.gate_b.max_reward_prediction_error) * 100:.0f}%")

    api_key = os.getenv("LP_RELAYER_API_KEY") or os.getenv("POLY_API_KEY")
    # 1. Fetch actual user rewards from Polymarket API
    async with httpx.AsyncClient(timeout=30.0) as client:
        actual_rewards = await calibrator.fetch_actual_user_rewards(wallet_address, client, api_key=api_key)

    logger.info(f"Retrieved {len(actual_rewards)} actual reward payout records from Polymarket API.")

    # 2. Load daily projected evaluations
    eval_dir = storage_path / "daily_evaluations_live"
    if not eval_dir.exists() or not list(eval_dir.glob("*.json")):
        logger.error("DATA_INSUFFICIENT: No live daily evaluations found. Shadow data is not permitted for Gate B.")
        sys.exit(1)

    projected_by_date: Dict[str, Decimal] = {}
    if eval_dir.exists():
        for f in sorted(eval_dir.glob("*.json")):
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                date_key = f.stem
                projected_reward = Decimal(str(data.get("total_rewards_accrued", "0.0")))
                projected_by_date[date_key] = projected_reward

    # Reconcile each payout against projected
    payout_records: List[Dict[str, Any]] = []
    
    # Sum actual rewards by date
    actual_by_date: Dict[str, Decimal] = {}
    for item in actual_rewards:
        raw_date = item.get("date", "")
        date_val = raw_date[:10] if "T" in raw_date else raw_date
        if not date_val: continue
        actual_val = Decimal(str(item.get("earnings", "0.0")))
        actual_by_date[date_val] = actual_by_date.get(date_val, Decimal("0.0")) + actual_val
        
    for date_val, actual_val in actual_by_date.items():
        if date_val not in projected_by_date:
            logger.error(f"DATA_INSUFFICIENT: No live projection found for date {date_val}")
            sys.exit(1)
        projected_val = projected_by_date[date_val]
        err = calibrator.record_observation("ALL_MARKETS", projected_val, actual_val)
        payout_records.append({
            "condition_id": "ALL_MARKETS",
            "date": date_val,
            "actual_reward": str(actual_val),
            "projected_reward": str(projected_val),
            "error_ratio": str(err),
        })

    if not actual_rewards:
        logger.error("DATA_INSUFFICIENT: No remote API reward records returned.")
        sys.exit(1)

    passes, mean_error = calibrator.evaluate_gate_b_accuracy()

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    recon_report = {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.sha256_hash,
        "reconciled_at": now_utc.isoformat(),
        "wallet_address": wallet_address,
        "records_count": len(calibrator.history),
        "mean_error_ratio": str(mean_error),
        "threshold": str(protocol.gates.gate_b.max_reward_prediction_error),
        "passes_gate_b_criterion": passes,
        "observations": [
            {
                "condition_id": h.get("condition_id", ""),
                "projected": str(h["projected"]),
                "actual": str(h["actual"]),
                "error_ratio": str(h["error_ratio"]),
            }
            for h in calibrator.history
        ],
    }

    recon_dir = storage_path / "reconciliation"
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_file = recon_dir / f"rewards_{now_utc.strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(recon_report, f, indent=2)

    logger.info(f"Mean reward prediction error: {float(mean_error) * 100:.2f}% (Threshold: {float(protocol.gates.gate_b.max_reward_prediction_error) * 100:.0f}%)")
    logger.info(f"Gate B Accuracy Status: {'PASSED' if passes else 'FAILED'}")
    logger.info(f"Saved reward reconciliation report to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
